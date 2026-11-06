"""Evaluation of the statement extraction on the 65-statement sample (protocol §6.6).

Three checks, none of which needs a hand label:

1. **Decision against the data.** The rate decision and its size are compared with the federal
   funds target from FRED: ``DFEDTAR`` (a single target) until 15 December 2008, then the upper
   bound of the range, ``DFEDTARU``. The change is the target on the day after the release
   minus the target on the last day before it, which absorbs FRED's inconsistent dating of the
   effective day.
2. **Evidence.** Every evidence span must appear verbatim in the statement; one that does not is
   a hallucination.
3. **Agreement with the reference reader** (Claude), field by field (``text_signals.evaluation``).

Usage: ``python -m text_signals.sources.fomc.extract_eval <model> [<reference model>]``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from text_signals.encoders import split_sentences
from text_signals.evaluation import FieldAgreement, field_agreement
from text_signals.fetch import Fetcher
from text_signals.llm import ExtractionRecord, read_cache
from text_signals.market import fetch_fred_series
from text_signals.sources.fomc.extract import (
    PROMPT,
    PROMPT_V3,
    SampleDoc,
    cache_path,
    procedure,
    sample_docs,
)
from text_signals.sources.fomc.extraction import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_V2,
    STANCE_FIELDS,
    StatementExtraction,
    StatementExtractionV2,
    evidence_not_in_text,
    to_spans,
    tone_score,
)

RANGE_START = date(2008, 12, 16)  # first day of DFEDTARU
ORDINAL = (*STANCE_FIELDS,)
CATEGORICAL = ("policy_action", "risk_balance")
DISSENTS = ("dissents_tighter", "dissents_easier", "dissents_other")


def policy_target(single: pd.Series, upper: pd.Series) -> pd.Series:
    """The federal funds target: the single target, then the upper bound of the range."""
    before = single.dropna()[single.dropna().index < RANGE_START]
    after = upper.dropna()[upper.dropna().index >= RANGE_START]
    joined: pd.Series = pd.concat([before, after])
    return joined


def target_change_bp(days: Sequence[date], target: pd.Series) -> list[int | None]:
    out: list[int | None] = []
    for day in days:
        pre = target[target.index < day]
        post = target[target.index <= day + timedelta(days=1)]
        if pre.empty or post.empty:
            out.append(None)
        else:
            out.append(round((float(post.iloc[-1]) - float(pre.iloc[-1])) * 100))
    return out


def decision_from_change(change: int | None) -> str | None:
    if change is None:
        return None
    return "raise" if change > 0 else "cut" if change < 0 else "hold"


def flat(extraction: StatementExtraction) -> dict[str, Any]:
    """One row per statement: each field's value, the dissent counts and the tone score."""
    row: dict[str, Any] = {f: getattr(extraction, f).value for f in (*ORDINAL, *CATEGORICAL)}
    row["change_bp"] = extraction.policy_action.change_bp
    for kind in ("tighter", "easier", "other"):
        row[f"dissents_{kind}"] = sum(d.preferred == kind for d in extraction.dissents)
    row["llm_tone"] = tone_score(extraction)
    return row


def load_reader(
    model: str, docs: Sequence[SampleDoc], proc: str = "single-pass", *, numbered: bool = False
) -> dict[str, ExtractionRecord]:
    """The records of one reader for the sample, with the current prompt, schema and procedure.

    With ``numbered``, the schema v2 records (prompt v3), turned into v1 form by ``to_spans``.
    """
    cache = read_cache(cache_path(model))
    wanted = {d.doc_id: d for d in docs}
    prompt, schema = (
        (PROMPT_V3.version, SCHEMA_VERSION_V2)
        if numbered
        else (
            PROMPT.version,
            SCHEMA_VERSION,
        )
    )
    out = {
        r.doc_id: r
        for r in cache.values()
        if r.doc_id in wanted
        and r.prompt_version == prompt
        and r.schema_version == schema
        and r.procedure == proc
    }
    if numbered:
        for doc_id, r in out.items():
            if r.output is not None:
                v2 = StatementExtractionV2.model_validate(r.output)
                v1 = to_spans(v2, split_sentences(wanted[doc_id].text))
                out[doc_id] = r.model_copy(update={"output": v1.model_dump(mode="json")})
    return out


def reader_table(records: dict[str, ExtractionRecord], docs: Sequence[SampleDoc]) -> pd.DataFrame:
    rows = []
    for d in docs:
        rec = records.get(d.doc_id)
        row: dict[str, Any] = {"code": d.code, "doc_id": d.doc_id, "valid": False}
        if rec is not None and rec.output is not None:
            ext = StatementExtraction.model_validate(rec.output)
            row |= flat(ext) | {"valid": True}
            row["hallucinated"] = ",".join(evidence_not_in_text(ext, d.text))
            row["n_evidence"] = len(ext.dissents) + sum(
                getattr(ext, f).evidence is not None for f in (*ORDINAL, *CATEGORICAL)
            )
        if rec is not None and rec.rejected:
            row["n_repairs"] = len(rec.rejected)
            row["first_problems"] = len(rec.rejected[0].problems)
            row["problems_left"] = len(rec.problems_left)
            first = StatementExtraction.model_validate_json(rec.rejected[0].raw_output)
            if row["valid"]:
                before, after = flat(first), flat(StatementExtraction.model_validate(rec.output))
                moved = [f for f in (*ORDINAL, *CATEGORICAL) if before[f] != after[f]]
                row["values_moved"] = ",".join(moved)
        if rec is not None:
            row |= {
                "latency_s": rec.latency_s,
                "prompt_tokens": rec.prompt_tokens,
                "completion_tokens": rec.completion_tokens,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> list[FieldAgreement]:
    both = a[a["valid"]].merge(b[b["valid"]], on="code", suffixes=("_a", "_b"))

    def col(frame: pd.DataFrame, name: str) -> list[Any]:
        return [None if pd.isna(v) else v for v in frame[name]]

    out = [
        field_agreement(f, col(both, f"{f}_a"), col(both, f"{f}_b"), ordinal=True)
        for f in (*ORDINAL, *DISSENTS)
    ]
    out += [
        field_agreement(f, col(both, f"{f}_a"), col(both, f"{f}_b"), ordinal=False)
        for f in CATEGORICAL
    ]
    return out


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def _num(x: float | None) -> str:
    return "–" if x is None else f"{x:.2f}"


def report(
    model: str,
    reference: str | None,
    data_dir: Path = Path("data"),
    *,
    repairs: int = 0,
    numbered: bool = False,
) -> str:
    """``model`` is read with ``repairs`` repair turns (0: single pass); the reference, single."""
    docs = sample_docs()
    with Fetcher(data_dir / "raw", delay_s=2.0) as fetcher:
        target = policy_target(
            fetch_fred_series(fetcher, "DFEDTAR"), fetch_fred_series(fetcher, "DFEDTARU")
        )
    days = [d.release_date for d in docs]
    truth = pd.DataFrame(
        {"code": [d.code for d in docs], "true_change_bp": target_change_bp(days, target)}
    )
    truth["true_decision"] = truth["true_change_bp"].map(decision_from_change)

    lines: list[str] = []
    tables = {}
    readers = [(model, procedure(repairs), numbered)] + (
        [(reference, "single-pass", False)] if reference else []
    )
    for name, proc, num in readers:
        t = reader_table(load_reader(name, docs, proc, numbered=num), docs).merge(truth, on="code")
        tables[name] = t
        suffix = ("" if proc == "single-pass" else f"_repair{repairs}") + (
            "_numbered" if num else ""
        )
        out = (
            data_dir / "processed" / "fomc" / f"extraction_eval_{cache_path(name).stem}{suffix}.csv"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(out, index=False)
        valid = t[t["valid"]]
        decided = valid[valid["true_decision"].notna()]
        hold_ok = (decided["policy_action"] == decided["true_decision"]) | (
            (decided["true_decision"] == "hold") & (decided["policy_action"] == "no_rate_decision")
        )
        size_ok = decided["change_bp"].fillna(0) == decided["true_change_bp"]
        n_spans = int(valid["n_evidence"].sum())
        n_bad = int(
            valid["hallucinated"]
            .fillna("")
            .str.split(",")
            .map(lambda x: len([s for s in x if s]))
            .sum()
        )
        lines += [
            f"### {name} ({proc}{', evidence by sentence number' if num else ''})",
            "",
            f"- outputs: {len(t)} statements, {int((~t['valid']).sum())} failed validation",
            f"- decision vs FRED target: {hold_ok.sum()}/{len(decided)} right direction, "
            f"{size_ok.sum()}/{len(decided)} right size in bp",
            f"- evidence spans not found verbatim: {n_bad}/{n_spans}",
        ]
        if t["latency_s"].notna().any():
            lines.append(
                f"- latency {t['latency_s'].median():.1f} s median; tokens "
                f"{t['prompt_tokens'].median():.0f} in, {t['completion_tokens'].median():.0f} "
                "out (median)"
            )
        if "n_repairs" in t:
            rep = t[t["n_repairs"].notna()]
            moved = (
                rep["values_moved"].fillna("").map(lambda x: len([v for v in x.split(",") if v]))
            )
            lines.append(
                f"- repair loop: {len(rep)} statements sent back; spans flagged on the first "
                f"answer {int(rep['first_problems'].sum())}, left after repair "
                f"{int(rep['problems_left'].sum())}; field values changed by the repair "
                f"{int(moved.sum())}"
            )
        wrong = decided[~(hold_ok & size_ok)]
        for r in wrong.itertuples():
            lines.append(
                f"  - {r.code}: read {r.policy_action} {r.change_bp}, "
                f"FRED {r.true_decision} {r.true_change_bp}"
            )
        lines.append("")
    if reference:
        lines += [
            f"### Agreement: {model} (a) vs {reference} (b)",
            "",
            "| Field | Same presence | Both filled | Only a | Only b | Exact | Within 1 | MAE "
            "| κ |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for f in agreement(tables[model], tables[reference]):
            lines.append(
                f"| {f.field} | {_pct(f.presence_agreement)} | {f.both_filled} | {f.only_a} "
                f"| {f.only_b} | {_pct(f.exact)} | {_pct(f.within_one)} | {_num(f.mae)} "
                f"| {_num(f.kappa)} |"
            )
        a, b = tables[model], tables[reference]
        tone = a[["code", "llm_tone"]].merge(b[["code", "llm_tone"]], on="code").dropna()
        lines += [
            "",
            f"Tone score (mean of the stance fields): correlation "
            f"{tone['llm_tone_x'].corr(tone['llm_tone_y']):.2f} on {len(tone)} statements, "
            f"mean gap {(tone['llm_tone_x'] - tone['llm_tone_y']).mean():+.2f}.",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("reference", nargs="?")
    parser.add_argument("--repairs", type=int, default=0)
    parser.add_argument("--numbered", action="store_true", help="the model's schema v2 run")
    args = parser.parse_args()
    print(report(args.model, args.reference, repairs=args.repairs, numbered=args.numbered))


if __name__ == "__main__":
    main()
