"""Counterfactual edits of the forward guidance sentence (protocol §6.7, test B of H3).

A model that knows the period of a statement could reproduce what it remembers instead of
reading the text. Here the forward guidance sentence of real statements is rewritten three
times (hawkish, dovish, neutral), every other sentence unchanged, and Phi-4 reads each version
with the chosen procedure. If the score follows the text, the forward guidance value moves
with the edit; if it follows memory, it stays near the original reading.

- ``draw_sample``: 10 statements from 2000–2018 and 10 from 2019 to the cutoff, among those
  whose original reading has a forward guidance value, plus every such statement after the
  cutoff (seed fixed in the protocol). No market data is used.
- The edits are hand-written in ``data/labels/fomc_counterfactual_v1.json``; ``edited_docs``
  checks that each original sentence occurs exactly once and builds the edited texts.

Usage: ``python -m text_signals.sources.fomc.counterfactual {run,report}``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from text_signals.encoders import split_sentences
from text_signals.llm import ExtractionRecord, read_cache
from text_signals.sources.fomc.extract import SampleDoc, all_docs, cache_path, run_local
from text_signals.sources.fomc.extraction import STANCE_FIELDS, StatementExtraction, tone_score

SEED = 20261108  # protocol §6.7, B
CUTOFF = date(2024, 6, 30)
STRATA = (
    ("2000-2018", date(2000, 1, 1), date(2018, 12, 31), 10),
    ("2019-cutoff", date(2019, 1, 1), CUTOFF, 10),
)
EDITS = Path("data/labels/fomc_counterfactual_v1.json")
EDITS_VERSION = "fomc-counterfactual-v1"
VERSIONS = ("hawkish", "dovish", "neutral")
OUT = Path("data/interim/fomc/counterfactual")
MODEL = "phi-4"


def _normalise(text: str) -> str:
    return " ".join(text.split()).lower()


def guidance_text(text: str, evidence: str) -> str | None:
    """The passage of ``text`` to rewrite: the sentence that holds the forward guidance evidence.

    Three rules, in order (protocol §6.7, B): (1) the one sentence that contains the span;
    (2) when the span runs over consecutive sentences, the block from the first to the last,
    verbatim; (3) when the span is not verbatim, the one sentence sharing the most words with
    it. ``None`` if no rule gives a single answer.
    """
    span = _normalise(evidence)
    sentences = split_sentences(text)
    hits = [s for s in sentences if span in _normalise(s)]
    if len(hits) == 1:
        return hits[0]
    for length in range(2, len(sentences) + 1):  # the shortest block first
        for i in range(len(sentences) - length + 1):
            j = i + length - 1
            if span in _normalise(" ".join(sentences[i : j + 1])):
                start = text.index(sentences[i])
                return text[start : text.index(sentences[j], start) + len(sentences[j])]
    words = set(span.split())
    overlap = [len(words & set(_normalise(s).split())) for s in sentences]
    best = max(overlap, default=0)
    return sentences[overlap.index(best)] if best and overlap.count(best) == 1 else None


def draw_sample(scores: pd.DataFrame, seed: int = SEED) -> list[tuple[str, str]]:
    """(doc_id, stratum) for the B sample: the two strata before the cutoff drawn at random,
    every statement after the cutoff kept. ``scores`` is the Phi-4 score table."""
    rng = np.random.default_rng(seed)
    s = scores[scores["llm_guidance"].notna()].copy()
    days = pd.to_datetime(s["release_date"]).dt.date
    out: list[tuple[str, str]] = []
    for name, start, end, k in STRATA:
        pool = sorted(s.loc[(days >= start) & (days <= end), "doc_id"])
        picked = rng.choice(len(pool), size=k, replace=False)
        out += [(pool[i], name) for i in sorted(picked)]
    out += [(d, "after-cutoff") for d in sorted(s.loc[days > CUTOFF, "doc_id"])]
    return out


def original_readings() -> dict[str, ExtractionRecord]:
    """Phi-4's original reading of every statement, as used for the scores (first answer)."""
    from text_signals.sources.fomc.extract import procedure
    from text_signals.sources.fomc.extract_eval import load_reader

    return load_reader(MODEL, all_docs(), procedure(1), first=True)


def load_edits(path: Path = EDITS) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["version"] != EDITS_VERSION:
        raise ValueError(f"unexpected edits version {data['version']}")
    edits: list[dict[str, str]] = data["edits"]
    return edits


def edited_docs(docs: list[SampleDoc], edits: list[dict[str, str]]) -> list[SampleDoc]:
    """Three edited versions of each statement; doc_id ``<doc_id>__<version>``."""
    by_id = {d.doc_id: d for d in docs}
    out = []
    for e in edits:
        d = by_id[e["doc_id"]]
        if d.text.count(e["original"]) != 1:
            raise ValueError(f"{d.doc_id}: original sentence not found exactly once")
        for v in VERSIONS:
            if not e[v].strip() or e[v] == e["original"]:
                raise ValueError(f"{d.doc_id}: {v} edit is empty or unchanged")
            text = d.text.replace(e["original"], e[v])
            out.append(SampleDoc(d.code, f"{d.doc_id}__{v}", d.release_date, text))
    return out


def _values(record: ExtractionRecord | None) -> dict[str, float | None]:
    if record is None or record.output is None:
        return {"fg": None, "tone": None, **{f: None for f in STANCE_FIELDS}}
    ext = StatementExtraction.model_validate(record.output)
    fields = {f: getattr(ext, f).value for f in STANCE_FIELDS}
    fg = ext.forward_guidance.value
    return {"fg": None if fg is None else float(fg), "tone": tone_score(ext), **fields}


def readings_table(
    edits: list[dict[str, str]],
    original: dict[str, ExtractionRecord],
    edited: dict[str, ExtractionRecord],
) -> pd.DataFrame:
    """One row per statement and version (original included), with FG, tone and each field."""
    rows = []
    for e in edits:
        for v in ("original", *VERSIONS):
            rec = (
                original.get(e["doc_id"]) if v == "original" else edited.get(f"{e['doc_id']}__{v}")
            )
            rows.append(
                {"doc_id": e["doc_id"], "stratum": e["stratum"], "version": v, **_values(rec)}
            )
    return pd.DataFrame(rows)


def run(base_url: str) -> list[ExtractionRecord]:
    """Phi-4 reads the 3 × 38 edited texts with the chosen procedure (a single pass, which is
    the first answer), cached apart from every other reading."""
    return run_local(base_url, edited_docs(all_docs(), load_edits()), OUT)


def edited_readings() -> dict[str, ExtractionRecord]:
    return {
        r.doc_id: r
        for r in read_cache(cache_path(MODEL, OUT)).values()
        if r.procedure == "single-pass"
    }


def recognised_by_probe() -> set[str]:
    """Statements the dating probe placed within one year of their release (§6.7)."""
    from text_signals.sources.fomc.lookahead import OUT as LOOKAHEAD

    docs = {d.doc_id: d.release_date.year for d in all_docs()}
    out = set()
    for r in read_cache(LOOKAHEAD / "date_probe" / f"{MODEL}.jsonl").values():
        if r.output is not None and abs(int(r.output["year"]) - docs[r.doc_id]) <= 1:
            out.add(r.doc_id)
    return out


def wide(readings: pd.DataFrame) -> pd.DataFrame:
    """One row per statement: FG, tone and the other stance fields for each version."""
    w = readings.pivot(index=["doc_id", "stratum"], columns="version")
    w.columns = [f"{col[0]}_{col[1]}" for col in w.columns.to_flat_index()]
    return w.reset_index()


def bootstrap_diff(
    a: np.ndarray, b: np.ndarray, *, reps: int = 10_000, seed: int = SEED
) -> tuple[float, float, float]:
    """mean(a) − mean(b) and its percentile 95% interval, resampling each group's statements."""
    rng = np.random.default_rng(seed)
    draws = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(reps)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def measures(w: pd.DataFrame) -> dict[str, float]:
    """B1–B3 of §6.7 on a set of statements. A missing FG value never counts as following."""
    h, d, n, o = (w[f"fg_{v}"] for v in ("hawkish", "dovish", "neutral", "original"))
    others = [f for f in STANCE_FIELDS if f != "forward_guidance"]
    spill = []
    for v in ("hawkish", "dovish"):
        for f in others:
            spill.append((w[f"{f}_{v}"] - w[f"{f}_original"]).abs())
    return {
        "n": float(len(w)),
        "B1_hawk_above_dove": float((h > d).mean()),
        "B2_hawk_ge_1": float((h >= 1).mean()),
        "B2_dove_le_m1": float((d <= -1).mean()),
        "B3_neutral_changed": float((n != o).mean()),
        "B3_spillover_mean_abs": float(pd.concat(spill).mean()),
        "fg_original_mean": float(o.mean()),
        "fg_hawkish_mean": float(h.mean()),
        "fg_dovish_mean": float(d.mean()),
        "fg_gap_mean": float((h - d).mean()),
        "tone_gap_mean": float((w["tone_hawkish"] - w["tone_dovish"]).mean()),
        "missing_fg_edited": float(pd.concat([h, d, n]).isna().sum()),
    }


def report(w: pd.DataFrame, flagged: set[str], recognised: set[str]) -> dict[str, dict[str, float]]:
    """B1–B4 on all statements and on those without other rate guidance (§6.7)."""
    out: dict[str, dict[str, float]] = {}
    for name, sub in (("all", w), ("without_other_guidance", w[~w["doc_id"].isin(flagged)])):
        before = sub[sub["stratum"] != "after-cutoff"]
        after = sub[sub["stratum"] == "after-cutoff"]
        gap_b = (before["fg_hawkish"] - before["fg_dovish"]).fillna(0).to_numpy(dtype=float)
        gap_a = (after["fg_hawkish"] - after["fg_dovish"]).fillna(0).to_numpy(dtype=float)
        diff, lo, hi = bootstrap_diff(gap_b, gap_a)
        out[f"{name}/before"] = measures(before)
        out[f"{name}/after"] = measures(after)
        out[f"{name}/B4"] = {"gap_before_minus_after": diff, "ci_low": lo, "ci_high": hi}
    rec = w[(w["stratum"] != "after-cutoff") & w["doc_id"].isin(recognised)]
    unrec = w[(w["stratum"] != "after-cutoff") & ~w["doc_id"].isin(recognised)]
    out["before/recognised"] = measures(rec)
    out["before/not_recognised"] = measures(unrec)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=["run", "report"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    if args.cmd == "run":
        records = run(args.base_url)
        print(f"{len(records)} records, {sum(r.error is not None for r in records)} invalid")
        return
    edits = load_edits()
    flagged = {e["doc_id"] for e in edits if e.get("other_rate_guidance")}
    w = wide(readings_table(edits, original_readings(), edited_readings()))
    w.to_csv(OUT / "readings_wide.csv", index=False)
    for name, m in report(w, flagged, recognised_by_probe()).items():
        print(name)
        for k, v in m.items():
            print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
