"""Look-ahead tests of the LLM score (protocol §6.7, H3).

A model trained on text written after a statement may know what followed it. Three checks:

- **cutoff split**: does the score's effect differ after the model's training cutoff?
- **anonymisation**: remove dates and names, read again, and compare the scores;
- **recognition probe**: can the model date an anonymised statement?

``anonymise`` rewrites a statement: dates with a month name become ``[date]``, a month alone
``[month]``, four-digit years ``[year]``, and the names of FOMC members (read from the
statements' own vote paragraphs and "Chairman …" phrases) ``[member]``. Policy wording, numbers
and rates stay as they are.

Usage: ``python -m text_signals.sources.fomc.lookahead {anonymised-run,probe-run,report}``.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from text_signals.llm import (
    Decoding,
    ExtractionRecord,
    LLMClient,
    Prompt,
    extract_all,
    read_cache,
)
from text_signals.sources.fomc.extract import SampleDoc, all_docs, cache_path, run_local

MONTHS = "January|February|March|April|June|July|August|September|October|November|December"
_NAME = r"[A-Z][a-z]+(?: [A-Z]\.)?(?: [A-Z][a-z]*[A-Z]?[a-z]+(?:-[A-Z][a-z]+)?)+"
_NOT_NAMES = {"Vice Chairman", "Vice Chair", "Federal Reserve", "Open Market"}
# a vote sentence ends at a period not preceded by a lone capital (an initial: "William J.")
_VOTE = re.compile(r"Voting (?:for|against).*?(?:(?<!\b[A-Z])\.(?=\s|$)|$)")
_CHAIR = re.compile(r"Chairman (" + _NAME + ")")


def member_names(texts: Iterable[str]) -> set[str]:
    """Full names of FOMC members as the statements write them in their vote paragraphs."""
    names: set[str] = set()
    for text in texts:
        flat = re.sub(r"\s+", " ", text).replace(", Jr.", "")
        for sentence in _VOTE.findall(flat):
            names |= {n for n in re.findall(_NAME, sentence) if n not in _NOT_NAMES}
        names |= set(_CHAIR.findall(flat))
    return names


def _surname(name: str) -> str:
    return name.split()[-1]


def anonymise(text: str, names: set[str]) -> str:
    out = text
    # names: full names first, then surnames alone; "Jr." goes with its name
    for name in sorted(names, key=len, reverse=True):
        out = re.sub(re.escape(name) + r"(?:,? Jr\.)?", "[member]", out)
    for surname in sorted({_surname(n) for n in names}, key=len, reverse=True):
        out = re.sub(r"\b" + re.escape(surname) + r"\b", "[member]", out)
    # dates with a month: "January 28, 2015", "January 28", "January 2015"
    out = re.sub(
        r"\b(?:" + MONTHS + r"|May)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?", "[date]", out
    )
    out = re.sub(r"\b(?:" + MONTHS + r"|May),?\s+\d{4}\b", "[date]", out)
    out = re.sub(r"\b(?:" + MONTHS + r")\b", "[month]", out)
    out = re.sub(r"\b(in|since|of|late|early|mid-) May\b", r"\1 [month]", out, flags=re.I)
    out = re.sub(r"\b(?:19|20)\d{2}\b", "[year]", out)
    return out


PROBE_VERSION = "fomc-date-probe-v1"


class DateGuess(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    year: int = Field(
        ge=1990, le=2030, description="The year the statement was most likely released."
    )


DATE_PROBE_V1 = Prompt(
    version=PROBE_VERSION,
    system=(
        "You are shown a statement of the Federal Open Market Committee (FOMC) from which dates "
        "and names have been removed. Give the year in which it was most likely released, as "
        "JSON with one field, 'year'. Answer with your best single guess."
    ),
    user_template="Statement:\n\n{text}",
)

OUT = Path("data/interim/fomc/lookahead")
MODEL = "phi-4"


def anonymised_docs() -> list[SampleDoc]:
    docs = all_docs()
    names = member_names(d.text for d in docs)
    return [SampleDoc(d.code, d.doc_id, d.release_date, anonymise(d.text, names)) for d in docs]


def run_anonymised(base_url: str) -> list[ExtractionRecord]:
    """The chosen procedure's values (a single pass equals the first answer) on the anonymised
    texts, cached apart from the original reading."""
    return run_local(base_url, anonymised_docs(), OUT / "anonymised")


def run_probe(base_url: str) -> list[ExtractionRecord]:
    client = LLMClient(base_url)
    model, model_path = client.model_info()
    docs = anonymised_docs()

    def run(doc_id: str, text: str) -> ExtractionRecord:
        return client.extract(
            doc_id=doc_id,
            text=text,
            prompt=DATE_PROBE_V1,
            schema=DateGuess,
            schema_version=PROBE_VERSION,
            decoding=Decoding(max_tokens=32),
            model=model,
            model_path=model_path,
        )

    return extract_all(
        ((d.doc_id, d.text) for d in docs),
        run,
        model=model,
        prompt_version=DATE_PROBE_V1.version,
        schema_version=PROBE_VERSION,
        cache_path=OUT / "date_probe" / f"{model}.jsonl",
    )


def cutoff_split(table: pd.DataFrame) -> dict[str, float]:
    """β of ``llm_tone`` before and after the training cutoff, and the HAC test that they are
    equal (an interaction in the primary regression)."""
    from text_signals.sources.fomc import study

    return _split_at_cutoff(study.primary_sample(table))


def _split_at_cutoff(sample: pd.DataFrame) -> dict[str, float]:
    from text_signals.event_study import fit
    from text_signals.sources.fomc import study

    sample = sample.copy()
    sample["after"] = sample["llm_after_cutoff"].astype(bool).astype(float)
    sample["tone_x_after"] = sample[study.SCORE] * sample["after"]
    regs = [study.SCORE, "tone_x_after", "after", study.CONTROL, *study.BASELINES]
    pooled = fit(sample, study.OUTCOME, regs, ["tone_x_after"])
    base = [study.SCORE, study.CONTROL, *study.BASELINES]
    before = fit(sample[sample["after"] == 0], study.OUTCOME, base, [study.SCORE])
    after = fit(sample[sample["after"] == 1], study.OUTCOME, base, [study.SCORE])
    sd = float(sample[study.SCORE].std(ddof=1))
    b, a = before.coefficients[study.SCORE], after.coefficients[study.SCORE]
    return {
        "n_before": float(before.n),
        "n_after": float(after.n),
        "beta_before_per_sd": b.beta * sd,
        "se_before_per_sd": b.se * sd,
        "p_before": before.wald_p,
        "beta_after_per_sd": a.beta * sd,
        "se_after_per_sd": a.se * sd,
        "p_after": after.wald_p,
        "interaction_per_sd": pooled.coefficients["tone_x_after"].beta * sd,
        "interaction_p": pooled.wald_p,
    }


PRESS_REGIME_START = "2019-01-01"  # every meeting has a press conference from here (§6.7, A2)


def regime_cutoff(table: pd.DataFrame, press_days: set[str]) -> dict[str, dict[str, float]]:
    """Test A of §6.7: the cutoff split inside the press conference regime.

    A1: statements released on a press conference day. A2: statements from 1 January 2019.
    Each gives β per SD before and after the cutoff and the HAC interaction test.
    """
    from text_signals.sources.fomc import study

    sample = study.primary_sample(table)
    dates = sample["release_date"].astype(str)
    return {
        "A1_press_days": _split_at_cutoff(sample[dates.isin(press_days)]),
        "A2_from_2019": _split_at_cutoff(sample[dates >= PRESS_REGIME_START]),
    }


def oos_gain_after_cutoff(table: pd.DataFrame) -> tuple[float, int]:
    """Σ (baseline − full) out-of-sample squared errors over the statements after the cutoff."""
    from text_signals.event_study import expanding_forecasts
    from text_signals.sources.fomc import study

    sample = study.primary_sample(table).reset_index(drop=True)
    years = pd.to_datetime(sample["release_date"]).dt.year
    full = expanding_forecasts(
        sample,
        study.OUTCOME,
        [study.SCORE, study.CONTROL, *study.BASELINES],
        years,
        study.FIRST_OOS_YEAR,
    )
    base = expanding_forecasts(
        sample, study.OUTCOME, [study.CONTROL, *study.BASELINES], years, study.FIRST_OOS_YEAR
    )
    after = sample["llm_after_cutoff"].astype(bool).to_numpy()
    y = sample[study.OUTCOME].to_numpy(dtype=float)
    gain = (y - base.to_numpy()) ** 2 - (y - full.to_numpy()) ** 2
    return float(gain[after].sum()), int(after.sum())


def anonymised_scores() -> pd.DataFrame:
    """``llm_tone`` of the anonymised texts, by document."""
    from text_signals.sources.fomc.extraction import StatementExtraction, tone_score

    rows = []
    for r in read_cache(cache_path(MODEL, OUT / "anonymised")).values():
        if r.procedure != "single-pass" or r.output is None:
            continue
        rows.append(
            {
                "doc_id": r.doc_id,
                "anon_tone": tone_score(StatementExtraction.model_validate(r.output)),
            }
        )
    return pd.DataFrame(rows).drop_duplicates("doc_id")


def anonymisation_effect(table: pd.DataFrame) -> dict[str, float]:
    from text_signals.event_study import fit
    from text_signals.sources.fomc import study

    t = table.merge(anonymised_scores(), on="doc_id", how="left")
    both = t.dropna(subset=["llm_tone", "anon_tone"])
    sample = study.primary_sample(t).dropna(subset=["anon_tone"])
    regs = ["anon_tone", study.CONTROL, *study.BASELINES]
    r = fit(sample, study.OUTCOME, regs, ["anon_tone"])
    orig = fit(sample, study.OUTCOME, [study.SCORE, study.CONTROL, *study.BASELINES], [study.SCORE])
    return {
        "n_compared": float(len(both)),
        "correlation": float(both["llm_tone"].corr(both["anon_tone"])),
        "mean_change": float((both["anon_tone"] - both["llm_tone"]).mean()),
        "share_changed": float(((both["anon_tone"] - both["llm_tone"]).abs() > 1e-9).mean()),
        "n_regression": float(r.n),
        "beta_anon_per_sd": r.coefficients["anon_tone"].per_sd,
        "p_anon": r.wald_p,
        "beta_orig_same_sample_per_sd": orig.coefficients[study.SCORE].per_sd,
        "p_orig_same_sample": orig.wald_p,
    }


def date_probe(table: pd.DataFrame) -> dict[str, float]:
    rows = []
    for r in read_cache(OUT / "date_probe" / f"{MODEL}.jsonl").values():
        if r.output is not None:
            rows.append({"doc_id": r.doc_id, "guess": int(r.output["year"])})
    guesses = pd.DataFrame(rows).drop_duplicates("doc_id")
    t = table.merge(guesses, on="doc_id")
    year = pd.to_datetime(t["release_date"]).dt.year
    err = (t["guess"] - year).abs()
    naive = (year - int(year.median())).abs()
    after = t["llm_after_cutoff"].astype(bool)
    return {
        "n": float(len(t)),
        "mae_years": float(err.mean()),
        "within_one_year": float((err <= 1).mean()),
        "exact_year": float((err == 0).mean()),
        "naive_mae_years": float(naive.mean()),
        "naive_within_one_year": float((naive <= 1).mean()),
        "mae_before_cutoff": float(err[~after].mean()),
        "mae_after_cutoff": float(err[after].mean()),
        "n_after_cutoff": float(after.sum()),
    }


CHRONO_SCORES = Path("data/processed/fomc/chrono_scores.csv")
VALIDITY_BAR = 0.30  # protocol §6.7, fixed before scoring


def reference_tone(model: str = "claude-opus-5-5") -> pd.DataFrame:
    """The reference reader's tone score on the statements it read (65 + 30)."""
    from text_signals.sources.fomc.extraction import StatementExtraction, tone_score

    rows = []
    for r in read_cache(cache_path(model)).values():
        if r.output is not None:
            tone = tone_score(StatementExtraction.model_validate(r.output))
            rows.append({"doc_id": r.doc_id, "reference_tone": tone})
    return pd.DataFrame(rows).drop_duplicates("doc_id")


def chrono_check(table: pd.DataFrame, chrono: pd.DataFrame) -> dict[str, float]:
    """Validity of the ChronoGPT reader, then the primary regression with its score (§6.7)."""
    from text_signals.event_study import fit
    from text_signals.sources.fomc import study

    t = table.merge(chrono[["doc_id", "chrono_tone"]], on="doc_id", how="left")
    ref = t.merge(reference_tone(), on="doc_id").dropna(subset=["chrono_tone", "reference_tone"])
    r_ref = float(ref["chrono_tone"].corr(ref["reference_tone"]))
    both = t.dropna(subset=["chrono_tone", "llm_tone"])
    sample = study.primary_sample(t).dropna(subset=["chrono_tone"])
    base = [study.CONTROL, *study.BASELINES]
    chrono_fit = fit(sample, study.OUTCOME, ["chrono_tone", *base], ["chrono_tone"])
    phi_fit = fit(sample, study.OUTCOME, [study.SCORE, *base], [study.SCORE])
    joint = fit(sample, study.OUTCOME, ["chrono_tone", study.SCORE, *base], ["chrono_tone"])
    return {
        "n_reference": float(len(ref)),
        "r_with_reference": r_ref,
        "valid": float(r_ref >= VALIDITY_BAR),
        "r_with_phi4_all": float(both["chrono_tone"].corr(both["llm_tone"])),
        "n_regression": float(chrono_fit.n),
        "chrono_per_sd": chrono_fit.coefficients["chrono_tone"].per_sd,
        "chrono_p": chrono_fit.wald_p,
        "phi4_same_sample_per_sd": phi_fit.coefficients[study.SCORE].per_sd,
        "phi4_same_sample_p": phi_fit.wald_p,
        "joint_chrono_per_sd": joint.coefficients["chrono_tone"].per_sd,
        "joint_chrono_p": joint.wald_p,
        "joint_phi4_per_sd": joint.coefficients[study.SCORE].per_sd,
        "joint_phi4_p": float(joint.coefficients[study.SCORE].p),
    }


def chrono_diagnostics(records: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float]:
    """Why the ChronoGPT reader fails (post hoc, not part of the fixed specification, §6.7).

    ``records`` holds one row per statement from the score cache (``vintage``,
    ``sentence_log_odds``, ``chrono_tone``). Separates what the score shares with the reference
    across vintages (era) from what it shares within a vintage (reading), and checks length.
    """
    r = records.copy()
    r["raw"] = r["sentence_log_odds"].apply(lambda xs: sum(xs) / len(xs))
    r["n_sentences"] = r["sentence_log_odds"].apply(len)
    m = r.merge(reference, on="doc_id").dropna(subset=["reference_tone"])
    within_chrono = m["chrono_tone"] - m.groupby("vintage")["chrono_tone"].transform("mean")
    within_ref = m["reference_tone"] - m.groupby("vintage")["reference_tone"].transform("mean")
    return {
        "n_reference": float(len(m)),
        "r_calibrated": float(m["chrono_tone"].corr(m["reference_tone"])),
        "spearman_calibrated": float(m["chrono_tone"].corr(m["reference_tone"], method="spearman")),
        "r_raw_no_anchor": float(m["raw"].corr(m["reference_tone"])),
        "r_within_vintage": float(within_chrono.corr(within_ref)),
        "r_with_n_sentences_all": float(r["chrono_tone"].corr(r["n_sentences"])),
        "r_with_vintage_all": float(r["chrono_tone"].corr(r["vintage"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cmd",
        choices=[
            "anonymised-run",
            "probe-run",
            "show",
            "report",
            "chrono",
            "chrono-diagnostics",
            "regime",
        ],
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    if args.cmd == "anonymised-run":
        records = run_anonymised(args.base_url)
        print(f"{len(records)} records, {sum(r.error is not None for r in records)} invalid")
    elif args.cmd == "probe-run":
        records = run_probe(args.base_url)
        print(f"{len(records)} records, {sum(r.error is not None for r in records)} invalid")
    elif args.cmd == "chrono":
        from text_signals.sources.fomc import study

        table = study.study_table(pd.read_csv(study.EVENTS), pd.read_csv(study.LLM_SCORES))
        for k, v in chrono_check(table, pd.read_csv(CHRONO_SCORES)).items():
            print(f"{k}: {v:.4f}")
    elif args.cmd == "regime":
        from text_signals.sources.fomc import study

        table = study.study_table(pd.read_csv(study.EVENTS), pd.read_csv(study.LLM_SCORES))
        for name, r in regime_cutoff(table, study.press_conference_days()).items():
            print(name)
            for side in ("before", "after"):
                b, se = r[f"beta_{side}_per_sd"], r[f"se_{side}_per_sd"]
                print(
                    f"  {side}: n={r[f'n_{side}']:.0f}  beta/SD={b:+.2f}  se={se:.2f}  "
                    f"95% [{b - 1.96 * se:+.2f}, {b + 1.96 * se:+.2f}]  "
                    f"p={r[f'p_{side}']:.3f}"
                )
            print(f"  interaction/SD={r['interaction_per_sd']:+.2f}  p={r['interaction_p']:.3f}")
    elif args.cmd == "chrono-diagnostics":
        from text_signals.sources.fomc.chrono_scores import CACHE

        cached = pd.read_json(CACHE, lines=True).drop_duplicates("doc_id", keep="last")
        for k, v in chrono_diagnostics(cached, reference_tone()).items():
            print(f"{k}: {v:.4f}")
    elif args.cmd == "report":
        from text_signals.sources.fomc import study

        table = study.study_table(pd.read_csv(study.EVENTS), pd.read_csv(study.LLM_SCORES))
        gain, n_after = oos_gain_after_cutoff(table)
        sections = {
            "cutoff split": cutoff_split(table),
            "out-of-sample gain after the cutoff": {"gain_bp2": gain, "n": float(n_after)},
            "anonymisation": anonymisation_effect(table),
            "date probe": date_probe(table),
        }
        for title, values in sections.items():
            print(f"## {title}")
            for k, v in values.items():
                print(f"{k}: {v:.4f}")
    else:
        docs = anonymised_docs()
        print(len(member_names(d.text for d in all_docs())), "member names")
        for d in (docs[0], docs[len(docs) // 2], docs[-1]):
            print(f"--- {d.doc_id}\n{d.text}\n")


if __name__ == "__main__":
    main()
