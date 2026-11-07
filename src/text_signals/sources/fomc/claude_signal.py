"""The second reader's score in the primary regression, next to Phi-4's (protocol §6.9).

Claude read the anonymised texts (in batches), so the like-for-like comparison is with Phi-4's
reading of the same anonymised texts (§6.7); Phi-4's primary score, on the original texts, is
shown too. All scores enter the primary specification (MP1 and the two baselines as controls,
HAC) on the same statements, and the out-of-sample test of §6.4 is repeated for each.

Usage: ``python -m text_signals.sources.fomc.claude_signal``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from text_signals.llm import read_cache
from text_signals.sources.fomc.claude_batch import CACHE
from text_signals.sources.fomc.extraction import StatementExtraction, tone_score

CLAUDE_CUTOFF = "2026-06-30"  # training data to June 2026 (§6.9)
SCORES = {
    "llm_tone": "Phi-4, original texts (primary)",
    "anon_tone": "Phi-4, anonymised texts",
    "claude_tone": "Claude, anonymised texts (batch)",
}


def claude_scores(path: Path = CACHE) -> pd.DataFrame:
    rows = [
        {
            "doc_id": r.doc_id,
            "claude_tone": tone_score(StatementExtraction.model_validate(r.output)),
        }
        for r in read_cache(path).values()
        if r.output is not None
    ]
    return pd.DataFrame(rows).drop_duplicates("doc_id", keep="last")


def compare(table: pd.DataFrame) -> pd.DataFrame:
    """β per SD, HAC p, incremental R² and out-of-sample results for each score."""
    from text_signals.event_study import clark_west, expanding_forecasts, fit, oos_r2
    from text_signals.sources.fomc import study

    base = [study.CONTROL, *study.BASELINES]
    sample = study.primary_sample(table).dropna(subset=list(SCORES)).reset_index(drop=True)
    r2_base = fit(sample, study.OUTCOME, base, [study.CONTROL]).r2
    years = pd.to_datetime(sample["release_date"]).dt.year
    y = sample[study.OUTCOME].to_numpy(dtype=float)
    small = expanding_forecasts(sample, study.OUTCOME, base, years, study.FIRST_OOS_YEAR)
    rows = []
    for score, label in SCORES.items():
        r = fit(sample, study.OUTCOME, [score, *base], [score])
        large = expanding_forecasts(
            sample, study.OUTCOME, [score, *base], years, study.FIRST_OOS_YEAR
        )
        keep = large.notna().to_numpy()
        f_small, f_large = small.to_numpy()[keep], large.to_numpy()[keep]
        _, cw_p = clark_west(y[keep], f_small, f_large)
        rows.append(
            {
                "score": score,
                "reader": label,
                "n": r.n,
                "beta_per_sd": r.coefficients[score].per_sd,
                "p_hac": r.wald_p,
                "incremental_r2": r.r2 - r2_base,
                "oos_r2_vs_baseline": oos_r2(
                    y[keep], large.to_numpy()[keep], small.to_numpy()[keep]
                ),
                "clark_west_p": cw_p,
            }
        )
    joint = fit(sample, study.OUTCOME, ["claude_tone", "llm_tone", *base], ["claude_tone"])
    rows.append(
        {
            "score": "claude_tone | llm_tone",
            "reader": "Claude next to Phi-4 (original)",
            "n": joint.n,
            "beta_per_sd": joint.coefficients["claude_tone"].per_sd,
            "p_hac": joint.wald_p,
            "incremental_r2": np.nan,
            "oos_r2_vs_baseline": np.nan,
            "clark_west_p": np.nan,
        }
    )
    rows.append(
        {
            "score": "llm_tone | claude_tone",
            "reader": "Phi-4 (original) next to Claude",
            "n": joint.n,
            "beta_per_sd": joint.coefficients["llm_tone"].per_sd,
            "p_hac": float(joint.coefficients["llm_tone"].p),
            "incremental_r2": np.nan,
            "oos_r2_vs_baseline": np.nan,
            "clark_west_p": np.nan,
        }
    )
    return pd.DataFrame(rows)


def study_table() -> pd.DataFrame:
    from text_signals.sources.fomc import study
    from text_signals.sources.fomc.lookahead import anonymised_scores

    table = study.study_table(pd.read_csv(study.EVENTS), pd.read_csv(study.LLM_SCORES))
    return table.merge(anonymised_scores(), on="doc_id", how="left").merge(
        claude_scores(), on="doc_id", how="left"
    )


def main() -> None:
    from text_signals.sources.fomc import study

    table = study_table()
    both = table.dropna(subset=["claude_tone", "llm_tone"])
    print(
        f"Claude scores: {table['claude_tone'].notna().sum()}; correlation with Phi-4: "
        f"original {both['claude_tone'].corr(both['llm_tone']):.3f}, anonymised "
        f"{both['claude_tone'].corr(both['anon_tone']):.3f}"
    )
    main_sample = study.primary_sample(table)
    after = (main_sample["release_date"].astype(str) > CLAUDE_CUTOFF).sum()
    print(f"primary-sample statements after Claude's cutoff: {after}")
    pd.set_option("display.width", 220)
    print(compare(table).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
