"""The table of the FOMC event study: one row per statement (protocol §6.3).

It joins the event table (market changes around each release, the policy surprise MP1, the
baseline scores and the sample flags, all keyed by the release; ``events.py``) with the LLM
scores (``llm_scores.py``). The LLM scores are read from the text only, so they are joined on the
document id; the point-in-time rule is carried by the event table, which keys every market
change to the release timestamp.

Usage: ``python -m text_signals.sources.fomc.study`` writes
``data/processed/fomc/study_table.csv``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from text_signals.event_study import EventRegression, fit

EVENTS = Path("data/processed/fomc/statement_events.csv")
LLM_SCORES = Path("data/processed/fomc/llm_scores_phi-4.csv")
OUT = Path("data/processed/fomc/study_table.csv")

# The primary test (protocol §6.3): outcome, score, decision control, baselines.
OUTCOME = "d_UST2Y_30m_bp"
SCORE = "llm_tone"
CONTROL = "MP1_bp"
BASELINES = ("lm_net_negative", "finbert_net_negative")


def study_table(events: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Every statement of the event table with its LLM scores, in release order.

    Fails if a statement has two score rows, or if the two tables disagree on a release date.
    """
    llm = scores.drop(columns=["release_date"])
    frame = events.merge(llm, on="doc_id", how="left", validate="one_to_one")
    check = events[["doc_id", "release_date"]].merge(
        scores[["doc_id", "release_date"]], on="doc_id", suffixes=("", "_llm")
    )
    bad = check[check["release_date"].astype(str) != check["release_date_llm"].astype(str)]
    if not bad.empty:
        raise ValueError(f"release dates differ for {list(bad['doc_id'])}")
    return frame.sort_values(["release_date", "doc_id"]).reset_index(drop=True)


def primary_sample(table: pd.DataFrame) -> pd.DataFrame:
    """The statements of the primary test: the main sample, with every variable present."""
    cols = [OUTCOME, SCORE, CONTROL, *BASELINES]
    main = table[table["in_main_sample"].astype(bool)]
    return main.dropna(subset=cols)


def primary_test(table: pd.DataFrame) -> EventRegression:
    """The one primary test of protocol §6.3: H0 β_LLM = 0, HAC Wald, main sample."""
    sample = primary_sample(table)
    return fit(sample, OUTCOME, [SCORE, CONTROL, *BASELINES], [SCORE])


def describe(r: EventRegression) -> str:
    lines = [
        f"{r.outcome} on {', '.join(r.coefficients)}; T = {r.n}, HAC lags {r.hac_lags}, "
        f"R² {r.r2:.3f}",
        "",
        "| Regressor | β | HAC se | t | p | SD | β per SD |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in r.coefficients.values():
        lines.append(
            f"| {c.name} | {c.beta:.3f} | {c.se:.3f} | {c.t:.2f} | {c.p:.3f} | {c.sd:.4g} "
            f"| {c.per_sd:.2f} |"
        )
    lines.append("")
    lines.append(
        f"Wald (HAC) of the tested coefficient = 0: χ²({r.wald_df}) = {r.wald_stat:.3f}, "
        f"p = {r.wald_p:.3f}"
    )
    return "\n".join(lines)


def main() -> None:
    import sys

    table = study_table(pd.read_csv(EVENTS), pd.read_csv(LLM_SCORES))
    if sys.argv[1:] == ["primary"]:
        print(describe(primary_test(table)))
        return
    table.to_csv(OUT, index=False)
    main_ = table[table["in_main_sample"].astype(bool)]
    used = primary_sample(table)
    missing = {c: int(main_[c].isna().sum()) for c in [OUTCOME, SCORE, CONTROL, *BASELINES]}
    print(
        f"{len(table)} statements, {len(main_)} in the main sample, {len(used)} in the primary "
        f"test; missing in the main sample: {missing} -> {OUT}"
    )


if __name__ == "__main__":
    main()
