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

from text_signals.event_study import (
    Diagnostics,
    EventRegression,
    Robustness,
    diagnostics,
    fit,
    qq_plot,
    robustness,
)

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


FIGURES = Path("docs/case-studies/figures")


def primary_diagnostics(table: pd.DataFrame) -> Diagnostics:
    """The assumption checks of protocol §6.3 on the primary test's sample and regressors."""
    return diagnostics(primary_sample(table), OUTCOME, [SCORE, CONTROL, *BASELINES])


def describe_diagnostics(d: Diagnostics) -> str:
    names = list(d.vif)
    lines = [f"T = {d.n}", "", "Correlations among regressors:", ""]
    lines.append("| | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    corr = d.correlations.to_numpy(dtype=float)
    for i, a in enumerate(names):
        lines.append(
            f"| {a} | " + " | ".join(f"{corr[i, j]:.2f}" for j in range(len(names))) + " |"
        )
    lines += [
        "",
        "VIF: "
        + ", ".join(f"{k} {v:.2f}" for k, v in d.vif.items())
        + f" (flag above {d.VIF_FLAG:g}: {d.vif_flagged or 'none'})",
        f"Breusch-Pagan LM {d.breusch_pagan[0]:.2f}, p = {d.breusch_pagan[1]:.3g}",
        f"White LM {d.white[0]:.2f}, p = {d.white[1]:.3g}",
        f"Breusch-Godfrey ({d.lags} lags) LM {d.breusch_godfrey[0]:.2f}, "
        f"p = {d.breusch_godfrey[1]:.3g}",
        f"Ljung-Box ({d.lags} lags) Q {d.ljung_box[0]:.2f}, p = {d.ljung_box[1]:.3g}",
        f"Jarque-Bera {d.jarque_bera[0]:.1f}, p = {d.jarque_bera[1]:.3g}; skew {d.skew:.2f}, "
        f"kurtosis {d.kurtosis:.2f}",
    ]
    return "\n".join(lines)


def primary_robustness(table: pd.DataFrame) -> tuple[Robustness, pd.DataFrame]:
    """The robustness checks of protocol §6.3 for β_LLM, and the influential statements."""
    sample = primary_sample(table).reset_index(drop=True)
    r = robustness(sample, OUTCOME, [SCORE, CONTROL, *BASELINES], SCORE)
    cols = ["release_date", OUTCOME, SCORE, CONTROL]
    return r, sample.loc[r.influential, cols]


def describe_robustness(r: Robustness, influential: pd.DataFrame, sd: float) -> str:
    lo, hi = r.bootstrap_ci
    rows = [
        ("OLS, HAC (primary)", r.beta, None, r.hac_p),
        ("OLS, HC3", r.beta, r.hc3_se, r.hc3_p),
        (
            f"Block bootstrap ({r.bootstrap_block}, {r.bootstrap_reps:,} reps)",
            r.beta,
            r.bootstrap_se,
            None,
        ),
        (
            f"Without {len(r.influential)} influential (Cook > 4/T), HAC",
            r.beta_without_influential,
            None,
            r.hac_p_without_influential,
        ),
        ("Huber robust regression", r.huber_beta, r.huber_se, r.huber_p),
    ]
    lines = [
        f"β of {r.name}, T = {r.n}; per SD = β × {sd:.3f}",
        "",
        "| Check | β | β per SD | se | p |",
        "|---|---|---|---|---|",
    ]
    for label, beta, se, p in rows:
        lines.append(
            f"| {label} | {beta:.3f} | {beta * sd:.2f} | "
            f"{'–' if se is None else f'{se:.3f}'} | {'–' if p is None else f'{p:.4f}'} |"
        )
    lines += [
        "",
        f"Bootstrap 95% interval of β: [{lo:.3f}, {hi:.3f}] "
        f"(per SD [{lo * sd:.2f}, {hi * sd:.2f}])",
        "",
        "Influential statements:",
        influential.round(2).to_string(index=False),
    ]
    return "\n".join(lines)


def main() -> None:
    import sys

    table = study_table(pd.read_csv(EVENTS), pd.read_csv(LLM_SCORES))
    if sys.argv[1:] == ["primary"]:
        print(describe(primary_test(table)))
        return
    if sys.argv[1:] == ["robustness"]:
        r, infl = primary_robustness(table)
        sd = float(primary_sample(table)[SCORE].std(ddof=1))
        print(describe_robustness(r, infl, sd))
        return
    if sys.argv[1:] == ["diagnostics"]:
        d = primary_diagnostics(table)
        print(describe_diagnostics(d))
        FIGURES.mkdir(parents=True, exist_ok=True)
        path = FIGURES / "fomc_primary_qq.png"
        qq_plot(d.residuals, str(path), title="Primary test: OLS residuals, 2-year yield, 30 min")
        print(f"QQ-plot -> {path}")
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
