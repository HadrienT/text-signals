"""Event-study regressions with the inference fixed in advance (protocol §6.3).

One observation per event, OLS, Newey–West (HAC) standard errors with
⌊4·(T/100)^(2/9)⌋ lags, chosen a priori and never from a pre-test. The effect is reported in
the outcome's units per standard deviation of the score, next to its significance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


def newey_west_lags(n: int) -> int:
    """⌊4·(T/100)^(2/9)⌋, the usual rule of thumb (Newey and West, 1994): 4 for T ≈ 250."""
    return int(math.floor(4 * (n / 100) ** (2 / 9)))


@dataclass(frozen=True)
class Coefficient:
    name: str
    beta: float
    se: float
    t: float
    p: float
    sd: float  # standard deviation of the regressor in the sample

    @property
    def per_sd(self) -> float:
        """Effect of a one-standard-deviation move in the regressor, in outcome units."""
        return self.beta * self.sd


@dataclass(frozen=True)
class EventRegression:
    outcome: str
    n: int
    hac_lags: int
    r2: float
    coefficients: dict[str, Coefficient]
    wald_stat: float  # HAC Wald test of the tested coefficient(s) = 0, chi-squared form
    wald_df: int
    wald_p: float


def fit(
    data: pd.DataFrame,
    outcome: str,
    regressors: Sequence[str],
    tested: Sequence[str],
    *,
    hac_lags: int | None = None,
) -> EventRegression:
    """OLS of ``outcome`` on a constant and ``regressors``, HAC variance, Wald test of
    ``tested`` = 0. Rows must be in event order (the HAC kernel uses it) and complete."""
    frame = data[[outcome, *regressors]]
    if frame.isna().any().any():
        raise ValueError("missing values: select the sample before fitting")
    n = len(frame)
    lags = newey_west_lags(n) if hac_lags is None else hac_lags
    x = sm.add_constant(frame[list(regressors)].astype(float))
    model = sm.OLS(frame[outcome].astype(float), x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    coefficients = {
        name: Coefficient(
            name,
            float(model.params[name]),
            float(model.bse[name]),
            float(model.tvalues[name]),
            float(model.pvalues[name]),
            float(frame[name].std(ddof=1)),
        )
        for name in regressors
    }
    restriction = np.zeros((len(tested), len(x.columns)))
    for i, name in enumerate(tested):
        restriction[i, list(x.columns).index(name)] = 1.0
    wald = model.wald_test(restriction, use_f=False, scalar=True)
    return EventRegression(
        outcome,
        n,
        lags,
        float(model.rsquared),
        coefficients,
        float(wald.statistic),
        len(tested),
        float(wald.pvalue),
    )


@dataclass(frozen=True)
class Diagnostics:
    """The checks of the assumptions table (protocol §6.3). Reported, never used to choose the
    estimator: the inference is fixed in advance."""

    n: int
    correlations: pd.DataFrame  # among the regressors
    vif: dict[str, float]
    breusch_pagan: tuple[float, float]  # LM statistic, p
    white: tuple[float, float]
    breusch_godfrey: tuple[float, float]  # LM statistic, p, at ``lags``
    ljung_box: tuple[float, float]  # Q statistic, p, at ``lags``
    jarque_bera: tuple[float, float]  # statistic, p
    skew: float
    kurtosis: float  # not excess: 3 for a normal
    lags: int
    residuals: np.ndarray  # OLS residuals in event order, for the QQ-plot

    VIF_FLAG = 5.0

    @property
    def vif_flagged(self) -> list[str]:
        return [k for k, v in self.vif.items() if v > self.VIF_FLAG]


def diagnostics(
    data: pd.DataFrame, outcome: str, regressors: Sequence[str], *, lags: int = 4
) -> Diagnostics:
    from statsmodels.stats.diagnostic import (
        acorr_breusch_godfrey,
        acorr_ljungbox,
        het_breuschpagan,
        het_white,
    )
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import jarque_bera

    frame = data[[outcome, *regressors]].astype(float)
    if frame.isna().any().any():
        raise ValueError("missing values: select the sample before the diagnostics")
    x = sm.add_constant(frame[list(regressors)])
    ols = sm.OLS(frame[outcome], x).fit()
    resid = np.asarray(ols.resid)
    vif = {
        name: float(variance_inflation_factor(x.to_numpy(), list(x.columns).index(name)))
        for name in regressors
    }
    bp = het_breuschpagan(resid, x)
    wh = het_white(resid, x)
    bg = acorr_breusch_godfrey(ols, nlags=lags, result_object=False)
    lb = acorr_ljungbox(resid, lags=[lags])
    jb, jb_p, skew, kurt = jarque_bera(resid)
    return Diagnostics(
        n=len(frame),
        correlations=frame[list(regressors)].corr(),
        vif=vif,
        breusch_pagan=(float(bp[0]), float(bp[1])),
        white=(float(wh[0]), float(wh[1])),
        breusch_godfrey=(float(bg[0]), float(bg[1])),
        ljung_box=(float(lb["lb_stat"].iloc[0]), float(lb["lb_pvalue"].iloc[0])),
        jarque_bera=(float(jb), float(jb_p)),
        skew=float(skew),
        kurtosis=float(kurt),
        lags=lags,
        residuals=resid,
    )


def qq_plot(residuals: np.ndarray, path: str, *, title: str, unit: str = "bp") -> None:
    """Sorted residuals against the quantiles of a normal with the residuals' standard
    deviation; the grey line is where normal residuals would fall."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    r = np.sort(residuals)
    n = len(r)
    q = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n) * r.std(ddof=1)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    lim = float(max(abs(r).max(), abs(q).max())) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="#9a9893", linewidth=1.5, zorder=1)
    ax.scatter(q, r, s=14, color="#2a78d6", edgecolor="#fcfcfb", linewidth=0.6, zorder=2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"Normal quantile ({unit})", color="#52514e")
    ax.set_ylabel(f"Residual ({unit})", color="#52514e")
    ax.set_title(title, color="#0b0b0b", fontsize=10, loc="left")
    ax.grid(color="#e6e5e1", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors="#52514e", labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@dataclass(frozen=True)
class Robustness:
    """The robustness checks of protocol §6.3 for one tested coefficient."""

    name: str
    n: int
    beta: float  # OLS, all events
    hac_p: float
    hc3_se: float
    hc3_p: float
    bootstrap_se: float  # moving-block pairs bootstrap of β
    bootstrap_ci: tuple[float, float]  # percentile 95%
    bootstrap_block: int
    bootstrap_reps: int
    cook_threshold: float  # 4 / T
    influential: list[int]  # row positions with Cook's distance above the threshold
    beta_without_influential: float
    hac_p_without_influential: float
    huber_beta: float
    huber_se: float
    huber_p: float


def block_bootstrap_beta(
    y: np.ndarray, x: np.ndarray, column: int, *, block: int, reps: int, seed: int
) -> np.ndarray:
    """OLS β of ``column`` on resampled moving blocks of ``block`` consecutive events (pairs
    bootstrap: each block keeps its events' outcome and regressors together, and the blocks
    keep the short-range dependence that HAC allows for)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    starts = np.arange(n - block + 1)
    k = int(np.ceil(n / block))
    out = np.empty(reps)
    for r in range(reps):
        idx = (rng.choice(starts, k)[:, None] + np.arange(block)[None, :]).ravel()[:n]
        coef, *_ = np.linalg.lstsq(x[idx], y[idx], rcond=None)
        out[r] = coef[column]
    return out


def robustness(
    data: pd.DataFrame,
    outcome: str,
    regressors: Sequence[str],
    tested: str,
    *,
    block: int = 4,
    reps: int = 9_999,
    seed: int = 0,
) -> Robustness:
    frame = data[[outcome, *regressors]].astype(float)
    if frame.isna().any().any():
        raise ValueError("missing values: select the sample before the robustness checks")
    n = len(frame)
    x = sm.add_constant(frame[list(regressors)])
    y = frame[outcome]
    ols = sm.OLS(y, x).fit()
    hac = fit(data, outcome, regressors, [tested])
    hc3 = ols.get_robustcov_results(cov_type="HC3")
    j = list(x.columns).index(tested)

    boots = block_bootstrap_beta(y.to_numpy(), x.to_numpy(), j, block=block, reps=reps, seed=seed)

    cook = np.asarray(ols.get_influence().cooks_distance[0])
    threshold = 4 / n
    flagged = [int(i) for i in np.flatnonzero(cook > threshold)]
    kept = data.iloc[[i for i in range(n) if i not in set(flagged)]]
    without = fit(kept, outcome, regressors, [tested])

    huber = sm.RLM(y, x, M=sm.robust.norms.HuberT()).fit()
    return Robustness(
        name=tested,
        n=n,
        beta=float(ols.params[tested]),
        hac_p=hac.wald_p,
        hc3_se=float(hc3.bse[j]),
        hc3_p=float(hc3.pvalues[j]),
        bootstrap_se=float(boots.std(ddof=1)),
        bootstrap_ci=(float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))),
        bootstrap_block=block,
        bootstrap_reps=reps,
        cook_threshold=threshold,
        influential=flagged,
        beta_without_influential=without.coefficients[tested].beta,
        hac_p_without_influential=without.wald_p,
        huber_beta=float(huber.params[tested]),
        huber_se=float(huber.bse[tested]),
        huber_p=float(huber.pvalues[tested]),
    )


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (step-down, monotone, capped at 1), in input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def expanding_forecasts(
    data: pd.DataFrame,
    outcome: str,
    regressors: Sequence[str],
    periods: pd.Series,
    first: int,
) -> pd.Series:
    """Out-of-sample OLS forecasts: each event of period p >= ``first`` is predicted by
    coefficients fitted on all events of earlier periods only (expanding window, one refit per
    period). Events before ``first`` get NaN. ``periods`` are integers aligned with ``data``."""
    y = data[outcome].astype(float).to_numpy()
    x = sm.add_constant(data[list(regressors)].astype(float)).to_numpy()
    per = periods.to_numpy()
    out = np.full(len(y), np.nan)
    for p in sorted(set(per[per >= first])):
        train, test = per < p, per == p
        coef, *_ = np.linalg.lstsq(x[train], y[train], rcond=None)
        out[test] = x[test] @ coef
    return pd.Series(out, index=data.index)


def oos_r2(y: np.ndarray, forecast: np.ndarray, benchmark: np.ndarray) -> float:
    """1 − Σ(y − forecast)² / Σ(y − benchmark)²: positive when the forecast beats the
    benchmark out of sample."""
    return float(1 - np.sum((y - forecast) ** 2) / np.sum((y - benchmark) ** 2))


def clark_west(
    y: np.ndarray, small: np.ndarray, large: np.ndarray, *, lags: int | None = None
) -> tuple[float, float]:
    """Clark–West (2007) test that the larger, nesting model forecasts better: the adjusted
    loss difference f = (y − ŝ)² − [(y − l̂)² − (ŝ − l̂)²], regressed on a constant with HAC
    variance. Returns (t, one-sided p)."""
    from scipy import stats

    f = (y - small) ** 2 - ((y - large) ** 2 - (small - large) ** 2)
    n = len(f)
    maxlags = newey_west_lags(n) if lags is None else lags
    res = sm.OLS(f, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    t = float(res.tvalues[0])
    return t, float(1 - stats.norm.cdf(t))


def cumulative_gain_plot(
    years: Sequence[int], gain: Sequence[float], path: str, *, title: str, ylabel: str
) -> None:
    """Cumulative out-of-sample gain of one model over another (Goyal–Welch style): rising
    where the model forecasts better, falling where it forecasts worse."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axhline(0, color="#9a9893", linewidth=1.0, zorder=1)
    ax.plot(
        list(years),
        list(gain),
        color="#2a78d6",
        linewidth=2,
        marker="o",
        markersize=4,
        markeredgecolor="#fcfcfb",
        markeredgewidth=0.8,
        zorder=2,
    )
    ax.set_ylabel(ylabel, color="#52514e")
    ax.set_title(title, color="#0b0b0b", fontsize=10, loc="left")
    ax.grid(color="#e6e5e1", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors="#52514e", labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@dataclass(frozen=True)
class PermutationTest:
    observed_t: float
    permuted_t: np.ndarray
    reps: int

    @property
    def p(self) -> float:
        """Share of permutations, the observed one included, with |t| ≥ |observed t|."""
        hits = int(np.sum(np.abs(self.permuted_t) >= abs(self.observed_t)))
        return (hits + 1) / (self.reps + 1)


def permutation_test(
    data: pd.DataFrame,
    outcome: str,
    regressors: Sequence[str],
    tested: str,
    *,
    reps: int = 9_999,
    seed: int = 0,
) -> PermutationTest:
    """Shuffle ``tested`` across events, keep everything else with its own event, refit, and
    collect the HAC t of ``tested``: its distribution when the regressor is unrelated to the
    event it is attached to."""
    frame = data[[outcome, *regressors]].astype(float).reset_index(drop=True)
    if frame.isna().any().any():
        raise ValueError("missing values: select the sample before the permutation test")
    lags = newey_west_lags(len(frame))
    x = sm.add_constant(frame[list(regressors)])
    j = list(x.columns).index(tested)
    y = frame[outcome].to_numpy()

    def t_of(xm: np.ndarray) -> float:
        res = sm.OLS(y, xm).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        return float(res.tvalues[j])

    xm = x.to_numpy().copy()
    observed = t_of(xm)
    rng = np.random.default_rng(seed)
    column = xm[:, j].copy()
    out = np.empty(reps)
    for r in range(reps):
        xm[:, j] = rng.permutation(column)
        out[r] = t_of(xm)
    return PermutationTest(observed, out, reps)
