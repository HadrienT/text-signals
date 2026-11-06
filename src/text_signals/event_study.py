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
