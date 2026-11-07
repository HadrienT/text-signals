"""Power of the primary test, computed before any score is joined to the outcome.

The minimum detectable effect (MDE) is the smallest true effect of a one-standard-deviation
change in the score that the test detects with a given probability (the power), at a given
false-positive rate (the size). It depends only on the noise left in the outcome after the
controls, the number of events and how much of the score the controls already explain; it does
not use the score itself.

Two estimates (protocol §6.3):

- ``mde_analytic``: the textbook formula for an OLS coefficient under homoskedastic normal
  errors, ``MDE = (z_{1-α/2} + z_{power}) · σ_e / sqrt(T · (1 − R²_x))``;
- ``simulated_power``: Monte Carlo with the outcome's own residuals resampled (so their fat
  tails are kept), a synthetic score and the primary test itself (OLS, HAC with 4 lags,
  two-sided Wald test).
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import numpy.typing as npt
import statsmodels.api as sm

FloatArray = npt.NDArray[np.float64]


def mde_analytic(
    sigma_e: float, n: int, *, r2_x: float = 0.0, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Minimum detectable effect, in the unit of the outcome per standard deviation of score.

    ``sigma_e``: residual standard deviation of the outcome after the controls. ``r2_x``: share of
    the score's variance explained by the controls (collinearity inflates the standard error).
    """
    if not 0 <= r2_x < 1:
        raise ValueError("r2_x must be in [0, 1)")
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return z * sigma_e / float(np.sqrt(n * (1 - r2_x)))


def simulated_power(
    residuals: FloatArray,
    control: FloatArray,
    beta: float,
    *,
    rho: float = 0.0,
    n_sims: int = 2000,
    alpha: float = 0.05,
    hac_lags: int = 4,
    seed: int = 0,
) -> float:
    """Share of simulations in which the primary test rejects β = 0.

    Each simulation draws a standardised score ``s`` with correlation ``rho`` with the
    control, resamples the residuals with replacement, builds ``y = β·s + e`` (the control's own
    effect does not change the test on β and is left out), regresses ``y`` on a constant, the
    control and ``s`` by OLS, and tests β with a HAC (Newey–West) variance.
    """
    rng = np.random.default_rng(seed)
    n = len(residuals)
    if len(control) != n:
        raise ValueError("residuals and control must have the same length")
    c = (control - control.mean()) / control.std()
    exog = np.column_stack([np.ones(n), control, np.zeros(n)])
    rejections = 0
    for _ in range(n_sims):
        s = rho * c + np.sqrt(1 - rho**2) * rng.standard_normal(n)
        s = (s - s.mean()) / s.std()
        y = beta * s + rng.choice(residuals, size=n, replace=True)
        exog[:, 2] = s
        fit = sm.OLS(y, exog).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        rejections += bool(fit.pvalues[2] < alpha)
    return rejections / n_sims
