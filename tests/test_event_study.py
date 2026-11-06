import numpy as np
import pandas as pd
import pytest

from text_signals.event_study import fit, newey_west_lags


def test_lag_rule_gives_four_for_the_fomc_sample() -> None:
    assert newey_west_lags(243) == 4
    assert newey_west_lags(100) == 4
    assert newey_west_lags(1000) == 6


def _data(beta: float, n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    score = rng.normal(0, 2, n)
    control = rng.normal(0, 1, n)
    y = 1.0 + beta * score + 3.0 * control + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "score": score, "control": control})


def test_recovers_a_known_effect_and_reports_it_per_sd() -> None:
    r = fit(_data(0.5), "y", ["score", "control"], ["score"])
    c = r.coefficients["score"]
    assert c.beta == pytest.approx(0.5, abs=0.05)
    assert c.per_sd == pytest.approx(c.beta * c.sd)
    assert c.sd == pytest.approx(2.0, rel=0.1)
    assert r.wald_p < 1e-6 and r.wald_df == 1
    assert r.wald_stat == pytest.approx(c.t**2, rel=1e-6)  # one restriction: Wald = t²


def test_no_effect_is_not_rejected_and_missing_values_are_refused() -> None:
    r = fit(_data(0.0, seed=1), "y", ["score", "control"], ["score"])
    assert r.wald_p > 0.05
    bad = _data(0.0).assign(score=lambda d: d["score"].where(d.index != 3))
    with pytest.raises(ValueError, match="missing"):
        fit(bad, "y", ["score", "control"], ["score"])
