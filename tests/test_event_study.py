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


def test_diagnostics_detect_what_they_should() -> None:
    from text_signals.event_study import diagnostics

    rng = np.random.default_rng(3)
    n = 400
    a = rng.normal(0, 1, n)
    clean = pd.DataFrame({"a": a, "b": rng.normal(0, 1, n)})
    clean["y"] = clean["a"] + rng.normal(0, 1, n)
    d = diagnostics(clean, "y", ["a", "b"])
    assert d.vif_flagged == [] and d.breusch_pagan[1] > 0.01 and d.jarque_bera[1] > 0.01

    collinear = clean.assign(b=a + rng.normal(0, 0.1, n))
    assert "a" in diagnostics(collinear, "y", ["a", "b"]).vif_flagged

    hetero = clean.assign(y=clean["a"] + rng.normal(0, 1, n) * (1 + 3 * np.abs(a)))
    assert diagnostics(hetero, "y", ["a", "b"]).breusch_pagan[1] < 0.01

    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.7 * e[t - 1] + rng.normal()
    auto = clean.assign(y=clean["a"] + e)
    d_auto = diagnostics(auto, "y", ["a", "b"])
    assert d_auto.breusch_godfrey[1] < 0.01 and d_auto.ljung_box[1] < 0.01

    fat = clean.assign(y=clean["a"] + rng.standard_t(2, n))
    assert diagnostics(fat, "y", ["a", "b"]).jarque_bera[1] < 0.01


def test_qq_plot_writes_a_png(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from text_signals.event_study import qq_plot

    path = tmp_path / "qq.png"
    qq_plot(np.random.default_rng(0).normal(size=50), str(path), title="t")
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_robustness_agrees_on_clean_data_and_flags_an_outlier() -> None:
    from text_signals.event_study import robustness

    d = _data(0.5, n=300, seed=4)
    d["y"] = d["y"] + np.where(d.index == 10, 60.0, 0.0)  # one large outlier
    r = robustness(d, "y", ["score", "control"], "score", reps=499)
    assert 10 in r.influential
    assert r.cook_threshold == pytest.approx(4 / 300)
    assert r.huber_beta == pytest.approx(0.5, abs=0.06)  # Huber resists the outlier
    assert r.bootstrap_ci[0] < r.beta < r.bootstrap_ci[1]
    assert r.hc3_p < 0.01 and r.huber_p < 0.01
    assert r.bootstrap_reps == 499 and r.bootstrap_block == 4


def test_block_bootstrap_keeps_blocks_of_consecutive_events() -> None:
    from text_signals.event_study import block_bootstrap_beta

    y = np.arange(20, dtype=float)
    x = np.column_stack([np.ones(20), np.arange(20, dtype=float)])
    b = block_bootstrap_beta(y, x, 1, block=4, reps=50, seed=0)
    assert np.allclose(b, 1.0)  # an exact line stays exact under any resampling


def test_holm_matches_a_hand_computation() -> None:
    from text_signals.event_study import holm

    # sorted p: 0.01, 0.02, 0.03, 0.5 -> 4*0.01, 3*0.02, max(0.06, 2*0.03), max(.., 1*0.5)
    assert holm([0.03, 0.5, 0.01, 0.02]) == pytest.approx([0.06, 0.5, 0.04, 0.06])
    assert holm([0.04, 0.03]) == pytest.approx([0.06, 0.06])  # monotone: 2*0.03, then max
    assert holm([0.6, 0.7]) == pytest.approx([1.0, 1.0])  # capped


def test_expanding_forecasts_never_use_their_own_period() -> None:
    from text_signals.event_study import expanding_forecasts

    n = 60
    periods = pd.Series(np.repeat(np.arange(6), 10))
    x = np.arange(n, dtype=float)
    data = pd.DataFrame({"x": x, "y": 2 * x})
    data.loc[50:, "y"] = 1000.0  # period 5 breaks the relation
    f = expanding_forecasts(data, "y", ["x"], periods, first=3)
    assert f[:30].isna().all()
    assert np.allclose(f[30:50], 2 * x[30:50])  # learnt from periods 0-2 and 0-3
    assert np.allclose(f[50:], 2 * x[50:])  # period 5 predicted without seeing itself


def test_oos_r2_and_clark_west_reward_a_useful_regressor() -> None:
    from text_signals.event_study import clark_west, expanding_forecasts, oos_r2

    rng = np.random.default_rng(5)
    n = 300
    d = pd.DataFrame({"s": rng.normal(0, 1, n), "c": rng.normal(0, 1, n)})
    d["y"] = 1.0 * d["s"] + 0.5 * d["c"] + rng.normal(0, 1, n)
    periods = pd.Series(np.repeat(np.arange(15), 20))
    full = expanding_forecasts(d, "y", ["s", "c"], periods, first=3).to_numpy()
    base = expanding_forecasts(d, "y", ["c"], periods, first=3).to_numpy()
    keep = ~np.isnan(full)
    y = d["y"].to_numpy()[keep]
    assert oos_r2(y, full[keep], base[keep]) > 0.2
    t, p = clark_west(y, base[keep], full[keep])
    assert t > 3 and p < 0.01
