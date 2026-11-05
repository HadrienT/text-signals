import numpy as np
import pytest

from text_signals.power import mde_analytic, simulated_power


def test_mde_formula() -> None:
    # (1.95996 + 0.84162) * 5 / sqrt(100) = 1.4008
    assert mde_analytic(5.0, 100) == pytest.approx(1.4008, abs=1e-4)
    # Half of the score explained by the controls: standard error times sqrt(2).
    assert mde_analytic(5.0, 100, r2_x=0.5) == pytest.approx(1.4008 * np.sqrt(2), abs=1e-4)


def test_simulation_has_the_right_size_and_power_with_normal_errors() -> None:
    rng = np.random.default_rng(1)
    residuals = rng.standard_normal(250) * 5
    control = rng.standard_normal(250)
    size = simulated_power(residuals, control, 0.0, n_sims=1000, seed=2)
    assert size == pytest.approx(0.05, abs=0.025)  # HAC is a little liberal in small samples
    mde = mde_analytic(float(residuals.std()), 250)
    power = simulated_power(residuals, control, mde, n_sims=1000, seed=3)
    assert power == pytest.approx(0.8, abs=0.06)
