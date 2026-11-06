import pytest

from text_signals.evaluation import cohen_kappa, field_agreement


def test_kappa_perfect_and_chance() -> None:
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == pytest.approx(1.0)
    # agreement exactly at chance: p_o = 0.5, p_e = 0.5
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    assert cohen_kappa(["a", "a"], ["a", "a"]) is None


def test_kappa_matches_a_hand_computation() -> None:
    # 2x2 table [[20, 5], [10, 15]] out of 50: p_o = 0.7, p_e = 0.5*0.6 + 0.5*0.4 = 0.5
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_quadratic_weights_penalise_large_gaps_more() -> None:
    a = [-2, -1, 0, 1, 2, 0]
    near = [-1, -1, 0, 1, 2, 0]
    far = [2, -1, 0, 1, 2, 0]
    k_near = cohen_kappa(a, near, ordinal=True)
    k_far = cohen_kappa(a, far, ordinal=True)
    assert k_near is not None and k_far is not None and k_near > k_far


def test_field_agreement_separates_presence_and_value() -> None:
    a = [1, 0, None, 2, None, -1]
    b = [1, 1, None, None, 0, -1]
    f = field_agreement("x", a, b, ordinal=True)
    assert (f.both_filled, f.both_empty, f.only_a, f.only_b) == (3, 1, 1, 1)
    assert f.presence_agreement == pytest.approx(4 / 6)
    assert f.exact == pytest.approx(2 / 3)
    assert f.within_one == pytest.approx(1.0)
    assert f.mae == pytest.approx(1 / 3)


def test_compare_kappas_separates_a_good_and_a_bad_reader() -> None:
    from text_signals.evaluation import compare_kappas

    ref = [-2, -1, 0, 1, 2] * 12
    good = list(ref)
    good[0] = -1  # one near miss
    bad = [-v for v in ref]  # every sign inverted
    c = compare_kappas(ref, good, bad, ordinal=True, n_boot=300)
    assert c.kappa_x is not None and c.kappa_y is not None and c.kappa_x > 0.9 > c.kappa_y
    assert c.ci_diff[0] > 0  # the difference is clear
    assert c.ci_x[0] <= c.kappa_x <= c.ci_x[1]
