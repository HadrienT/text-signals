import pandas as pd
import pytest

from text_signals.sources.fomc.study import primary_sample, study_table


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doc_id": ["b", "a", "c"],
            "release_date": ["2001-02-01", "2001-01-03", "2001-03-20"],
            "d_UST2Y_30m_bp": [1.0, -2.0, 0.5],
            "MP1_bp": [0.0, -10.0, None],
            "lm_net_negative": [0.01, 0.02, 0.0],
            "finbert_net_negative": [0.1, 0.2, 0.0],
            "in_main_sample": [True, True, True],
        }
    )


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doc_id": ["a", "b", "c"],
            "release_date": ["2001-01-03", "2001-02-01", "2001-03-20"],
            "llm_tone": [-0.5, 0.25, 1.0],
        }
    )


def test_join_is_in_release_order_and_keeps_every_statement() -> None:
    t = study_table(_events(), _scores())
    assert list(t["doc_id"]) == ["a", "b", "c"]
    assert list(t["llm_tone"]) == [-0.5, 0.25, 1.0]


def test_join_refuses_a_date_mismatch_and_duplicate_scores() -> None:
    bad = _scores().assign(release_date=["2001-01-04", "2001-02-01", "2001-03-20"])
    with pytest.raises(ValueError, match="release dates differ"):
        study_table(_events(), bad)
    with pytest.raises(pd.errors.MergeError):
        study_table(_events(), pd.concat([_scores(), _scores().iloc[:1]]))


def test_primary_sample_drops_rows_missing_a_variable() -> None:
    t = study_table(_events(), _scores())
    assert list(primary_sample(t)["doc_id"]) == ["a", "b"]  # "c" has no MP1


def test_primary_test_uses_the_frozen_specification() -> None:
    import numpy as np

    from text_signals.sources.fomc.study import primary_test

    rng = np.random.default_rng(0)
    n = 120
    table = pd.DataFrame(
        {
            "doc_id": [f"d{i:03d}" for i in range(n)],
            "release_date": pd.date_range("2000-01-01", periods=n, freq="45D").astype(str),
            "llm_tone": rng.normal(0, 1, n),
            "MP1_bp": rng.normal(0, 5, n),
            "lm_net_negative": rng.normal(0, 0.01, n),
            "finbert_net_negative": rng.normal(0, 0.2, n),
            "in_main_sample": True,
        }
    )
    table["d_UST2Y_30m_bp"] = 2 * table["llm_tone"] + 0.3 * table["MP1_bp"] + rng.normal(0, 3, n)
    r = primary_test(table)
    assert list(r.coefficients) == [
        "llm_tone",
        "MP1_bp",
        "lm_net_negative",
        "finbert_net_negative",
    ]
    assert r.wald_df == 1 and r.hac_lags == 4 and r.n == n


def test_robustness_runner_lists_influential_statements_by_date() -> None:
    import numpy as np

    from text_signals.sources.fomc.study import describe_robustness, primary_robustness

    rng = np.random.default_rng(1)
    n = 80
    table = pd.DataFrame(
        {
            "doc_id": [f"d{i:03d}" for i in range(n)],
            "release_date": pd.date_range("2000-01-01", periods=n, freq="45D").astype(str),
            "llm_tone": rng.normal(0, 1, n),
            "MP1_bp": rng.normal(0, 5, n),
            "lm_net_negative": rng.normal(0, 0.01, n),
            "finbert_net_negative": rng.normal(0, 0.2, n),
            "in_main_sample": True,
        }
    )
    table["d_UST2Y_30m_bp"] = table["llm_tone"] + rng.normal(0, 1, n)
    table.loc[5, "d_UST2Y_30m_bp"] = 40.0
    r, infl = primary_robustness(table)
    assert table.loc[5, "release_date"] in set(infl["release_date"])
    assert "Huber" in describe_robustness(r, infl, 1.0)
