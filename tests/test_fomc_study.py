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
