from datetime import date

import numpy as np
import pandas as pd
import pytest

from text_signals.sources.fomc.counterfactual import (
    VERSIONS,
    bootstrap_diff,
    edited_docs,
    guidance_text,
    load_edits,
    measures,
    report,
)
from text_signals.sources.fomc.extract import SampleDoc
from text_signals.sources.fomc.extraction import STANCE_FIELDS

TEXT = (
    "The Committee decided today to keep its target at 1 percent. Growth is solid. "
    "The Committee believes that policy accommodation can be maintained for a considerable "
    "period. Inflation is low. The Committee will respond as needed."
)


def test_guidance_text_takes_the_sentence_holding_the_evidence() -> None:
    out = guidance_text(TEXT, "policy accommodation can be maintained")
    assert out is not None and out.startswith("The Committee believes")


def test_guidance_text_takes_the_shortest_block_when_the_span_crosses_sentences() -> None:
    out = guidance_text(TEXT, "considerable period. Inflation is low.")
    assert out is not None
    assert out.startswith("The Committee believes") and out.endswith("Inflation is low.")


def test_guidance_text_falls_back_to_the_closest_sentence_when_not_verbatim() -> None:
    out = guidance_text(TEXT, "The Committee believes accommodation maintained considerable period")
    assert out is not None and out.startswith("The Committee believes")


def _edit(original: str) -> dict[str, str]:
    return {
        "doc_id": "d1",
        "stratum": "2000-2018",
        "original": original,
        "hawkish": "Rates will rise.",
        "dovish": "Rates will stay low.",
        "neutral": "Same meaning.",
    }


def test_edited_docs_replace_only_the_passage_and_check_it() -> None:
    doc = SampleDoc("d1", "d1", date(2004, 1, 1), TEXT)
    out = edited_docs([doc], [_edit("Inflation is low.")])
    assert [d.doc_id for d in out] == [f"d1__{v}" for v in VERSIONS]
    assert "Rates will rise." in out[0].text and "Inflation is low." not in out[0].text
    assert out[0].text.startswith("The Committee decided")
    with pytest.raises(ValueError):
        edited_docs([doc], [_edit("Not in the text.")])


def test_the_committed_edits_file_is_complete() -> None:
    edits = load_edits()
    assert len(edits) == 38
    for e in edits:
        assert all(e[v] and e[v] != e["original"] for v in VERSIONS)


def _wide(fg: dict[str, list[float]], stratum: list[str]) -> pd.DataFrame:
    n = len(stratum)
    w = pd.DataFrame({"doc_id": [f"d{i}" for i in range(n)], "stratum": stratum})
    for v, values in fg.items():
        w[f"fg_{v}"] = values
        w[f"tone_{v}"] = values
        for f in STANCE_FIELDS:
            if f != "forward_guidance":
                w[f"{f}_{v}"] = 0.0
    return w


def test_measures_count_following_and_missing_values() -> None:
    w = _wide(
        {
            "original": [0, 0, 1, -1],
            "hawkish": [2, 1, 1, np.nan],
            "dovish": [-2, -1, 1, -2],
            "neutral": [0, 1, 1, -1],
        },
        ["2000-2018"] * 4,
    )
    m = measures(w)
    assert m["B1_hawk_above_dove"] == 0.5  # equal and missing do not count
    assert m["B3_neutral_changed"] == 0.25
    assert m["B3_spillover_mean_abs"] == 0.0
    assert m["missing_fg_edited"] == 1


def test_report_compares_before_and_after_and_drops_flagged() -> None:
    before = ["2000-2018"] * 6
    after = ["after-cutoff"] * 6
    w = _wide(
        {
            "original": [0.0] * 12,
            "hawkish": [1.0] * 6 + [2.0] * 6,
            "dovish": [0.0] * 6 + [-2.0] * 6,
            "neutral": [0.0] * 12,
        },
        before + after,
    )
    r = report(w, flagged={"d0", "d1"}, recognised={"d0", "d2"})
    assert r["all/B4"]["gap_before_minus_after"] == -3.0
    assert r["all/B4"]["ci_high"] < 0
    assert r["without_other_guidance/before"]["n"] == 4
    assert r["before/recognised"]["n"] == 2


def test_bootstrap_diff_interval_covers_the_estimate() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 50), rng.normal(0, 1, 50)
    d, lo, hi = bootstrap_diff(a, b, reps=2000)
    assert lo < d < hi and lo > 0
