from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest

from text_signals.documents import Document
from text_signals.encoders import (
    FINBERT,
    EncoderSpec,
    SentenceProbs,
    read_cache,
    score_documents,
    split_sentences,
)

# Paragraphs of the statement of 28 October 2026, and the vote of 9 December 2003, whose page
# breaks a line inside a name.
STATEMENT_2026 = (
    "The Federal Open Market Committee approved the following statement for release by a "
    "10 – 2 vote:\n\nInflation remains elevated. The Committee will deliver price stability."
    "\n\nVoting against the monetary policy action were Lisa D. Cook and Beth M. Hammack, who "
    "preferred to raise the target range for the federal funds rate by 1/4 percentage point at "
    "this meeting."
)
VOTE_2003 = "Donald L. Kohn; Mark W. Olson; and Robert T.\n\nParry."


def test_split_sentences_on_a_real_statement() -> None:
    assert split_sentences(STATEMENT_2026) == [
        "The Federal Open Market Committee approved the following statement for release by a "
        "10 – 2 vote:",
        "Inflation remains elevated.",
        "The Committee will deliver price stability.",
        "Voting against the monetary policy action were Lisa D. Cook and Beth M. Hammack, who "
        "preferred to raise the target range for the federal funds rate by 1/4 percentage point "
        "at this meeting.",
    ]


def test_split_sentences_keeps_names_and_abbreviations_whole() -> None:
    assert split_sentences(VOTE_2003) == ["Donald L. Kohn; Mark W. Olson; and Robert T. Parry."]
    assert split_sentences("The U.S. Treasury agreed. Rates fell.") == [
        "The U.S. Treasury agreed.",
        "Rates fell.",
    ]


def _doc(doc_id: str, text: str) -> Document:
    return Document(
        doc_id=doc_id,
        corpus="fomc",
        doc_type="statement",
        event_date=date(2026, 10, 28),
        release_date=date(2026, 10, 28),
        release_time_et=None,
        release_time_source="none",
        unscheduled=False,
        url="https://example.org",
        raw_sha256="0" * 64,
        title="t",
        text=text,
    )


class FakeModel:
    """Positive for sentences with "strong", negative for "weak", neutral otherwise."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, sentences: Sequence[str]) -> SentenceProbs:
        self.calls += 1
        table = {"strong": [0.8, 0.1, 0.1], "weak": [0.1, 0.7, 0.2]}
        probs = [next((v for k, v in table.items() if k in s), [0.0, 0.0, 1.0]) for s in sentences]
        return SentenceProbs(labels=FINBERT.labels, probs=probs, truncated=[False] * len(probs))


def test_score_is_the_mean_over_sentences(tmp_path: Path) -> None:
    model = FakeModel()
    doc = _doc("d1", "Growth is strong. Hiring is weak. Prices are stable. Spending is weak.")
    [score] = score_documents([doc], FINBERT, lambda: model, tmp_path / "s.jsonl")
    assert score.n_sentences == 4
    assert score.mean_probs["positive"] == pytest.approx((0.8 + 0.1 + 0.0 + 0.1) / 4)
    assert score.mean_probs["negative"] == pytest.approx((0.1 + 0.7 + 0.0 + 0.7) / 4)


def test_cached_documents_do_not_load_the_model(tmp_path: Path) -> None:
    cache = tmp_path / "s.jsonl"
    docs = [_doc("d1", "Growth is strong."), _doc("d2", "Hiring is weak.")]
    first = FakeModel()
    score_documents(docs, FINBERT, lambda: first, cache)
    assert first.calls == 2

    def no_model() -> FakeModel:
        raise AssertionError("the model was loaded although every document is cached")

    again = score_documents(docs, FINBERT, no_model, cache)
    assert [s.doc_id for s in again] == ["d1", "d2"]
    assert len(read_cache(cache)) == 2


def test_a_new_revision_is_not_served_from_the_cache(tmp_path: Path) -> None:
    cache = tmp_path / "s.jsonl"
    doc = _doc("d1", "Growth is strong.")
    score_documents([doc], FINBERT, FakeModel, cache)
    newer = EncoderSpec(**{**FINBERT.__dict__, "revision": "f" * 40})
    model = FakeModel()
    [score] = score_documents([doc], newer, lambda: model, cache)
    assert model.calls == 1
    assert score.revision == "f" * 40


def test_label_order_is_checked(tmp_path: Path) -> None:
    def swapped(sentences: Sequence[str]) -> SentenceProbs:
        out = FakeModel()(sentences)
        return out.model_copy(update={"labels": ("negative", "positive", "neutral")})

    with pytest.raises(ValueError, match="labels"):
        score_documents([_doc("d1", "Growth is strong.")], FINBERT, lambda: swapped, tmp_path / "s")
