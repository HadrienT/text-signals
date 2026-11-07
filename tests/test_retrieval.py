from dataclasses import dataclass
from datetime import date

import numpy as np

from text_signals.retrieval import (
    BM25,
    MIN_WORDS,
    PIECE_WORDS,
    Index,
    Passage,
    passages,
    passages_sha256,
    rrf,
    tokenize,
)


@dataclass(frozen=True)
class Doc:
    doc_id: str
    doc_type: str
    release_date: date
    text: str


def test_wrapped_lines_are_joined_and_short_units_dropped() -> None:
    text = (
        "Minutes of the Federal Open Market Committee\n\n"
        "The Committee discussed the outlook for\n\n"
        "inflation and agreed to keep the rate unchanged.\n\n"
        "MICHELLE SMITH. Victoria.\n\n"
        "Growth was solid in the quarter and the labor market strengthened."
    )
    out = passages(Doc("d", "minutes", date(2001, 1, 31), text))
    texts = [p.text for p in out]
    # the header has no terminal punctuation, so it is joined to the wrapped sentence
    assert texts[0].endswith(
        "discussed the outlook for inflation and agreed to keep the rate unchanged."
    )
    assert "MICHELLE SMITH. Victoria." not in texts  # fewer than MIN_WORDS words
    assert [p.passage_id for p in out] == ["d#p1", "d#p2"]
    assert all(len(t.split()) >= MIN_WORDS for t in texts)


def test_long_units_are_cut_at_sentence_boundaries() -> None:
    sentence = "The Committee judged that the economy expanded at a moderate pace this year. "
    text = sentence * 40  # 520 words, one block
    out = passages(Doc("d", "press_conference", date(2020, 1, 29), text))
    assert len(out) > 1
    assert all(len(p.text.split()) <= PIECE_WORDS for p in out)
    assert all(p.text.endswith("this year.") for p in out)


def test_tokenize_keeps_numbers_and_hyphenated_terms() -> None:
    assert tokenize("Raise to 1-1/4 percent; mortgage-backed securities.") == [
        "raise",
        "to",
        "1-1",
        "4",
        "percent",
        "mortgage-backed",
        "securities",
    ]


def test_bm25_ranks_the_matching_passage_first_and_weights_rare_terms() -> None:
    docs = [
        tokenize("inflation remains elevated"),
        tokenize("the labor market remains strong"),
        tokenize("the committee will keep the rate for an extended period"),
    ]
    bm = BM25(docs)
    s = bm.scores(tokenize("extended period"))
    assert int(np.argmax(s)) == 2 and s[0] == 0
    assert bm.idf("extended") > bm.idf("remains")


def test_rrf_combines_ranks() -> None:
    assert rrf([[1, 2, 3], [3, 1, 2]]) == [1, 3, 2]


def _index() -> Index:
    items = [
        Passage("a#p1", "a", "statement", date(2008, 12, 16), "rates low for some time"),
        Passage("b#p1", "b", "statement", date(2009, 3, 18), "rates low for an extended period"),
        Passage("c#p1", "c", "statement", date(2011, 8, 9), "rates low at least through mid-2013"),
    ]
    emb = np.eye(3, dtype=np.float32)
    return Index(items, emb)


def test_as_of_excludes_later_passages() -> None:
    index = _index()
    hits = index.search("rates low extended period", as_of=date(2009, 1, 31), method="bm25")
    assert [h.passage.doc_id for h in hits] == ["a"]
    hits = index.search("rates low extended period", method="bm25")
    assert hits[0].passage.doc_id == "b"


def test_dense_and_hybrid_search_use_the_query_embedding() -> None:
    index = _index()
    q = np.array([0, 0, 1], dtype=np.float32)
    dense = index.search("x", method="dense", query_embedding=q)
    assert dense[0].passage.doc_id == "c"
    hybrid = index.search("extended period", method="hybrid", query_embedding=q, k=2)
    assert {h.passage.doc_id for h in hybrid} == {"b", "c"}
    assert [h.rank for h in hybrid] == [1, 2]
    filtered = index.search("x", method="dense", query_embedding=q, as_of=date(2010, 1, 1))
    assert "c" not in {h.passage.doc_id for h in filtered}


def test_passage_fingerprint_changes_with_text() -> None:
    a = [Passage("a#p1", "a", "statement", date(2000, 1, 1), "one")]
    b = [Passage("a#p1", "a", "statement", date(2000, 1, 1), "two")]
    assert passages_sha256(a) != passages_sha256(b)


def test_header_is_indexed_but_not_part_of_the_text() -> None:
    doc = Doc("s", "statement", date(2026, 7, 29), "The vote was 9 to 3 on this statement today.")
    (p,) = passages(doc, "FOMC statement released July 29, 2026.")
    assert p.text == "The vote was 9 to 3 on this statement today."
    assert p.indexed_text().startswith("FOMC statement released July 29, 2026.")
    assert p.indexed_text(with_header=False) == p.text
    assert passages_sha256([p]) != passages_sha256([p], with_header=False)
    with_h = Index([p], with_header=True).search("July 2026", method="bm25")
    without = Index([p], with_header=False).search("July 2026", method="bm25")
    assert len(with_h) == 1 and without == []
