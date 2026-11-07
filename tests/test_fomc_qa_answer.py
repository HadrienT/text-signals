from datetime import date

import pytest
from pydantic import ValidationError

from text_signals.retrieval import Hit, Passage
from text_signals.sources.fomc.qa_answer import (
    Answer,
    Citation,
    check_citations,
    json_schema_for,
    render,
)

P1 = Passage(
    "fomc-statement-20081216#p4",
    "fomc-statement-20081216",
    "statement",
    date(2008, 12, 16),
    "In particular, the Committee anticipates that weak economic conditions are likely to "
    "warrant exceptionally low levels of the federal funds rate for some time.",
    "FOMC statement released December 16, 2008.",
)


def test_answer_schema_enforces_refusal_consistency() -> None:
    Answer(refused=True, answer="", citations=[])
    Answer(refused=False, answer="x", citations=[Citation(passage_id="a", quote="b")])
    with pytest.raises(ValidationError):
        Answer(refused=True, answer="something", citations=[])
    with pytest.raises(ValidationError):
        Answer(refused=False, answer="x", citations=[])


def test_render_shows_ids_headers_and_the_as_of_date() -> None:
    text = render("How long?", date(2009, 1, 31), [Hit(P1, 1)])
    assert text.startswith("Question (asked as of 2009-01-31): How long?")
    assert "[fomc-statement-20081216#p4]\nFOMC statement released December 16, 2008.\n" in text
    assert "asked as of" not in render("How long?", None, [Hit(P1, 1)])


def test_json_schema_restricts_cited_ids_to_those_shown() -> None:
    schema = json_schema_for(["a#p1", "b#p2"])
    assert schema["$defs"]["Citation"]["properties"]["passage_id"]["enum"] == ["a#p1", "b#p2"]
    assert list(schema["properties"]) == ["refused", "answer", "citations"]


def test_citation_check_is_verbatim_up_to_case_spacing_and_quotes() -> None:
    good = Citation(
        passage_id=P1.passage_id,
        quote="exceptionally  low levels of the Federal funds rate for some time",
    )
    paraphrase = Citation(passage_id=P1.passage_id, quote="rates will stay very low for a while")
    header_only = Citation(
        passage_id=P1.passage_id, quote="FOMC statement released December 16, 2008"
    )
    unknown = Citation(passage_id="other#p1", quote="for some time")
    a = Answer(
        refused=False, answer="For some time.", citations=[good, paraphrase, header_only, unknown]
    )
    checks = check_citations(a, [P1])
    assert [c.verbatim for c in checks] == [True, False, False, False]
    assert [c.in_retrieved for c in checks] == [True, True, True, False]
