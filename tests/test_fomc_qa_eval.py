from datetime import date

from text_signals.retrieval import Passage
from text_signals.sources.fomc.qa_eval import Question, assign_split, load_questions, validate

P = [
    Passage("s1#p1", "s1", "statement", date(2008, 12, 16), "0 to 1/4 percent"),
    Passage("s2#p1", "s2", "statement", date(2009, 3, 18), "for an extended period"),
]


def _q(
    qid: str, refs: list[str], *, as_of: date | None = None, answerable: bool = True
) -> Question:
    return Question(
        qid=qid,
        type="lookup" if answerable else "unanswerable",
        as_of=as_of,
        answerable=answerable,
        question="?",
        reference_answer="!",
        reference_passages=refs,
    )


def test_validate_accepts_a_consistent_set() -> None:
    qs = [
        _q("a", ["s1#p1"]),
        _q("b", ["s1#p1"], as_of=date(2009, 1, 1)),
        _q("u", [], answerable=False),
    ]
    assert validate(qs, P) == []


def test_validate_flags_unknown_late_and_misplaced_references() -> None:
    qs = [
        _q("a", ["nope#p1"]),
        _q("b", ["s2#p1"], as_of=date(2009, 1, 1)),
        _q("c", []),
        _q("u", ["s1#p1"], answerable=False),
    ]
    problems = validate(qs, P)
    assert any("unknown passage" in p for p in problems)
    assert any("after as_of" in p for p in problems)
    assert any("c: no reference" in p for p in problems)
    assert any("u: an unanswerable" in p for p in problems)


def test_split_draws_two_dev_questions_per_type_reproducibly() -> None:
    qs = load_questions()
    split = assign_split(qs)
    assert split == assign_split(qs)
    by_type: dict[str, int] = {}
    for q in qs:
        if split[q.qid] == "dev":
            by_type[q.type] = by_type.get(q.type, 0) + 1
    assert by_type == {t: 2 for t in {q.type for q in qs}}
    assert sum(s == "test" for s in split.values()) == len(qs) - 10
