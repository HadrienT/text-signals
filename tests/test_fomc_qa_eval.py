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


def test_first_hit_rank_and_metrics() -> None:
    from text_signals.sources.fomc.qa_eval import first_hit_rank, retrieval_metrics

    assert first_hit_rank(["a", "b", "c"], ["c", "b"]) == 2
    assert first_hit_rank(["a"], ["z"]) is None
    m = retrieval_metrics([1, 3, None, 9])
    assert m["recall@1"] == 0.25 and m["recall@5"] == 0.5 and m["recall@8"] == 0.5
    assert m["mrr@8"] == (1 + 1 / 3) / 4


def test_answer_report_counts_refusals_judgments_and_cutoff() -> None:
    from text_signals.sources.fomc.qa_eval import answer_report

    qs = [
        Question(
            qid="a",
            type="lookup",
            as_of=None,
            answerable=True,
            question="?",
            reference_answer="!",
            reference_passages=["fomc-statement-20081216#p1"],
        ),
        Question(
            qid="b",
            type="lookup",
            as_of=None,
            answerable=True,
            question="?",
            reference_answer="!",
            reference_passages=["fomc-statement-20260729#p1"],
        ),
        Question(
            qid="c",
            type="lookup",
            as_of=None,
            answerable=True,
            question="?",
            reference_answer="!",
            reference_passages=["fomc-statement-20260729#p5"],
        ),
        Question(
            qid="u",
            type="unanswerable",
            as_of=None,
            answerable=False,
            question="?",
            reference_answer="!",
            reference_passages=[],
        ),
    ]
    exact = {"verbatim": True, "category": "exact"}
    punct = {"verbatim": False, "category": "punctuation"}
    rows = [
        {"qid": "a", "refused": False, "checks": [exact, punct]},
        {"qid": "b", "refused": False, "checks": [exact]},
        {"qid": "c", "refused": True, "checks": []},
        {"qid": "u", "refused": True, "checks": []},
    ]
    judged = [
        {"qid": "a", "correctness": "right", "faithfulness": "supported"},
        {"qid": "b", "correctness": "wrong", "faithfulness": "unsupported"},
    ]
    r = answer_report(qs, rows, judged)
    assert r["false_refusals"] == 1 and r["correct_refusals"] == 1
    assert r["citations_strict"] == 2 / 3 and r["citations_punctuation"] == 1
    assert r["correctness_right"] == 1 and r["faithfulness_unsupported"] == 1
    assert r["n_before_cutoff"] == 1 and r["right_before_cutoff"] == 1
    assert r["n_after_cutoff"] == 2 and r["right_after_cutoff"] == 0
