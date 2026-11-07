"""The evaluation set of the FOMC retrieval agent (protocol §6.8).

``data/labels/fomc_qa_v1.json`` holds the hand-written questions with their reference answers
and passages. ``validate`` checks them against the index: every reference passage exists and,
for a question asked ``as_of`` a date, was released on or before it; unanswerable questions have
no reference. ``assign_split`` draws the development questions, two of each type, with a fixed
seed; the split is stored in ``data/labels/fomc_qa_v1_split.json`` before any system answer.

Usage: ``python -m text_signals.sources.fomc.qa_eval split``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from text_signals.retrieval import Passage

QUESTIONS = Path("data/labels/fomc_qa_v1.json")
SPLIT = Path("data/labels/fomc_qa_v1_split.json")
VERSION = "fomc-qa-v1"
SPLIT_SEED = 20261109
DEV_PER_TYPE = 2

QuestionType = Literal["lookup", "change", "press_conference", "minutes", "unanswerable"]


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    qid: str
    type: QuestionType
    as_of: date | None
    answerable: bool
    question: str
    reference_answer: str
    reference_passages: list[str]


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["version"] != VERSION:
        raise ValueError(f"unexpected version {data['version']}")
    return [Question.model_validate(q) for q in data["questions"]]


def validate(questions: list[Question], passages: list[Passage]) -> list[str]:
    """Problems with the set; an empty list means it is consistent with the index."""
    by_id = {p.passage_id: p for p in passages}
    problems = []
    if len({q.qid for q in questions}) != len(questions):
        problems.append("duplicate qid")
    for q in questions:
        if q.answerable != (q.type != "unanswerable"):
            problems.append(f"{q.qid}: answerable flag does not match the type")
        if not q.answerable and q.reference_passages:
            problems.append(f"{q.qid}: an unanswerable question has reference passages")
        if q.answerable and not q.reference_passages:
            problems.append(f"{q.qid}: no reference passage")
        for pid in q.reference_passages:
            p = by_id.get(pid)
            if p is None:
                problems.append(f"{q.qid}: unknown passage {pid}")
            elif q.as_of is not None and p.release_date > q.as_of:
                problems.append(f"{q.qid}: {pid} released after as_of {q.as_of}")
    return problems


def assign_split(questions: list[Question], seed: int = SPLIT_SEED) -> dict[str, str]:
    """qid -> "dev" or "test": ``DEV_PER_TYPE`` development questions drawn from each type."""
    rng = np.random.default_rng(seed)
    split = {q.qid: "test" for q in questions}
    for t in sorted({q.type for q in questions}):
        pool = sorted(q.qid for q in questions if q.type == t)
        for i in rng.choice(len(pool), size=DEV_PER_TYPE, replace=False):
            split[pool[int(i)]] = "dev"
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=["split"])
    parser.parse_args()
    from text_signals.sources.fomc.qa import all_passages, load_documents

    questions = load_questions()
    problems = validate(questions, all_passages(load_documents()))
    if problems:
        raise SystemExit("\n".join(problems))
    split = assign_split(questions)
    SPLIT.write_text(
        json.dumps({"version": VERSION, "seed": SPLIT_SEED, "split": split}, indent=1) + "\n"
    )
    dev = sorted(q for q, s in split.items() if s == "dev")
    print(f"{len(questions)} questions valid; dev: {dev} -> {SPLIT}")


if __name__ == "__main__":
    main()
