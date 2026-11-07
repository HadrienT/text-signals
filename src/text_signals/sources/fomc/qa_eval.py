"""The evaluation set of the FOMC retrieval agent (protocol §6.8).

``data/labels/fomc_qa_v1.json`` holds the hand-written questions with their reference answers
and passages. ``validate`` checks them against the index: every reference passage exists and,
for a question asked ``as_of`` a date, was released on or before it; unanswerable questions have
no reference. ``assign_split`` draws the development questions, two of each type, with a fixed
seed; the split is stored in ``data/labels/fomc_qa_v1_split.json`` before any system answer.

Usage: ``python -m text_signals.sources.fomc.qa_eval {split,retrieval} [--split dev|test]``.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

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


KS = (1, 5, 8)


def first_hit_rank(retrieved: list[str], references: list[str]) -> int | None:
    """Rank (from 1) of the first reference passage among the retrieved, None if absent."""
    refs = set(references)
    return next((i for i, pid in enumerate(retrieved, start=1) if pid in refs), None)


def retrieval_metrics(ranks: list[int | None], ks: tuple[int, ...] = KS) -> dict[str, float]:
    """recall@k (share with a reference in the top k) and MRR@max(k) (0 when none in it)."""
    n = len(ranks)
    out = {"n": float(n)}
    for k in ks:
        out[f"recall@{k}"] = sum(r is not None and r <= k for r in ranks) / n
    top = max(ks)
    out[f"mrr@{top}"] = sum(1 / r for r in ranks if r is not None and r <= top) / n
    return out


CUTOFF = date(2024, 6, 30)  # Phi-4's training-data cutoff (§5.2)


def _doc_date(passage_id: str) -> date:
    """The release date in a passage id, e.g. ``fomc-statement-20260729#p1``."""
    match = re.search(r"(\d{8})#", passage_id)
    if match is None:
        raise ValueError(f"no date in {passage_id}")
    d = match.group(1)
    return date(int(d[:4]), int(d[4:6]), int(d[6:]))


def answer_report(
    questions: list[Question], rows: list[dict[str, Any]], judgments: list[dict[str, str]]
) -> dict[str, float]:
    """The answering metrics of §6.8 from the checked answers and the judgments."""
    q = {x.qid: x for x in questions}
    j = {x["qid"]: x for x in judgments}
    answerable = [r for r in rows if q[r["qid"]].answerable]
    unanswerable = [r for r in rows if not q[r["qid"]].answerable]
    answered = [r for r in answerable if not r["refused"]]
    checks = [c for r in rows if not r["refused"] for c in r["checks"]]

    def share(items: list[Any], pred: Any) -> float:
        return sum(1 for i in items if pred(i)) / len(items) if items else float("nan")

    def after(r: dict[str, Any]) -> bool:
        """Is a reference passage from a document released after the cutoff?"""
        return any(_doc_date(pid) > CUTOFF for pid in q[r["qid"]].reference_passages)

    out = {
        "n_answerable": float(len(answerable)),
        "false_refusals": float(len(answerable) - len(answered)),
        "n_answered": float(len(answered)),
        "n_unanswerable": float(len(unanswerable)),
        "correct_refusals": float(sum(bool(r["refused"]) for r in unanswerable)),
        "n_citations": float(len(checks)),
        "citations_strict": share(checks, lambda c: c["verbatim"]),
    }
    for cat in ("exact", "punctuation", "wrong_passage", "not_found"):
        out[f"citations_{cat}"] = float(sum(c["category"] == cat for c in checks))
    for scale, values in (
        ("correctness", ("right", "partial", "wrong")),
        ("faithfulness", ("supported", "partly", "unsupported")),
    ):
        for v in values:
            out[f"{scale}_{v}"] = float(sum(j[r["qid"]].get(scale) == v for r in answered))
    for name, sub in (
        ("before_cutoff", [r for r in answerable if not after(r)]),
        ("after_cutoff", [r for r in answerable if after(r)]),
    ):
        out[f"n_{name}"] = float(len(sub))
        out[f"right_{name}"] = float(
            sum(not r["refused"] and j[r["qid"]].get("correctness") == "right" for r in sub)
        )
    return out


def load_split(path: Path = SPLIT) -> dict[str, str]:
    split: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))["split"]
    return split


RETRIEVAL_OUT = Path("data/interim/fomc/retrieval")


def run_retrieval(split_name: str) -> dict[str, dict[str, float]]:
    """recall@k and MRR of BM25, the encoder and their fusion on one split (answerable only),
    with and without the passage headers."""
    from text_signals.retrieval import BGE_SMALL, Method, load_dense
    from text_signals.sources.fomc.qa import MODELS, build_index

    embed = load_dense(BGE_SMALL, MODELS)
    indexes = {h: build_index(embed, with_header=h) for h in (True, False)}
    split = load_split()
    qs = [q for q in load_questions() if split[q.qid] == split_name and q.answerable]
    rows = []
    out = {}
    methods: tuple[Method, ...] = ("bm25", "dense", "hybrid")
    for with_header, method in [(h, m) for h in (True, False) for m in methods]:
        index = indexes[with_header]
        name = f"{method}{'' if with_header else '-noheader'}"
        ranks: list[int | None] = []
        for q in qs:
            emb = embed([BGE_SMALL.query_prefix + q.question])[0]
            hits = index.search(
                q.question, k=max(KS), as_of=q.as_of, method=method, query_embedding=emb
            )
            ids = [h.passage.passage_id for h in hits]
            rank = first_hit_rank(ids, q.reference_passages)
            ranks.append(rank)
            rows.append({"qid": q.qid, "method": name, "rank": rank, "retrieved": ids})
        out[name] = retrieval_metrics(ranks)
    RETRIEVAL_OUT.mkdir(parents=True, exist_ok=True)
    (RETRIEVAL_OUT / f"retrieval_{split_name}.json").write_text(
        json.dumps({"metrics": out, "rows": rows}, indent=1)
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=["split", "retrieval", "answers"])
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()
    if args.cmd == "answers":
        rows = json.loads(
            Path(f"data/interim/fomc/qa/answers-{args.split}-fomc-qa-v2.json").read_text()
        )
        judged = json.loads(
            Path(f"data/labels/fomc_qa_v1_judgments_{args.split}.json").read_text()
        )["judgments"]
        for k, v in answer_report(load_questions(), rows, judged).items():
            print(f"{k}: {v:.3f}" if k.startswith("citations_strict") else f"{k}: {v:.0f}")
        return
    if args.cmd == "retrieval":
        for method, m in run_retrieval(args.split).items():
            print(method, {k: round(v, 3) for k, v in m.items()})
        return
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
