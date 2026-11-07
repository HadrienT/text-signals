"""Answering a question from retrieved passages, with checked citations (protocol §6.8).

The frozen retriever (fusion of BM25 and the encoder, passage headers, top 8, ``as_of``
filter) gives the passages. The model reads the question and those passages only and answers in
a JSON schema: first whether it refuses, then the answer, then its citations (a passage id
and a verbatim quote). The grammar restricts a cited id to the ids shown. Code then checks that
each quote occurs in its passage, up to case, spacing and typographic quotes, the rule of the
evidence check (§6.6).

Usage: ``python -m text_signals.sources.fomc.qa_answer run --split dev [--base-url …]``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from text_signals.llm import Decoding, ExtractionRecord, LLMClient, Prompt, extract_all
from text_signals.retrieval import Hit, Passage
from text_signals.sources.fomc.extraction import _normalise

K = 8
SCHEMA_VERSION = "fomc-qa-answer-v1"
OUT = Path("data/interim/fomc/qa")


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passage_id: str = Field(description="The id of a passage shown, as written in brackets.")
    quote: str = Field(description="Words copied exactly from that passage.")


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    refused: bool = Field(description="True when the passages do not contain the answer.")
    answer: str = Field(description="The answer; empty when refused.")
    citations: list[Citation] = Field(description="Support for the answer; empty when refused.")

    @model_validator(mode="after")
    def _consistent(self) -> Answer:
        if self.refused and (self.answer.strip() or self.citations):
            raise ValueError("a refusal has no answer and no citation")
        if not self.refused and not self.citations:
            raise ValueError("an answer needs at least one citation")
        return self


QA_V1 = Prompt(
    version="fomc-qa-v1",
    system=(
        "You answer questions about documents of the Federal Open Market Committee (FOMC): "
        "statements, minutes and press conference transcripts. You are given a question and "
        "numbered passages. Each passage begins with its id in square brackets, then a line "
        "naming its document and release date, then its text.\n\n"
        "Rules:\n"
        "1. Use only the passages. Do not use anything you know from elsewhere, even if you "
        "are sure of it.\n"
        "2. If the passages do not contain the answer, refuse: set refused to true, leave "
        "answer empty and give no citations. Refusing is better than guessing.\n"
        "3. Otherwise set refused to false, answer briefly and precisely, and support every "
        "claim with at least one citation. A citation gives the passage id and a quote copied "
        "word for word from that passage's text (not from its header): one sentence or part of "
        "one, without changing, adding or dropping any word.\n"
        "4. When the question is asked as of a date, only documents released on or before that "
        "date exist. Answer as someone on that date would, from the passages shown.\n\n"
        'Reply in JSON: {"refused": bool, "answer": str, "citations": '
        '[{"passage_id": str, "quote": str}]}.'
    ),
    user_template="{text}",
)


def render(question: str, as_of: date | None, hits: list[Hit]) -> str:
    """The user message: the question (with its as-of date) and the passages with their ids."""
    when = f" (asked as of {as_of.isoformat()})" if as_of is not None else ""
    blocks = [f"[{h.passage.passage_id}]\n{h.passage.header}\n{h.passage.text}" for h in hits]
    return f"Question{when}: {question}\n\nPassages:\n\n" + "\n\n".join(blocks)


def json_schema_for(passage_ids: list[str]) -> dict[str, Any]:
    """The answer schema with ``passage_id`` restricted to the ids shown."""
    schema = Answer.model_json_schema()
    schema["$defs"]["Citation"]["properties"]["passage_id"]["enum"] = passage_ids
    return schema


class CitationCheck(BaseModel):
    passage_id: str
    in_retrieved: bool
    verbatim: bool


def check_citations(answer: Answer, shown: list[Passage]) -> list[CitationCheck]:
    """For each citation: was its passage shown, and is its quote in that passage's text?"""
    by_id = {p.passage_id: p for p in shown}
    out = []
    for c in answer.citations:
        p = by_id.get(c.passage_id)
        verbatim = (
            p is not None and bool(c.quote.strip()) and (_normalise(c.quote) in _normalise(p.text))
        )
        out.append(
            CitationCheck(passage_id=c.passage_id, in_retrieved=p is not None, verbatim=verbatim)
        )
    return out


def retrieved(split_name: str) -> list[tuple[str, date | None, str, list[Hit]]]:
    """(qid, as_of, question, hits) for a split, from the frozen retriever."""
    from text_signals.retrieval import BGE_SMALL, load_dense
    from text_signals.sources.fomc.qa import MODELS, build_index
    from text_signals.sources.fomc.qa_eval import load_questions, load_split

    embed = load_dense(BGE_SMALL, MODELS)
    index = build_index(embed)
    split = load_split()
    out = []
    for q in load_questions():
        if split[q.qid] != split_name:
            continue
        emb = embed([BGE_SMALL.query_prefix + q.question])[0]
        hits = index.search(q.question, k=K, as_of=q.as_of, method="hybrid", query_embedding=emb)
        out.append((q.qid, q.as_of, q.question, hits))
    return out


def run(split_name: str, base_url: str) -> list[dict[str, Any]]:
    """Answer every question of a split; cache the model records, write the checked answers."""
    client = LLMClient(base_url)
    model, model_path = client.model_info()
    decoding = Decoding()
    items = retrieved(split_name)
    shown = {qid: hits for qid, _, _, hits in items}
    texts = {qid: render(question, as_of, hits) for qid, as_of, question, hits in items}

    def ask(qid: str, text: str) -> ExtractionRecord:
        return client.extract(
            doc_id=qid,
            text=text,
            prompt=QA_V1,
            schema=Answer,
            schema_version=SCHEMA_VERSION,
            decoding=decoding,
            model=model,
            model_path=model_path,
            json_schema=json_schema_for([h.passage.passage_id for h in shown[qid]]),
        )

    records = extract_all(
        texts.items(),
        ask,
        model=model,
        prompt_version=QA_V1.version,
        schema_version=SCHEMA_VERSION,
        cache_path=OUT / f"answers-{model}.jsonl",
    )
    rows = []
    for r in records:
        passages = [h.passage for h in shown[r.doc_id]]
        answer = Answer.model_validate(r.output) if r.output is not None else None
        checks = [] if answer is None else check_citations(answer, passages)
        rows.append(
            {
                "qid": r.doc_id,
                "model": r.model,
                "prompt_version": r.prompt_version,
                "retrieved": [p.passage_id for p in passages],
                "valid": answer is not None,
                "error": r.error,
                "refused": None if answer is None else answer.refused,
                "answer": None if answer is None else answer.answer,
                "citations": [] if answer is None else [c.model_dump() for c in answer.citations],
                "checks": [c.model_dump() for c in checks],
                "latency_s": r.latency_s,
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"answers-{split_name}.json").write_text(json.dumps(rows, indent=1))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=["run"])
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    rows = run(args.split, args.base_url)
    invalid = sum(not r["valid"] for r in rows)
    refused = sum(bool(r["refused"]) for r in rows)
    checks = [c for r in rows for c in r["checks"]]
    verbatim = sum(c["verbatim"] for c in checks)
    print(
        f"{len(rows)} questions, {invalid} invalid, {refused} refused; "
        f"{verbatim}/{len(checks)} citations verbatim"
    )


if __name__ == "__main__":
    main()
