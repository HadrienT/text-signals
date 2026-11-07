"""The FOMC retrieval index and question answering (protocol §6.8).

Passages come from the statements, the minutes and the press conference transcripts. Their
embeddings are cached under ``data/interim/fomc/retrieval``, keyed by the encoder revision and a
fingerprint of the passage set, so a rebuild with the same inputs does no inference.

Usage:
  ``python -m text_signals.sources.fomc.qa index``: build the passages and their embeddings;
  ``python -m text_signals.sources.fomc.qa search "query" [--as-of YYYY-MM-DD] [--method …]``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from text_signals.documents import Document
from text_signals.retrieval import (
    BGE_SMALL,
    Embed,
    FloatArray,
    Index,
    Passage,
    load_dense,
    passages,
    passages_sha256,
)

CORPUS = Path("data/interim/fomc")
SOURCES = ("statements", "minutes", "press_conferences")
OUT = CORPUS / "retrieval"
MODELS = Path("data/raw/huggingface")


def load_documents(root: Path = CORPUS) -> list[Document]:
    docs = []
    for name in SOURCES:
        with (root / f"{name}.jsonl").open(encoding="utf-8") as fh:
            docs += [Document.model_validate_json(line) for line in fh if line.strip()]
    return sorted(docs, key=lambda d: (d.release_date, d.doc_id))


def all_passages(docs: list[Document]) -> list[Passage]:
    return [p for d in docs for p in passages(d)]


def embeddings(items: list[Passage], embed: Embed | None = None) -> FloatArray:
    """Passage embeddings, from the cache when the passages and the encoder are unchanged."""
    key = f"{BGE_SMALL.revision[:8]}-{passages_sha256(items)[:16]}"
    path = OUT / f"embeddings-{key}.npy"
    if path.exists():
        matrix: FloatArray = np.load(path).astype(np.float32)
        return matrix
    if embed is None:
        embed = load_dense(BGE_SMALL, MODELS)
    matrix = embed([p.text for p in items])
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    (OUT / f"embeddings-{key}.json").write_text(
        json.dumps(
            {
                "model_id": BGE_SMALL.model_id,
                "revision": BGE_SMALL.revision,
                "weights_sha256": BGE_SMALL.weights_sha256,
                "n_passages": len(items),
                "passages_sha256": passages_sha256(items),
            },
            indent=1,
        )
    )
    return matrix


def build_index(embed: Embed | None = None) -> Index:
    items = all_passages(load_documents())
    return Index(items, embeddings(items, embed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=["index", "search"])
    parser.add_argument("query", nargs="?")
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("-k", type=int, default=8)
    args = parser.parse_args()
    if args.cmd == "index":
        docs = load_documents()
        items = all_passages(docs)
        by_type: dict[str, int] = {}
        for p in items:
            by_type[p.doc_type] = by_type.get(p.doc_type, 0) + 1
        words = [len(p.text.split()) for p in items]
        print(f"{len(docs)} documents, {len(items)} passages {by_type}")
        print(f"words per passage: median {int(np.median(words))}, max {max(words)}")
        matrix = embeddings(items)
        print(f"embeddings {matrix.shape}")
        return
    embed = load_dense(BGE_SMALL, MODELS)
    index = build_index(embed)
    q = embed([BGE_SMALL.query_prefix + args.query])[0]
    for hit in index.search(
        args.query, k=args.k, as_of=args.as_of, method=args.method, query_embedding=q
    ):
        p = hit.passage
        print(f"{hit.rank}. {p.passage_id} ({p.release_date}) {p.text[:200]}")


if __name__ == "__main__":
    main()
