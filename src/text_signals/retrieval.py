"""Passage retrieval over a document corpus: BM25, a dense encoder, and their fusion.

The unit is the **passage** (protocol §6.8), cut from a document's parsed text by four rules:

1. blocks separated by a blank line;
2. a block that does not end with terminal punctuation is joined to the next one (lines
   wrapped in the source, as in the older minutes);
3. a unit longer than ``MAX_WORDS`` is cut at sentence boundaries into pieces of at most
   ``PIECE_WORDS`` words (a dense encoder reads about 380 words at most);
4. units shorter than ``MIN_WORDS`` words are not indexed (headers, a moderator's "Victoria.").

A passage keeps its document's release date, and every search can carry an ``as_of`` date: only
passages released on or before it are ranked, so a question asked as of a date gets no help
from what was written later (the point-in-time rule of the study).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from text_signals.encoders import split_sentences

PASSAGE_VERSION = "1"
MAX_WORDS = 250
PIECE_WORDS = 200
MIN_WORDS = 5
_TERMINAL = re.compile(r"[.?!:;\"”’)\]]$")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.'’-][a-z0-9]+)*")

FloatArray = npt.NDArray[np.float32]


class HasText(Protocol):
    @property
    def doc_id(self) -> str: ...
    @property
    def doc_type(self) -> str: ...
    @property
    def release_date(self) -> date: ...
    @property
    def text(self) -> str: ...


@dataclass(frozen=True)
class Passage:
    passage_id: str  # "<doc_id>#p<n>", n from 1 in reading order
    doc_id: str
    doc_type: str
    release_date: date
    text: str


def _blocks(text: str) -> list[str]:
    out: list[str] = []
    pending: list[str] = []
    for block in (b.strip() for b in text.split("\n\n")):
        if not block:
            continue
        pending.append(" ".join(block.split()))
        if _TERMINAL.search(block):
            out.append(" ".join(pending))
            pending = []
    if pending:
        out.append(" ".join(pending))
    return out


def _pieces(unit: str) -> list[str]:
    if len(unit.split()) <= MAX_WORDS:
        return [unit]
    out: list[str] = []
    current: list[str] = []
    words = 0
    for sentence in split_sentences(unit):
        n = len(sentence.split())
        if current and words + n > PIECE_WORDS:
            out.append(" ".join(current))
            current, words = [], 0
        current.append(sentence)
        words += n
    if current:
        out.append(" ".join(current))
    return out


def passages(doc: HasText) -> list[Passage]:
    """The indexed passages of one document, in reading order."""
    units = [p for b in _blocks(doc.text) for p in _pieces(b) if len(p.split()) >= MIN_WORDS]
    return [
        Passage(f"{doc.doc_id}#p{i}", doc.doc_id, doc.doc_type, doc.release_date, text)
        for i, text in enumerate(units, start=1)
    ]


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 over tokenised passages (k1 = 1.2, b = 0.75, protocol §6.8)."""

    def __init__(self, docs: Sequence[Sequence[str]], *, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lengths = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg_length = float(self.lengths.mean()) if self.n else 0.0
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                postings[term].append((i, tf))
        self.postings = {
            t: (np.array([i for i, _ in p]), np.array([tf for _, tf in p], dtype=np.float32))
            for t, p in postings.items()
        }

    def idf(self, term: str) -> float:
        df = len(self.postings[term][0]) if term in self.postings else 0
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: Sequence[str]) -> FloatArray:
        out = np.zeros(self.n, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg_length)
        for term in set(query):
            if term not in self.postings:
                continue
            idx, tf = self.postings[term]
            out[idx] += self.idf(term) * tf * (self.k1 + 1) / (tf + norm[idx])
        return out


def rrf(rankings: Sequence[Sequence[int]], *, k: int = 60) -> list[int]:
    """Reciprocal rank fusion: Σ 1 / (k + rank), rank from 1; ties broken by first appearance."""
    score: dict[int, float] = defaultdict(float)
    order: dict[int, int] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            score[item] += 1 / (k + rank)
            order.setdefault(item, len(order))
    return sorted(score, key=lambda i: (-score[i], order[i]))


@dataclass(frozen=True)
class DenseSpec:
    """A pinned sentence-embedding model on the Hugging Face Hub (CLS pooling, normalised)."""

    model_id: str
    revision: str
    weights_file: str
    weights_sha256: str
    query_prefix: str
    max_length: int = 512


# BAAI/bge-small-en-v1.5 (MIT), revision of 2024-02-22; query instruction from its model card.
BGE_SMALL = DenseSpec(
    model_id="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    weights_file="model.safetensors",
    weights_sha256="3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
    query_prefix="Represent this sentence for searching relevant passages: ",
)

Embed = Callable[[Sequence[str]], FloatArray]


def load_dense(
    spec: DenseSpec, cache_dir: Path, *, device: str | None = None, batch_size: int = 64
) -> Embed:
    """The pinned encoder (weights sha256 checked), returning L2-normalised CLS embeddings."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, AutoTokenizer

    weights = Path(
        hf_hub_download(
            spec.model_id, spec.weights_file, revision=spec.revision, cache_dir=cache_dir
        )
    )
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != spec.weights_sha256:
        raise ValueError(
            f"{spec.model_id}: weights sha256 {digest}, expected {spec.weights_sha256}"
        )
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        spec.model_id, revision=spec.revision, cache_dir=cache_dir
    )
    model = AutoModel.from_pretrained(
        spec.model_id, revision=spec.revision, cache_dir=cache_dir, use_safetensors=True
    )
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    model.eval()

    def embed(texts: Sequence[str]) -> FloatArray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                list(texts[i : i + batch_size]),
                padding=True,
                truncation=True,
                max_length=spec.max_length,
                return_tensors="pt",
            ).to(dev)
            with torch.no_grad():
                cls = model(**enc).last_hidden_state[:, 0].float()
            out.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 0), np.float32)

    return embed


Method = Literal["bm25", "dense", "hybrid"]


@dataclass(frozen=True)
class Hit:
    passage: Passage
    rank: int


class Index:
    """BM25 and dense embeddings over the same passages, searched with an ``as_of`` filter."""

    def __init__(self, items: Sequence[Passage], embeddings: FloatArray | None = None):
        self.passages = list(items)
        self.bm25 = BM25([tokenize(p.text) for p in self.passages])
        self.embeddings = embeddings
        self.dates = np.array([p.release_date.toordinal() for p in self.passages])

    def _ranking(self, scores: FloatArray, allowed: npt.NDArray[np.bool_], n: int) -> list[int]:
        masked = np.where(allowed, scores, -np.inf)
        top = np.argsort(-masked, kind="stable")[:n]
        return [int(i) for i in top if np.isfinite(masked[i])]

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        as_of: date | None = None,
        method: Method = "hybrid",
        query_embedding: FloatArray | None = None,
        candidates: int = 50,
    ) -> list[Hit]:
        """The top ``k`` passages released on or before ``as_of`` (all when ``None``)."""
        allowed = (
            np.ones(len(self.passages), dtype=bool)
            if as_of is None
            else self.dates <= as_of.toordinal()
        )
        rankings = []
        if method in ("bm25", "hybrid"):
            rankings.append(self._ranking(self.bm25.scores(tokenize(query)), allowed, candidates))
        if method in ("dense", "hybrid"):
            if self.embeddings is None or query_embedding is None:
                raise ValueError("dense search needs passage and query embeddings")
            dense = self.embeddings @ query_embedding.reshape(-1)
            rankings.append(self._ranking(dense.astype(np.float32), allowed, candidates))
        order = rankings[0] if len(rankings) == 1 else rrf(rankings)
        return [Hit(self.passages[i], r) for r, i in enumerate(order[:k], start=1)]


def passages_sha256(items: Sequence[Passage]) -> str:
    """A fingerprint of the passage set, to key an embedding cache."""
    h = hashlib.sha256(PASSAGE_VERSION.encode())
    for p in items:
        h.update(p.passage_id.encode())
        h.update(b"\0")
        h.update(p.text.encode())
        h.update(b"\0")
    return h.hexdigest()
