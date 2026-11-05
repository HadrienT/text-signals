"""Sentence-level encoder baselines: split a document into sentences, classify each one with a
pinned Hugging Face model, and average the class probabilities over the document.

The model is pinned three ways: the repository ``revision`` (a commit hash), the sha256 of the
weights file, and the label order read from its config. A change in any of them raises instead
of silently scoring with a different model. Scores are cached per (text sha256, model,
revision, splitter version), so a long run can be resumed and a rerun does no inference.

The document score is defined in the protocol (§5.2) and fixed before any regression.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from text_signals.documents import Document

# Bump when ``split_sentences`` changes: cached scores of the old version are then ignored.
SPLITTER_VERSION = "1"

# Tokens ending in a period that do not end a sentence in Fed texts.
_ABBREVIATIONS = frozenset(
    {"U.S.", "a.m.", "p.m.", "Jr.", "Mr.", "Ms.", "Mrs.", "Dr.", "St.", "No."}
)
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“(])")
_INITIAL = re.compile(r"(?:^|\s)[A-Z]\.$")


def split_sentences(text: str) -> list[str]:
    """Sentences of ``text``: paragraphs first, then sentence ends inside each paragraph.

    A period followed by a capital letter ends a sentence unless it closes a known abbreviation
    ("U.S.") or a middle initial ("Lisa D. Cook"). Whitespace inside a sentence is collapsed.
    """
    # Some older pages break a line inside a name ("Robert T.<br>Parry"): a paragraph that ends
    # on an initial or an abbreviation continues in the next one.
    paragraphs: list[str] = []
    for chunk in re.split(r"\n\s*\n", text):
        chunk = " ".join(chunk.split())
        if not chunk:
            continue
        if paragraphs and _continues(paragraphs[-1]):
            paragraphs[-1] += " " + chunk
        else:
            paragraphs.append(chunk)
    sentences: list[str] = []
    for paragraph in paragraphs:
        start = 0
        for m in _BOUNDARY.finditer(paragraph):
            head = paragraph[start : m.start()]
            if _continues(head):
                continue
            sentences.append(head)
            start = m.end()
        sentences.append(paragraph[start:])
    return [s for s in sentences if s]


def _continues(head: str) -> bool:
    """Whether ``head`` ends on an abbreviation or an initial rather than a sentence end."""
    return head.rsplit(" ", 1)[-1] in _ABBREVIATIONS or bool(_INITIAL.search(head))


@dataclass(frozen=True)
class EncoderSpec:
    """A pinned sentence classifier on the Hugging Face Hub."""

    name: str  # short name used in column names, e.g. "finbert"
    model_id: str
    revision: str  # commit hash of the model repository
    weights_file: str
    weights_sha256: str
    labels: tuple[str, ...]  # class names in the order of the model's output logits
    max_length: int = 512  # tokens; longer sentences are truncated and counted


# FinBERT (Araci, 2019): BERT-base further pre-trained on Reuters TRC2 and fine-tuned on the
# Financial PhraseBank (Malo et al., 2014). Revision of 2026-10-12, whose model.safetensors is
# a conversion of the 2020 pytorch_model.bin (checked tensor by tensor, protocol §5.2).
FINBERT = EncoderSpec(
    name="finbert",
    model_id="ProsusAI/finbert",
    revision="0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031",
    weights_file="model.safetensors",
    weights_sha256="e5897858ff819aad7629b96ce521ae5477952d03634ef5dd30ed2d76357a9f00",
    labels=("positive", "negative", "neutral"),
)


class SentenceProbs(BaseModel):
    """Class probabilities for each sentence, in ``labels`` order, and truncation flags."""

    labels: tuple[str, ...]
    probs: list[list[float]]
    truncated: list[bool]


# A classifier maps sentences to their class probabilities.
Classifier = Callable[[Sequence[str]], SentenceProbs]


class EncoderScore(BaseModel):
    """One document scored by one encoder, with what is needed to audit and reproduce it."""

    doc_id: str
    text_sha256: str
    model_id: str
    revision: str
    weights_sha256: str
    splitter_version: str
    labels: tuple[str, ...]
    n_sentences: int
    n_truncated: int
    mean_probs: dict[str, float]  # label -> mean probability over sentences
    sentences: list[str]
    sentence_probs: list[list[float]]

    @property
    def cache_key(self) -> tuple[str, str, str, str]:
        return (self.text_sha256, self.model_id, self.revision, self.splitter_version)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def score_document(doc: Document, spec: EncoderSpec, classify: Classifier) -> EncoderScore:
    sentences = split_sentences(doc.text)
    if not sentences:
        raise ValueError(f"{doc.doc_id}: no sentence to score")
    out = classify(sentences)
    if out.labels != spec.labels:
        raise ValueError(f"{spec.model_id}: labels {out.labels}, expected {spec.labels}")
    if len(out.probs) != len(sentences):
        raise ValueError(f"{spec.model_id}: {len(out.probs)} outputs for {len(sentences)} inputs")
    n = len(sentences)
    mean = {label: sum(p[i] for p in out.probs) / n for i, label in enumerate(spec.labels)}
    return EncoderScore(
        doc_id=doc.doc_id,
        text_sha256=text_sha256(doc.text),
        model_id=spec.model_id,
        revision=spec.revision,
        weights_sha256=spec.weights_sha256,
        splitter_version=SPLITTER_VERSION,
        labels=spec.labels,
        n_sentences=n,
        n_truncated=sum(out.truncated),
        mean_probs=mean,
        sentences=sentences,
        sentence_probs=out.probs,
    )


def read_cache(path: Path) -> dict[tuple[str, str, str, str], EncoderScore]:
    if not path.exists():
        return {}
    scores = (
        EncoderScore.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return {s.cache_key: s for s in scores}


def score_documents(
    docs: Iterable[Document],
    spec: EncoderSpec,
    classify: Callable[[], Classifier],
    cache_path: Path,
) -> list[EncoderScore]:
    """Score every document, reading from and appending to the JSONL cache at ``cache_path``.

    ``classify`` is a factory, called only if some document is missing from the cache, so a
    fully cached run never loads the model.
    """
    cache = read_cache(cache_path)
    model: Classifier | None = None
    results = []
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        key = (text_sha256(doc.text), spec.model_id, spec.revision, SPLITTER_VERSION)
        cached = cache.get(key)
        if cached is None or cached.weights_sha256 != spec.weights_sha256:
            if model is None:
                model = classify()
            cached = score_document(doc, spec, model)
            with cache_path.open("a", encoding="utf-8") as fh:
                fh.write(cached.model_dump_json() + "\n")
            cache[key] = cached
        results.append(cached.model_copy(update={"doc_id": doc.doc_id}))
    return results


def load_hf_classifier(
    spec: EncoderSpec, cache_dir: Path, *, device: str | None = None, batch_size: int = 32
) -> Classifier:
    """Download the pinned model into ``cache_dir`` and return a classifier on ``device``.

    Inference runs in float32, in evaluation mode, with a fixed batch size. Imports torch and
    transformers lazily, so that the rest of the package does not need them.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.model_id, revision=spec.revision, cache_dir=cache_dir, use_safetensors=True
    )
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    labels = tuple(id2label[i] for i in range(len(id2label)))
    if labels != spec.labels:
        raise ValueError(f"{spec.model_id}: config labels {labels}, expected {spec.labels}")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    model.eval()

    def classify(sentences: Sequence[str]) -> SentenceProbs:
        probs: list[list[float]] = []
        truncated: list[bool] = []
        for i in range(0, len(sentences), batch_size):
            batch = list(sentences[i : i + batch_size])
            full = tokenizer(batch, truncation=False)["input_ids"]
            truncated.extend(len(ids) > spec.max_length for ids in full)
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=spec.max_length,
                return_tensors="pt",
            ).to(dev)
            with torch.no_grad():
                logits = model(**enc).logits.float()
            probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
        return SentenceProbs(labels=labels, probs=probs, truncated=truncated)

    return classify


def provenance(spec: EncoderSpec) -> dict[str, str]:
    """Library versions and pins to log next to a scoring run."""
    import torch
    import transformers

    return {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "weights_sha256": spec.weights_sha256,
        "splitter_version": SPLITTER_VERSION,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
