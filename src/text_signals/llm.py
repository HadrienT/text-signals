"""Structured extraction with a local LLM, with provenance and a resumable cache.

The model is served by ``llama-server`` with its OpenAI-compatible API. Each request asks for a
JSON object constrained by the schema of a Pydantic model (``response_format`` of type
``json_schema``), with deterministic decoding (temperature 0, fixed seed).

Every result is an ``ExtractionRecord``: the raw output as returned, the parsed object (or the
validation error), the model as reported by the server, the prompt and schema versions, the
decoding parameters, the token counts and the latency. Records are appended to a JSONL cache
keyed by (text sha256, model, prompt version, schema version), so that a long run can stop and
resume, and a rerun makes no request.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError


class Decoding(BaseModel):
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 2048


class Prompt(BaseModel):
    """A versioned prompt: a system message and a user template with ``{text}``."""

    version: str
    system: str
    user_template: str

    def messages(self, text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user_template.format(text=text)},
        ]


class ExtractionRecord(BaseModel):
    doc_id: str
    text_sha256: str
    model: str  # as reported by the server
    model_path: str | None  # the weights file, from the server's /props
    prompt_version: str
    schema_version: str
    decoding: Decoding
    created_at: datetime
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    raw_output: str
    output: dict[str, Any] | None  # the validated object, as JSON
    error: str | None  # why the raw output was rejected

    @property
    def cache_key(self) -> tuple[str, str, str, str]:
        return (self.text_sha256, self.model, self.prompt_version, self.schema_version)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class LLMClient:
    """A thin client for one model served by an OpenAI-compatible ``llama-server``."""

    def __init__(self, base_url: str, *, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=600.0)

    def model_info(self) -> tuple[str, str | None]:
        """The served model name and, when the server exposes it, the path of its weights."""
        models = self._client.get(f"{self.base_url}/v1/models").raise_for_status().json()
        name = str(models["data"][0]["id"])
        try:
            props = self._client.get(f"{self.base_url}/props").raise_for_status().json()
            path = props.get("model_path")
        except httpx.HTTPError:
            path = None
        return name, None if path is None else str(path)

    def extract(
        self,
        *,
        doc_id: str,
        text: str,
        prompt: Prompt,
        schema: type[BaseModel],
        schema_version: str,
        decoding: Decoding,
        model: str,
        model_path: str | None,
    ) -> ExtractionRecord:
        body = {
            "model": model,
            "messages": prompt.messages(text),
            "temperature": decoding.temperature,
            "seed": decoding.seed,
            "max_tokens": decoding.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            },
        }
        start = time.monotonic()
        response = self._client.post(f"{self.base_url}/v1/chat/completions", json=body)
        response.raise_for_status()
        latency = time.monotonic() - start
        payload = response.json()
        raw = str(payload["choices"][0]["message"]["content"])
        usage = payload.get("usage") or {}
        output: dict[str, Any] | None = None
        error: str | None = None
        try:
            output = schema.model_validate_json(raw).model_dump(mode="json")
        except ValidationError as exc:
            error = str(exc)
        return ExtractionRecord(
            doc_id=doc_id,
            text_sha256=text_sha256(text),
            model=str(payload.get("model") or model),
            model_path=model_path,
            prompt_version=prompt.version,
            schema_version=schema_version,
            decoding=decoding,
            created_at=datetime.now(UTC),
            latency_s=latency,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw_output=raw,
            output=output,
            error=error,
        )

    def close(self) -> None:
        self._client.close()


def read_cache(path: Path) -> dict[tuple[str, str, str, str], ExtractionRecord]:
    if not path.exists():
        return {}
    records = (
        ExtractionRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return {r.cache_key: r for r in records}


def extract_all(
    docs: Iterable[tuple[str, str]],
    run: Callable[[str, str], ExtractionRecord],
    *,
    model: str,
    prompt_version: str,
    schema_version: str,
    cache_path: Path,
    on_record: Callable[[ExtractionRecord, bool], None] | None = None,
) -> list[ExtractionRecord]:
    """Extract every ``(doc_id, text)``, appending each new record to the cache as it comes.

    ``run(doc_id, text)`` makes one request. A record that failed validation is cached too (it
    is a result: the model's failure rate is reported); delete it from the cache to retry.
    ``on_record(record, from_cache)`` is called after each document, e.g. to log progress.
    """
    cache = read_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for doc_id, text in docs:
        key = (text_sha256(text), model, prompt_version, schema_version)
        record = cache.get(key)
        from_cache = record is not None
        if record is None:
            record = run(doc_id, text)
            with cache_path.open("a", encoding="utf-8") as fh:
                fh.write(record.model_dump_json() + "\n")
            cache[key] = record
        if on_record is not None:
            on_record(record, from_cache)
        out.append(record.model_copy(update={"doc_id": doc_id}))
    return out
