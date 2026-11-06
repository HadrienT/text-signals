"""Structured extraction with a local LLM, with provenance and a resumable cache.

The model is served by ``llama-server`` with its OpenAI-compatible API. Each request asks for a
JSON object constrained by the schema of a Pydantic model (``response_format`` of type
``json_schema``), with deterministic decoding (temperature 0, fixed seed).

Every result is an ``ExtractionRecord``: the raw output as returned, the parsed object (or the
validation error), the model as reported by the server, the prompt and schema versions, the
decoding parameters, the token counts and the latency. Records are appended to a JSONL cache
keyed by (text sha256, model, prompt version, schema version, procedure), so that a long run can
stop and resume, and a rerun makes no request.

**Repair loop** (``extract_with_repair``). One request is a *single pass*: the model answers once
and the answer is kept as it is. With a repair loop, a *verifier* (plain code, no model) checks
the answer; if it finds problems, they are sent back to the model as a new user turn in the same
conversation (the model keeps no memory, so the whole history is resent), and the model answers
again. This is the basic shape of an LLM environment: act, verify, feed the verdict back. Every
attempt is stored, so the first answer and the repaired one can both be measured, together with
what the repair cost in latency and tokens.
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


class Attempt(BaseModel):
    """One answer of the model that the verifier rejected, and what it was told."""

    raw_output: str
    problems: list[str]  # what the verifier found, as sent back to the model
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None


class ExtractionRecord(BaseModel):
    doc_id: str
    text_sha256: str
    model: str  # as reported by the server
    model_path: str | None  # the weights file, from the server's /props
    prompt_version: str
    schema_version: str
    decoding: Decoding | None  # None when the decoding cannot be set (a hosted assistant)
    created_at: datetime
    latency_s: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    raw_output: str
    output: dict[str, Any] | None  # the validated object, as JSON
    error: str | None  # why the raw output was rejected
    # "single-pass", or "repair-<n>[:<repair prompt version>]" for at most n repair turns
    procedure: str = "single-pass"
    # earlier answers rejected by the verifier, in order; the fields above are the last answer
    rejected: list[Attempt] = []
    problems_left: list[str] = []  # what the verifier still found in the last answer

    @property
    def cache_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.text_sha256,
            self.model,
            self.prompt_version,
            self.schema_version,
            self.procedure,
        )


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RepairPrompt(BaseModel):
    """A versioned message that sends the verifier's problems back to the model."""

    version: str
    template: str  # with ``{problems}``, one problem per line

    def message(self, problems: list[str]) -> str:
        return self.template.format(problems="\n".join(f"- {p}" for p in problems))


def repair_procedure(max_repairs: int, repair: RepairPrompt) -> str:
    return f"repair-{max_repairs}:{repair.version}"


def _parse(raw: str, schema: type[BaseModel]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return schema.model_validate_json(raw).model_dump(mode="json"), None
    except ValidationError as exc:
        return None, str(exc)


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

    def _request(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        decoding: Decoding,
        model: str,
        json_schema: dict[str, Any] | None = None,
    ) -> tuple[str, float, dict[str, Any], str]:
        """One chat completion constrained by ``schema``: (raw text, latency, usage, model).

        ``json_schema`` replaces the schema sent to the server, e.g. a stricter version of
        ``schema`` for one document; the answer is still validated against ``schema``.
        """
        body = {
            "model": model,
            "messages": messages,
            "temperature": decoding.temperature,
            "seed": decoding.seed,
            "max_tokens": decoding.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": json_schema or schema.model_json_schema(),
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
        return raw, latency, payload.get("usage") or {}, str(payload.get("model") or model)

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
        json_schema: dict[str, Any] | None = None,
    ) -> ExtractionRecord:
        """A single pass: one request, the answer kept as it is."""
        raw, latency, usage, served = self._request(
            prompt.messages(text), schema, decoding, model, json_schema
        )
        output, error = _parse(raw, schema)
        return ExtractionRecord(
            doc_id=doc_id,
            text_sha256=text_sha256(text),
            model=served,
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

    def extract_with_repair(
        self,
        *,
        doc_id: str,
        text: str,
        prompt: Prompt,
        repair: RepairPrompt,
        verify: Callable[[BaseModel, str], list[str]],
        max_repairs: int,
        schema: type[BaseModel],
        schema_version: str,
        decoding: Decoding,
        model: str,
        model_path: str | None,
    ) -> ExtractionRecord:
        """Answer, verify, and send the problems back, at most ``max_repairs`` times.

        ``verify(parsed, text)`` returns the problems of a valid answer (empty when it is
        accepted). An answer that fails schema validation counts as a problem too. The record
        holds the last answer; earlier ones are in ``rejected``. Latency and tokens are summed
        over all turns, since that is what the answer cost.
        """
        messages = prompt.messages(text)
        rejected: list[Attempt] = []
        total_latency, total_in, total_out = 0.0, 0, 0
        while True:
            raw, latency, usage, served = self._request(messages, schema, decoding, model)
            total_latency += latency
            total_in += int(usage.get("prompt_tokens") or 0)
            total_out += int(usage.get("completion_tokens") or 0)
            output, error = _parse(raw, schema)
            if error is not None:
                problems = [f"The answer does not follow the schema: {error}"]
            else:
                problems = verify(schema.model_validate(output), text)
            if not problems or len(rejected) == max_repairs:
                break
            rejected.append(
                Attempt(
                    raw_output=raw,
                    problems=problems,
                    latency_s=latency,
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                )
            )
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": repair.message(problems)},
            ]
        return ExtractionRecord(
            doc_id=doc_id,
            text_sha256=text_sha256(text),
            model=served,
            model_path=model_path,
            prompt_version=prompt.version,
            schema_version=schema_version,
            decoding=decoding,
            created_at=datetime.now(UTC),
            latency_s=total_latency,
            prompt_tokens=total_in,
            completion_tokens=total_out,
            raw_output=raw,
            output=output,
            error=error,
            procedure=repair_procedure(max_repairs, repair),
            rejected=rejected,
            problems_left=problems,
        )

    def close(self) -> None:
        self._client.close()


def read_cache(path: Path) -> dict[tuple[str, str, str, str, str], ExtractionRecord]:
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
    procedure: str = "single-pass",
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
        key = (text_sha256(text), model, prompt_version, schema_version, procedure)
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
