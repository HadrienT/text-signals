"""The second reader in batches (protocol §6.9, batch trial).

Several statements are read in one ``claude -p`` call, to lower the quota used per statement.
To limit recognition, each batch takes the **anonymised** texts (§6.7: dates, years and member
names removed), in a random order, each under a neutral code (S01, S02, …), and the model is told
to read each statement on its own. The readings are validated one by one and stored as one
record per statement, with the batch, the position and the batch's usage in ``meta``.

Usage: ``python -m text_signals.sources.fomc.claude_batch trial``.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from text_signals.claude_code import parse, run_raw
from text_signals.llm import ExtractionRecord, text_sha256
from text_signals.sources.fomc.claude_reader import MODEL, WORKDIR, trial_docs
from text_signals.sources.fomc.extract import PROMPT, SampleDoc, all_docs
from text_signals.sources.fomc.extraction import SCHEMA_VERSION, StatementExtraction
from text_signals.sources.fomc.lookahead import anonymise, member_names

BATCH_VERSION = "fomc-batch-v1"
BATCH_SEED = 20261111
BATCH_SIZE = 30
CACHE = Path("data/interim/fomc/extractions/claude-code") / f"{MODEL}-batch.jsonl"

BATCH_RULES = (
    "\n\nYou will receive several statements in one message, each introduced by a code in "
    "square brackets, such as [S07]. Dates and names have been replaced by placeholders such as "
    "[date] and [member]. Read each statement on its own, as if it were the only one: do not "
    "compare statements, rank them against each other, or let one reading influence another. "
    "Evidence spans are copied from the statement they belong to. Return exactly one reading "
    'per statement, in the "readings" list, each with its code.'
)


def batch_schema(codes: list[str]) -> dict[str, Any]:
    """A list of readings, each the statement schema plus its code."""
    base = copy.deepcopy(StatementExtraction.model_json_schema())
    defs = base.pop("$defs", {})
    base["properties"] = {"code": {"type": "string", "enum": codes}} | base["properties"]
    base["required"] = ["code", *base["required"]]
    return {
        "type": "object",
        "properties": {
            "readings": {
                "type": "array",
                "items": base,
                "minItems": len(codes),
                "maxItems": len(codes),
            }
        },
        "required": ["readings"],
        "additionalProperties": False,
        "$defs": defs,
    }


def draw_batch(
    docs: list[SampleDoc], exclude: set[str], n: int = BATCH_SIZE, seed: int = BATCH_SEED
) -> list[SampleDoc]:
    """``n`` statements drawn at random, in random order (the order they are shown in)."""
    rng = np.random.default_rng(seed)
    pool = sorted((d for d in docs if d.doc_id not in exclude), key=lambda d: d.doc_id)
    return [pool[int(i)] for i in rng.choice(len(pool), size=n, replace=False)]


def render(texts: list[str]) -> tuple[list[str], str]:
    codes = [f"S{i:02d}" for i in range(1, len(texts) + 1)]
    blocks = [f"[{c}]\n{t}" for c, t in zip(codes, texts, strict=True)]
    return codes, "Statements:\n\n" + "\n\n".join(blocks)


def read_batch(docs: list[SampleDoc], batch_id: str) -> list[ExtractionRecord]:
    names = member_names(d.text for d in all_docs())
    texts = [anonymise(d.text, names) for d in docs]
    codes, user = render(texts)
    payload, rc = run_raw(
        PROMPT.system + BATCH_RULES,
        user,
        batch_schema(codes),
        workdir=WORKDIR,
        model=MODEL,
        effort="high",
        timeout_s=3600,
    )
    readings: dict[str, Any] = {}
    batch_error = None
    out = payload.get("structured_output")
    if payload.get("is_error") or not isinstance(out, dict):
        batch_error = f"batch failed: {payload.get('subtype')} {payload.get('api_error_status')}"
    else:
        readings = {r.get("code"): r for r in out.get("readings", [])}
    meta_batch = {
        "backend": "claude-code",
        "batch_version": BATCH_VERSION,
        "batch_id": batch_id,
        "batch_size": len(docs),
        "effort": "high",
        "model_usage": payload.get("modelUsage"),
        "cost_usd_equivalent_batch": payload.get("total_cost_usd"),
        "duration_s_batch": (payload.get("duration_ms") or 0) / 1000,
        "returncode": rc,
        "anonymised": True,
    }
    records = []
    for pos, (code, d, text) in enumerate(zip(codes, docs, texts, strict=True), start=1):
        raw = readings.get(code)
        output: dict[str, Any] | None = None
        error: str | None = batch_error or f"no reading for {code}"
        if raw is not None:
            reading = {k: v for k, v in raw.items() if k != "code"}
            output, error = parse({"structured_output": reading}, StatementExtraction)
        records.append(
            ExtractionRecord(
                doc_id=d.doc_id,
                text_sha256=text_sha256(text),
                model=MODEL,
                model_path=None,
                prompt_version=f"{PROMPT.version}+{BATCH_VERSION}",
                schema_version=SCHEMA_VERSION,
                decoding=None,
                created_at=datetime.now(UTC),
                latency_s=None,
                prompt_tokens=None,
                completion_tokens=None,
                raw_output=json.dumps(raw),
                output=output,
                error=error,
                procedure=f"batch-{len(docs)}",
                meta=meta_batch | {"code": code, "position": pos},
            )
        )
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=["trial"])
    parser.parse_args()
    docs = all_docs()
    exclude = {d.doc_id for d in trial_docs(docs)}
    batch = draw_batch(docs, exclude)
    records = read_batch(batch, "trial-1")
    meta = records[0].meta or {}
    cost = meta.get("cost_usd_equivalent_batch") or 0
    valid = sum(r.output is not None for r in records)
    print(
        f"{len(records)} statements, {valid} valid, {meta.get('duration_s_batch', 0):.0f}s, "
        f"API-equivalent ${cost:.2f} (${cost / len(records):.3f} per statement)"
    )
    print(
        {
            m: {
                k: u.get(k)
                for k in (
                    "inputTokens",
                    "outputTokens",
                    "cacheReadInputTokens",
                    "cacheCreationInputTokens",
                    "costUSD",
                )
            }
            for m, u in (meta.get("model_usage") or {}).items()
        }
    )


if __name__ == "__main__":
    main()
