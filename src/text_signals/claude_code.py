"""Claude Code in non-interactive mode as an extraction backend (protocol §6.9).

Each call runs ``claude -p`` in a fresh process:
- no tools, no MCP servers, no session persistence;
- from an empty working directory, so no project ``CLAUDE.md`` or memory is loaded;
- with the prompt's system message in place of Claude Code's own;
- with the schema passed as ``--json-schema``.

``ANTHROPIC_API_KEY`` and the parent session's ``CLAUDE_CODE_*`` variables are removed from the
environment, so the call runs on the logged-in subscription and is not attached to this session.

Temperature and seed cannot be set: the record's decoding is ``None``. The models used, their
token counts and the API-equivalent cost reported by the tool go to ``record.meta``.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from text_signals.llm import ExtractionRecord, Prompt, text_sha256


def clean_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """The environment without an API key and without the parent session's variables."""
    base = dict(os.environ if env is None else env)
    return {
        k: v
        for k, v in base.items()
        if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
        and not k.startswith("CLAUDE_CODE_")
    }


def command(model: str, effort: str, system: str, schema: dict[str, Any]) -> list[str]:
    return [
        "claude",
        "-p",
        "--model",
        model,
        "--effort",
        effort,
        "--output-format",
        "json",
        "--tools",
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--system-prompt",
        system,
        "--json-schema",
        json.dumps(schema),
    ]


def parse(
    payload: dict[str, Any], schema: type[BaseModel]
) -> tuple[dict[str, Any] | None, str | None]:
    """The validated output, or the reason it was rejected."""
    if payload.get("is_error"):
        return None, f"claude -p error: {payload.get('subtype')} {payload.get('api_error_status')}"
    out = payload.get("structured_output")
    if out is None:
        return None, "no structured output"
    try:
        return schema.model_validate(out).model_dump(mode="json"), None
    except ValidationError as e:
        return None, str(e)


class ClaudeCodeReader:
    """Reads one document per ``claude -p`` process."""

    def __init__(
        self,
        workdir: Path,
        *,
        model: str = "claude-opus-5-5",
        effort: str = "high",
        timeout_s: float = 600,
    ) -> None:
        self.workdir = workdir
        self.model = model
        self.effort = effort
        self.timeout_s = timeout_s

    def extract(
        self,
        *,
        doc_id: str,
        text: str,
        prompt: Prompt,
        schema: type[BaseModel],
        schema_version: str,
    ) -> ExtractionRecord:
        self.workdir.mkdir(parents=True, exist_ok=True)
        user = prompt.user_template.format(text=text)
        proc = subprocess.run(
            command(self.model, self.effort, prompt.system, schema.model_json_schema()),
            input=user,
            capture_output=True,
            text=True,
            cwd=self.workdir,
            env=clean_env(),
            timeout=self.timeout_s,
            check=False,
        )
        try:
            payload: dict[str, Any] = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = {"is_error": True, "subtype": "unparsable", "stderr": proc.stderr[-2000:]}
        output, error = parse(payload, schema)
        usage = payload.get("modelUsage") or {}
        main = usage.get(self.model, {})
        return ExtractionRecord(
            doc_id=doc_id,
            text_sha256=text_sha256(text),
            model=self.model,
            model_path=None,
            prompt_version=prompt.version,
            schema_version=schema_version,
            decoding=None,
            created_at=datetime.now(UTC),
            latency_s=(payload.get("duration_ms") or 0) / 1000 or None,
            prompt_tokens=(
                main.get("inputTokens", 0)
                + main.get("cacheReadInputTokens", 0)
                + main.get("cacheCreationInputTokens", 0)
            )
            or None,
            completion_tokens=main.get("outputTokens"),
            raw_output=json.dumps(payload.get("structured_output") or payload.get("result")),
            output=output,
            error=error,
            meta={
                "backend": "claude-code",
                "effort": self.effort,
                "model_usage": usage,
                "cost_usd_equivalent": payload.get("total_cost_usd"),
                "num_turns": payload.get("num_turns"),
                "returncode": proc.returncode,
            },
        )
