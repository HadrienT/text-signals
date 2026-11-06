import json
from pathlib import Path

import httpx

from text_signals.llm import Decoding, ExtractionRecord, LLMClient, extract_all, read_cache
from text_signals.sources.fomc.extraction import SCHEMA_VERSION, StatementExtraction
from text_signals.sources.fomc.prompts import STATEMENT_V1

EMPTY = {"value": None, "evidence": None}
VALID = {
    "policy_action": {"value": "hold", "change_bp": 0, "evidence": "decided to maintain"},
    "forward_guidance": EMPTY,
    "inflation_assessment": {"value": 1, "evidence": "Inflation remains elevated."},
    "labor_market_assessment": EMPTY,
    "growth_assessment": EMPTY,
    "balance_sheet": EMPTY,
    "risk_balance": EMPTY,
    "dissents": [],
}


def _server(content: str, seen: list[dict[str, object]]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "test-model"}]})
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": "/opt/llm/models/test.gguf"})
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 300},
            },
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def _extract(client: LLMClient, text: str = "Inflation remains elevated.") -> ExtractionRecord:
    name, path = client.model_info()
    return client.extract(
        doc_id="d1",
        text=text,
        prompt=STATEMENT_V1,
        schema=StatementExtraction,
        schema_version=SCHEMA_VERSION,
        decoding=Decoding(),
        model=name,
        model_path=path,
    )


def test_request_is_deterministic_and_schema_constrained() -> None:
    seen: list[dict[str, object]] = []
    record = _extract(LLMClient("http://llm", client=_server(json.dumps(VALID), seen)))
    [body] = seen
    assert body["temperature"] == 0.0 and body["seed"] == 0
    fmt = body["response_format"]
    assert isinstance(fmt, dict) and fmt["type"] == "json_schema"
    assert record.output is not None and record.error is None
    assert record.output["inflation_assessment"]["value"] == 1
    assert (record.model, record.model_path) == ("test-model", "/opt/llm/models/test.gguf")
    assert (record.prompt_tokens, record.completion_tokens) == (900, 300)
    assert record.prompt_version == "fomc-statement-prompt-v1"


def test_invalid_output_is_kept_with_its_error() -> None:
    bad = dict(VALID, inflation_assessment={"value": 1, "evidence": None})
    record = _extract(LLMClient("http://llm", client=_server(json.dumps(bad), [])))
    assert record.output is None
    assert record.error is not None and "evidence" in record.error
    assert record.raw_output == json.dumps(bad)


def test_cache_resumes_without_new_requests(tmp_path: Path) -> None:
    seen: list[dict[str, object]] = []
    client = LLMClient("http://llm", client=_server(json.dumps(VALID), seen))
    cache = tmp_path / "x.jsonl"
    docs = [("d1", "Text one."), ("d2", "Text two.")]

    def run(doc_id: str, text: str) -> ExtractionRecord:
        return _extract(client, text).model_copy(update={"doc_id": doc_id})

    def extract() -> list[ExtractionRecord]:
        return extract_all(
            docs,
            run,
            model="test-model",
            prompt_version=STATEMENT_V1.version,
            schema_version=SCHEMA_VERSION,
            cache_path=cache,
        )

    extract()
    assert len(seen) == 2
    again = extract()
    assert len(seen) == 2  # served from the cache
    assert [r.doc_id for r in again] == ["d1", "d2"]
    assert len(read_cache(cache)) == 2


def test_prompt_shows_the_text_only() -> None:
    messages = STATEMENT_V1.messages("Inflation remains elevated.")
    assert messages[1]["content"] == "Statement:\n\nInflation remains elevated."
    assert "Do not use what you may know" in messages[0]["content"]


def test_prompt_v2_carries_the_field_definitions() -> None:
    from text_signals.sources.fomc.prompts import STATEMENT_V2

    system = STATEMENT_V2.messages("x")[0]["content"]
    assert system.startswith(STATEMENT_V1.system)
    assert "further firming or gradual increases may be needed" in system
