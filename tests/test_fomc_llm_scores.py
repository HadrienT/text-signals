from datetime import date

from text_signals.llm import Decoding, ExtractionRecord
from text_signals.sources.fomc.llm_scores import score_table

EMPTY = {"value": None, "evidence": None}


def _record(doc_id: str, guidance: int, inflation: int) -> ExtractionRecord:
    output = {
        "policy_action": {"value": "hold", "change_bp": 0, "evidence": "x"},
        "forward_guidance": {"value": guidance, "evidence": "x"},
        "inflation_assessment": {"value": inflation, "evidence": "x"},
        "labor_market_assessment": EMPTY,
        "growth_assessment": EMPTY,
        "balance_sheet": EMPTY,
        "risk_balance": EMPTY,
        "dissents": [],
    }
    return ExtractionRecord(
        doc_id=doc_id,
        text_sha256="s",
        model="phi-4",
        model_path="/m.gguf",
        prompt_version="p",
        schema_version="v",
        decoding=Decoding(),
        created_at="2026-11-06T00:00:00Z",
        latency_s=1.0,
        prompt_tokens=1,
        completion_tokens=1,
        raw_output="{}",
        output=output,
        error=None,
    )


def test_scores_are_in_release_order_with_changes_and_the_cutoff_flag() -> None:
    days = {"b": date(2024, 9, 18), "a": date(2024, 6, 12), "c": date(2024, 11, 7)}
    records = {"a": _record("a", -1, 1), "b": _record("b", 1, 1)}  # "c" has no record
    t = score_table("phi-4", records, days)
    assert list(t["doc_id"]) == ["a", "b", "c"]
    assert list(t["llm_tone"][:2]) == [0.0, 1.0]
    assert list(t["llm_guidance"][:2]) == [-1.0, 1.0]
    assert t["d_llm_tone"][1] == 1.0 and t["d_llm_tone"].isna()[2]
    assert list(t["llm_after_cutoff"]) == [False, True, True]
    assert list(t["llm_valid"]) == [True, True, False]


def test_rows_with_an_invalid_first_answer_are_flagged() -> None:
    days = {"a": date(2024, 6, 12)}
    t = score_table("phi-4", {"a": _record("a", 0, 0)}, days, {"a"})
    assert t["llm_values_from"][0] == "repaired (first invalid)"
