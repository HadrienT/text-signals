import pytest
from pydantic import ValidationError

from text_signals.sources.fomc.extraction import (
    SCHEMA_VERSION,
    StatementExtraction,
    evidence_not_in_text,
)

# Part of the statement of 28 October 2026.
TEXT = (
    "The Committee decided to maintain the target range for the federal funds rate at 3-3/4 to "
    "4 percent, in support of the Federal Reserve's dual mandate.\n\nInflation remains "
    "elevated.\n\nVoting against the monetary policy action were Lisa D. Cook and Beth M. "
    "Hammack, who preferred to raise the target range for the federal funds rate by 1/4 "
    "percentage point at this meeting."
)

EMPTY = {"value": None, "evidence": None}


def _extraction(**fields: object) -> StatementExtraction:
    base: dict[str, object] = {
        "policy_action": {"value": "hold", "change_bp": 0, "evidence": "decided to maintain"},
        "forward_guidance": EMPTY,
        "inflation_assessment": {"value": 1, "evidence": "Inflation remains elevated."},
        "labor_market_assessment": EMPTY,
        "growth_assessment": EMPTY,
        "balance_sheet": EMPTY,
        "risk_balance": EMPTY,
        "dissents": [],
    }
    base.update(fields)
    return StatementExtraction.model_validate(base)


def test_a_valid_extraction_has_no_hallucinated_evidence() -> None:
    dissent = {
        "member": "Lisa D. Cook",
        "preferred": "tighter",
        "evidence": "who preferred to raise the target range",
    }
    ex = _extraction(dissents=[dissent])
    assert evidence_not_in_text(ex, TEXT) == []
    assert SCHEMA_VERSION == "fomc-statement-v1"


def test_evidence_not_in_the_text_is_flagged() -> None:
    ex = _extraction(inflation_assessment={"value": 2, "evidence": "Inflation is far too high."})
    assert evidence_not_in_text(ex, TEXT) == ["inflation_assessment"]


def test_evidence_match_ignores_case_spacing_and_typographic_quotes() -> None:
    ex = _extraction(
        policy_action={
            "value": "hold",
            "change_bp": 0,
            "evidence": "in support of the Federal Reserve’s   dual mandate",
        }
    )
    assert evidence_not_in_text(ex, TEXT) == []


@pytest.mark.parametrize(
    "field",
    [
        {"value": 1, "evidence": None},  # a value without evidence
        {"value": None, "evidence": "Inflation remains elevated."},  # evidence without a value
        {"value": 3, "evidence": "Inflation remains elevated."},  # outside -2..+2
    ],
)
def test_invalid_stance_fields_are_rejected(field: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _extraction(inflation_assessment=field)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _extraction(tone={"value": 1, "evidence": "x"})


def test_json_schema_carries_the_definitions() -> None:
    schema = StatementExtraction.model_json_schema()
    guidance = schema["$defs"]["ForwardGuidance"]["properties"]["value"]
    assert "-2: an explicit commitment to keep rates low" in guidance["description"]
