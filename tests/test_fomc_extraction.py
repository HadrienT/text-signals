import pytest
from pydantic import ValidationError

from text_signals.sources.fomc.extraction import (
    SCHEMA_VERSION,
    StatementExtraction,
    evidence_not_in_text,
    guidance_score,
    score_changes,
    tone_score,
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


def test_tone_score_is_the_mean_of_the_filled_stance_fields() -> None:
    ex = _extraction(
        forward_guidance={"value": 2, "evidence": "decided to maintain"},
        labor_market_assessment={"value": -1, "evidence": "decided to maintain"},
    )
    # inflation +1, guidance +2, labour -1; growth and balance sheet null.
    assert tone_score(ex) == pytest.approx(2 / 3)
    assert guidance_score(ex) == 2.0


def test_scores_are_missing_when_nothing_is_filled() -> None:
    ex = _extraction(inflation_assessment=EMPTY)
    assert tone_score(ex) is None
    assert guidance_score(ex) is None


def test_score_changes_skip_missing_scores() -> None:
    assert score_changes([0.5, None, 1.0, -0.5]) == [None, None, 0.5, -1.5]


def test_schema_v2_restricts_every_evidence_to_the_sentence_numbers() -> None:
    from text_signals.sources.fomc.extraction import schema_v2_for

    schema = schema_v2_for(3)
    found = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict) and "evidence" in props:
                found.append(props["evidence"])
            for v in node.values():
                walk(v)

    walk(schema)
    assert len(found) == 8  # seven fields and the dissent
    for ev in found:
        options = ev.get("anyOf", [ev])
        assert {"enum": [1, 2, 3]} in options or ev.get("enum") == [1, 2, 3]


def test_to_spans_turns_numbers_back_into_sentences() -> None:
    from text_signals.sources.fomc.extraction import (
        StatementExtractionV2,
        numbered,
        to_spans,
    )

    sentences = ["The Committee decided to hold.", "Inflation remains elevated."]
    assert (
        numbered(sentences) == "[1] The Committee decided to hold.\n[2] Inflation remains elevated."
    )
    empty = {"value": None, "evidence": None}
    v2 = StatementExtractionV2.model_validate(
        {
            "policy_action": {"value": "hold", "change_bp": 0, "evidence": 1},
            "forward_guidance": empty,
            "inflation_assessment": {"value": 1, "evidence": 2},
            "labor_market_assessment": empty,
            "growth_assessment": empty,
            "balance_sheet": empty,
            "risk_balance": empty,
            "dissents": [],
        }
    )
    v1 = to_spans(v2, sentences)
    assert v1.inflation_assessment.evidence == "Inflation remains elevated."
    assert v1.policy_action.evidence == "The Committee decided to hold."
    assert v1.inflation_assessment.value == 1
    with pytest.raises(IndexError):
        to_spans(v2, sentences[:1])


def test_closed_schema_allows_only_whole_sentences_of_the_text() -> None:
    from text_signals.sources.fomc.extraction import schema_v1_closed

    sentences = [
        "Inflation remains elevated.",
        "The Committee’s target is 2 percent.",
        "Inflation remains elevated.",
    ]
    schema = schema_v1_closed(sentences)
    enums = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict) and "evidence" in props:
                ev = props["evidence"]
                enums.append(next(o["enum"] for o in ev.get("anyOf", [ev]) if "enum" in o))
            for v in node.values():
                walk(v)

    walk(schema)
    assert len(enums) == 8
    assert all(e == sentences[:2] for e in enums)  # duplicates listed once


def test_prompt_v4_asks_for_one_whole_sentence() -> None:
    from text_signals.sources.fomc.prompts import STATEMENT_V2, STATEMENT_V4

    assert "one whole sentence" in STATEMENT_V4.system
    assert "part of one. Never paraphrase" not in STATEMENT_V4.system
    assert STATEMENT_V4.user_template == STATEMENT_V2.user_template
