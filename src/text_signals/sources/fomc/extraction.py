"""Structured extraction schema for FOMC statements (protocol §5.2), version 1.

The LLM fills one ``StatementExtraction`` per statement. Every stance field is an integer from
−2 to +2 with the same sign convention, **positive = hawkish** (pointing to higher policy rates),
negative = dovish, or ``None`` when the statement says nothing on it. Every filled field carries
``evidence``: a span copied verbatim from the statement, so that each value can be checked against
the text and a span that is not in the text counts as a hallucination.

The field descriptions are the definitions: they are sent to the model as part of the JSON
schema and they are the anchors of the hand-labelling guide. Changing one changes the measure,
so it requires a new ``SCHEMA_VERSION``.

``policy_action`` is a quality check only. It does not enter the tone score, because MP1 already
measures the surprise in the decision (protocol §6.2).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "fomc-statement-v1"

Stance = Literal[-2, -1, 0, 1, 2]

_EVIDENCE = (
    "A span copied word for word from the statement that supports the value: one sentence or "
    "part of one. Null only when the value is null."
)


class _Field(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence: str | None = Field(description=_EVIDENCE)

    @model_validator(mode="after")
    def _evidence_iff_value(self) -> _Field:
        value = getattr(self, "value", None)
        if (value is None) != (self.evidence is None):
            raise ValueError("evidence must be given exactly when a value is given")
        return self


class ForwardGuidance(_Field):
    value: Stance | None = Field(
        description=(
            "What the statement signals about the future path of the policy rate, beyond "
            "today's decision. -2: an explicit commitment to keep rates low or to ease further "
            "for an extended period or until stated conditions are met. -1: a lean towards "
            "easing or towards keeping an accommodative stance (patience). 0: guidance that "
            "leans neither way (data dependence only). +1: a lean towards further tightening "
            "(further firming or gradual increases may be needed). +2: a clear signal of "
            "ongoing or forceful tightening. Null: no forward-looking statement about rates."
        )
    )


class InflationAssessment(_Field):
    value: Stance | None = Field(
        description=(
            "How concerned the Committee is that inflation is too high, now or in its outlook. "
            "-2: concern that inflation is too low or that prices may fall. -1: inflation "
            "subdued or below the goal. 0: inflation near the goal, stable or contained. "
            "+1: inflation elevated, rising, or above the goal. +2: high inflation described as "
            "a primary concern or risk. Null: inflation not assessed."
        )
    )


class LaborMarketAssessment(_Field):
    value: Stance | None = Field(
        description=(
            "How strong the Committee judges the labour market. -2: deteriorating sharply or "
            "very weak. -1: soft, weakening, or with significant slack. 0: steady or mixed. "
            "+1: solid or improving. +2: very strong or tight. Null: not assessed."
        )
    )


class GrowthAssessment(_Field):
    value: Stance | None = Field(
        description=(
            "How strong the Committee judges economic activity. -2: contracting or "
            "deteriorating sharply. -1: slow, softening, or below trend. 0: moderate or mixed. "
            "+1: solid or picking up. +2: strong or above trend. Null: not assessed."
        )
    )


class BalanceSheet(_Field):
    value: Stance | None = Field(
        description=(
            "The stance of balance-sheet policy (asset purchases and runoff). -2: new or larger "
            "asset purchases. -1: continued purchases or reinvestment that maintain "
            "accommodation, or an end or slowdown of runoff. 0: a balance-sheet statement that "
            "leans neither way. +1: tapering of purchases, or ongoing runoff. +2: faster runoff "
            "or asset sales. Null: balance sheet not mentioned."
        )
    )


class RiskBalance(_Field):
    value: Literal["inflation", "balanced", "growth"] | None = Field(
        description=(
            "Where the statement places the main risk. 'inflation': risks weighted towards "
            "higher inflation. 'growth': risks weighted towards weaker activity or employment, "
            "or towards inflation too low. 'balanced': risks described as balanced. Null: no "
            "statement on the balance of risks."
        )
    )


class PolicyAction(_Field):
    value: Literal["raise", "cut", "hold", "no_rate_decision"] | None = Field(
        description=(
            "Today's decision on the policy rate (the federal funds target or target range). "
            "'no_rate_decision': the statement is about something else (swap lines, a "
            "framework, a facility). Null only if the text is unreadable."
        )
    )
    change_bp: int | None = Field(
        description=(
            "Size of the change in the target, in basis points, positive for a raise and "
            "negative for a cut; 0 for a hold; null when there is no rate decision."
        )
    )


class Dissent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    member: str = Field(description="Name of the member who voted against, as written.")
    preferred: Literal["tighter", "easier", "other"] = Field(
        description=(
            "What the member preferred: 'tighter' (a higher rate, less accommodation), "
            "'easier' (a lower rate, more accommodation), 'other' (a different reason, or "
            "not stated)."
        )
    )
    evidence: str = Field(description=_EVIDENCE)


class StatementExtraction(BaseModel):
    """Everything the model extracts from one FOMC statement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_action: PolicyAction
    forward_guidance: ForwardGuidance
    inflation_assessment: InflationAssessment
    labor_market_assessment: LaborMarketAssessment
    growth_assessment: GrowthAssessment
    balance_sheet: BalanceSheet
    risk_balance: RiskBalance
    dissents: list[Dissent] = Field(
        description="Every member who voted against the policy action; empty if none."
    )


def _normalise(text: str) -> str:
    """Lower case, one space between words, typographic quotes and dashes made plain."""
    table = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
    return re.sub(r"\s+", " ", text.translate(table)).strip().lower()


def evidence_spans(extraction: StatementExtraction) -> dict[str, str]:
    """Field name -> evidence span, for every filled field and every dissent."""
    spans: dict[str, str] = {}
    for name in StatementExtraction.model_fields:
        field = getattr(extraction, name)
        if isinstance(field, _Field) and field.evidence is not None:
            spans[name] = field.evidence
    for i, dissent in enumerate(extraction.dissents):
        spans[f"dissents[{i}]"] = dissent.evidence
    return spans


def evidence_not_in_text(extraction: StatementExtraction, text: str) -> list[str]:
    """Fields whose evidence does not appear verbatim in ``text`` (up to case, spacing, quotes).

    This is the hallucination check of protocol §6.6.
    """
    haystack = _normalise(text)
    return [
        name
        for name, span in evidence_spans(extraction).items()
        if _normalise(span) not in haystack
    ]


def evidence_problems(extraction: BaseModel, text: str) -> list[str]:
    """The verifier of the repair loop: one message per evidence span not found in ``text``.

    The messages are sent back to the model as they are, so they quote the span and say what is
    wrong in plain words.
    """
    if not isinstance(extraction, StatementExtraction):
        raise TypeError("expected a StatementExtraction")
    spans = evidence_spans(extraction)
    return [
        f'{name}: the evidence "{spans[name]}" does not appear word for word in the statement.'
        for name in evidence_not_in_text(extraction, text)
    ]


# The aggregate scores (protocol §5.2, fixed 2026-11-06 before any model output was seen).
STANCE_FIELDS = (
    "forward_guidance",
    "inflation_assessment",
    "labor_market_assessment",
    "growth_assessment",
    "balance_sheet",
)


def tone_score(extraction: StatementExtraction) -> float | None:
    """Primary LLM score: the equal-weight mean of the stance fields that are filled.

    From −2 (dovish) to +2 (hawkish); ``None`` when no stance field is filled. No weight is
    chosen, so nothing is fitted to the outcome.
    """
    values = [getattr(extraction, f).value for f in STANCE_FIELDS]
    filled = [v for v in values if v is not None]
    return sum(filled) / len(filled) if filled else None


def guidance_score(extraction: StatementExtraction) -> float | None:
    """Secondary LLM score: ``forward_guidance`` alone, ``None`` when the statement has none."""
    value = extraction.forward_guidance.value
    return None if value is None else float(value)


def score_changes(scores: Sequence[float | None]) -> list[float | None]:
    """H2: each score minus the most recent earlier score that is not missing.

    ``scores`` are in release order. The first score, and every missing score, has no change.
    """
    out: list[float | None] = []
    previous: float | None = None
    for score in scores:
        out.append(None if score is None or previous is None else score - previous)
        if score is not None:
            previous = score
    return out


# ---------------------------------------------------------------------------------------------
# Schema v2 (2026-11-06): evidence by sentence number instead of by copy (protocol §6.6).
#
# The statement is shown to the model as numbered sentences (``numbered``), and every evidence
# is the number of one sentence. The values and their definitions are those of v1: only the way
# of citing changes. A request sends ``schema_v2_for(n)``, in which each evidence may only be
# one of 1..n, so the grammar of the server makes a citation that is not in the text impossible.
# ``to_spans`` turns a v2 answer back into a v1 one, with each number replaced by its sentence,
# so that every v1 metric applies unchanged.
# ---------------------------------------------------------------------------------------------

SCHEMA_VERSION_V2 = "fomc-statement-v2"

_EVIDENCE_V2 = (
    "The number of the sentence of the statement that supports the value. Null only when the "
    "value is null."
)


class ForwardGuidanceN(ForwardGuidance):
    evidence: int | None = Field(description=_EVIDENCE_V2)  # type: ignore[assignment]


class InflationAssessmentN(InflationAssessment):
    evidence: int | None = Field(description=_EVIDENCE_V2)  # type: ignore[assignment]


class LaborMarketAssessmentN(LaborMarketAssessment):
    evidence: int | None = Field(description=_EVIDENCE_V2)  # type: ignore[assignment]


class GrowthAssessmentN(GrowthAssessment):
    evidence: int | None = Field(description=_EVIDENCE_V2)  # type: ignore[assignment]


class BalanceSheetN(BalanceSheet):
    evidence: int | None = Field(description=_EVIDENCE_V2)  # type: ignore[assignment]


class RiskBalanceN(RiskBalance):
    evidence: int | None = Field(description=_EVIDENCE_V2)  # type: ignore[assignment]


class PolicyActionN(PolicyAction):
    evidence: int | None = Field(description=_EVIDENCE_V2)  # type: ignore[assignment]


class DissentN(Dissent):
    evidence: int = Field(description=_EVIDENCE_V2)  # type: ignore[assignment]


class StatementExtractionV2(BaseModel):
    """Schema v2: the fields of v1, each evidence a sentence number."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_action: PolicyActionN
    forward_guidance: ForwardGuidanceN
    inflation_assessment: InflationAssessmentN
    labor_market_assessment: LaborMarketAssessmentN
    growth_assessment: GrowthAssessmentN
    balance_sheet: BalanceSheetN
    risk_balance: RiskBalanceN
    dissents: list[DissentN] = Field(
        description="Every member who voted against the policy action; empty if none."
    )


def numbered(sentences: Sequence[str]) -> str:
    """The statement as the model sees it in v2: one ``[k] sentence`` per line, from 1."""
    return "\n".join(f"[{k}] {s}" for k, s in enumerate(sentences, 1))


def schema_v2_for(n_sentences: int) -> dict[str, object]:
    """The v2 JSON schema with every evidence restricted to the numbers 1..``n_sentences``."""
    schema = StatementExtractionV2.model_json_schema()
    allowed = list(range(1, n_sentences + 1))

    def restrict(node: object) -> None:
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict) and "evidence" in props:
                ev = props["evidence"]
                keep = {k: v for k, v in ev.items() if k in ("description", "title")}
                nullable = "anyOf" in ev
                props["evidence"] = keep | (
                    {"anyOf": [{"enum": allowed}, {"type": "null"}]}
                    if nullable
                    else {"enum": allowed}
                )
            for value in node.values():
                restrict(value)
        elif isinstance(node, list):
            for value in node:
                restrict(value)

    restrict(schema)
    return schema


def to_spans(extraction: StatementExtractionV2, sentences: Sequence[str]) -> StatementExtraction:
    """The v1 form of a v2 answer: each sentence number replaced by its sentence.

    Raises ``IndexError`` on a number outside 1..len(sentences), which the restricted schema
    rules out.
    """

    def span(k: int | None) -> str | None:
        if k is None:
            return None
        if not 1 <= k <= len(sentences):
            raise IndexError(f"sentence {k} of {len(sentences)}")
        return sentences[k - 1]

    data = extraction.model_dump(mode="json")
    for name, value in data.items():
        if name == "dissents":
            for d in value:
                d["evidence"] = span(d["evidence"])
        else:
            value["evidence"] = span(value["evidence"])
    return StatementExtraction.model_validate(data)
