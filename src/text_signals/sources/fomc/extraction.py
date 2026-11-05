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
