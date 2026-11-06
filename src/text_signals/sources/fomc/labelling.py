"""The hand-labelled evaluation set: which statements, in which order (protocol §6.6).

The sample is drawn once, from a fixed seed, before any model output exists. It is stratified by
decade and by policy regime (zero lower bound or not): each cell gets a share of the 60
statements proportional to its size, with at least ``MIN_PER_CELL`` (largest-remainder
rounding). Five practice statements, outside the sample, are for training the labeller; 15 of the
60 are labelled a second time, later, to measure the labeller's own consistency.

Statements are shown under neutral codes in a random order, so that the order says nothing about
the period.
"""

from __future__ import annotations

import math
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from text_signals.sources.fomc.extraction import SCHEMA_VERSION, Stance

SAMPLE_VERSION = "fomc-label-sample-v1"
SEED = 20261106
N_LABEL = 60
N_PRACTICE = 5
N_SECOND_PASS = 15
MIN_PER_CELL = 4


def allocate(sizes: dict[str, int], total: int, minimum: int) -> dict[str, int]:
    """Proportional allocation with a floor per cell and largest-remainder rounding."""
    floor = {k: min(minimum, n) for k, n in sizes.items()}
    rest = total - sum(floor.values())
    if rest < 0:
        raise ValueError("the floors exceed the total")
    spare = {k: n - floor[k] for k, n in sizes.items()}
    weight = sum(spare.values())
    shares = {k: rest * s / weight for k, s in spare.items()}
    out = {k: floor[k] + math.floor(v) for k, v in shares.items()}
    order = sorted(shares, key=lambda k: shares[k] - math.floor(shares[k]), reverse=True)
    for k in order[: total - sum(out.values())]:
        out[k] += 1
    return out


def stratum(release_date: str, zlb: bool) -> str:
    return f"{release_date[:3]}0s-{'zlb' if zlb else 'normal'}"


def draw_sample(events: pd.DataFrame) -> pd.DataFrame:
    """The labelling plan: one row per statement to label or practise on.

    ``events`` is the statement event table; only the main sample is drawn from. Columns of the
    result: ``code``, ``doc_id``, ``release_date``, ``stratum``, ``role`` ("label" or
    "practice"), ``second_pass``.
    """
    pool = events[events["in_main_sample"]].copy()
    pool["stratum"] = [
        stratum(str(d), bool(z)) for d, z in zip(pool["release_date"], pool["zlb"], strict=True)
    ]
    pool = pool.sort_values("doc_id").reset_index(drop=True)
    rng = random.Random(SEED)
    sizes = pool.groupby("stratum").size().to_dict()
    quota = allocate({str(k): int(v) for k, v in sizes.items()}, N_LABEL, MIN_PER_CELL)
    chosen: list[str] = []
    for cell in sorted(quota):
        ids = sorted(pool.loc[pool["stratum"] == cell, "doc_id"])
        chosen += rng.sample(ids, quota[cell])
    rest = sorted(set(pool["doc_id"]) - set(chosen))
    practice = rng.sample(rest, N_PRACTICE)
    second = set(rng.sample(sorted(chosen), N_SECOND_PASS))

    label_order = rng.sample(chosen, len(chosen))
    rows = [
        {"code": f"S{i:02d}", "doc_id": d, "role": "label", "second_pass": d in second}
        for i, d in enumerate(label_order, start=1)
    ] + [
        {"code": f"P{i}", "doc_id": d, "role": "practice", "second_pass": False}
        for i, d in enumerate(sorted(practice), start=1)
    ]
    plan = pd.DataFrame(rows).merge(pool[["doc_id", "release_date", "stratum"]], on="doc_id")
    plan["sample_version"] = SAMPLE_VERSION
    cols = ["code", "doc_id", "release_date", "stratum", "role", "second_pass", "sample_version"]
    return plan[cols]


_FRACTIONS = {
    "1/8": "an eighth",
    "1/4": "a quarter",
    "3/8": "three eighths",
    "1/2": "a half",
    "5/8": "five eighths",
    "3/4": "three quarters",
    "7/8": "seven eighths",
}
_UNICODE_FRACTIONS = {"¼": "1/4", "½": "1/2", "¾": "3/4", "⅛": "1/8"}
_MIXED = re.compile(r"\b(\d+)[-\s]?(\d/\d)\b")
_SIMPLE = re.compile(r"(?<![\d/])(\d/\d)(?![\d/])")


def speakable(text: str) -> str:
    """The text with the Fed's fractions written in words, for the listening copy only.

    "5-1/4 percent" becomes "5 and a quarter percent" and "1/4 percentage point" becomes "a
    quarter percentage point". Text-to-speech engines read "5-1/4" as "five minus one slash
    four", and a PDF line break inside "2-1/2" can split it into two sentences. The labelled
    text keeps the original wording; only what is heard changes.
    """
    # The PDF's standard font has no glyph for these: they would become "?" and split sentences.
    text = re.sub(r"(\d)\s*[–—]\s*(\d)", r"\1 to \2", text)  # a vote of 10–2
    text = text.replace("‐", "-").replace("‑", "-").replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"').replace("—", ", ").replace("–", ", ")
    for char, plain in _UNICODE_FRACTIONS.items():
        text = re.sub(rf"(\d){char}", rf"\1-{plain}", text).replace(char, plain)

    def mixed(m: re.Match[str]) -> str:
        frac = _FRACTIONS.get(m.group(2))
        return m.group(0) if frac is None else f"{m.group(1)} and {frac}"

    def simple(m: re.Match[str]) -> str:
        return _FRACTIONS.get(m.group(1), m.group(0))

    return _SIMPLE.sub(simple, _MIXED.sub(mixed, text))


def chapter_title(code: str) -> str:
    return f"Practice statement {code[1:]}" if code.startswith("P") else f"Statement {code}"


def write_listening_pdf(plan: pd.DataFrame, texts: dict[str, str], path: Path) -> None:
    """A PDF with one chapter per statement, practice first, for the AudioBook reader.

    Each chapter is titled by its code only: no date, no URL. Fractions are written in words
    (``speakable``). Paragraphs flow over A4 pages; the outline gives AudioBook its chapters.
    """
    import pymupdf

    width, height, margin = 595.0, 842.0, 60.0
    doc = pymupdf.open()
    toc: list[list[int | str]] = []
    ordered = pd.concat([plan[plan["role"] == "practice"], plan[plan["role"] == "label"]])
    for code, doc_id in zip(ordered["code"], ordered["doc_id"], strict=True):
        page = doc.new_page(width=width, height=height)
        toc.append([1, chapter_title(code), doc.page_count])
        page.insert_textbox(
            pymupdf.Rect(margin, margin, width - margin, margin + 40),
            chapter_title(code),
            fontsize=20,
            fontname="hebo",
        )
        y = margin + 60
        for paragraph in speakable(texts[doc_id]).split("\n\n"):
            while True:
                if y > height - margin - 30:  # no room left for a line
                    page = doc.new_page(width=width, height=height)
                    y = margin
                rect = pymupdf.Rect(margin, y, width - margin, height - margin)
                left = page.insert_textbox(rect, paragraph, fontsize=11, fontname="tiro")
                if left >= 0:
                    y = height - margin - left + 12
                    break
                # Nothing is written when the text does not fit: move to a new page.
                if y == margin:
                    raise ValueError(f"{doc_id}: a paragraph longer than a page")
                page = doc.new_page(width=width, height=height)
                y = margin
    doc.set_toc(toc)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)


# ---------------------------------------------------------------------------------------------
# Human labels. The labeller fills the same fields as the model, without evidence spans, and may
# mark any field as unsure. One JSON file per statement and pass under data/labels/.

GUIDE_VERSION = "fomc-label-guide-v1"
Pass = Literal["practice", "first", "second"]

# Plain-French help shown next to each field. The binding definitions are the English field
# descriptions of the extraction schema, shown alongside; this text only explains them.
HELP_FR: dict[str, str] = {
    "policy_action": (
        "La décision du jour sur le taux directeur, écrite en toutes lettres : « decided to "
        "raise / lower / maintain the target range ». Indique la taille en points de base "
        "(1/4 point = 25 pb ; négatif pour une baisse). « Pas de décision de taux » quand le "
        "texte parle d'autre chose (lignes de swap, cadre de politique, facilité)."
    ),
    "forward_guidance": (
        "Ce que la Fed laisse entendre pour la SUITE des taux, pas la décision du jour. "
        "−2 : promesse explicite de garder les taux bas longtemps (« for an extended period », "
        "« until … »). −1 : penche vers la patience ou la baisse. 0 : ne penche d'aucun côté. "
        "+1 : laisse entendre d'autres hausses (« further firming », « gradual increases »). "
        "+2 : hausses fortes ou continues clairement annoncées. Vide : rien sur l'avenir des taux."
    ),
    "inflation_assessment": (
        "À quel point la Fed s'inquiète d'une inflation trop HAUTE. −2 : peur d'une inflation "
        "trop basse ou d'une baisse des prix. −1 : inflation faible, sous l'objectif. 0 : proche "
        "de l'objectif, stable. +1 : élevée, en hausse, au-dessus de l'objectif. +2 : présentée "
        "comme LA préoccupation. Vide : l'inflation n'est pas jugée."
    ),
    "labor_market_assessment": (
        "La solidité de l'emploi selon la Fed. −2 : se dégrade fortement. −1 : faible, beaucoup "
        "de chômage. 0 : stable ou mitigé. +1 : solide, s'améliore. +2 : très fort, tendu. "
        "Vide : non évoqué."
    ),
    "growth_assessment": (
        "La vigueur de l'activité économique. −2 : recul, forte dégradation. −1 : lente, "
        "ralentit. 0 : modérée ou mitigée. +1 : solide, accélère. +2 : forte. Vide : non évoqué."
    ),
    "balance_sheet": (
        "Achats ou réductions d'obligations par la Fed (QE / QT). −2 : nouveaux achats ou achats "
        "plus gros. −1 : achats ou réinvestissements maintenus, ou fin/ralentissement de la "
        "réduction. 0 : neutre. +1 : réduction des achats, ou réduction du bilan en cours. "
        "+2 : réduction accélérée ou ventes. Vide : bilan non évoqué."
    ),
    "risk_balance": (
        "De quel côté la Fed voit le risque principal : inflation trop haute, ou croissance / "
        "emploi trop faibles (ou inflation trop basse), ou équilibré. Vide : pas dit."
    ),
    "dissents": (
        "Le paragraphe des votes : « Voting against the action was … who preferred … ». Compte "
        "les votes contre selon ce que le membre voulait : plus de fermeté (taux plus haut), "
        "plus de souplesse (taux plus bas), ou autre raison."
    ),
}


class HumanLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    pass_: Pass = Field(alias="pass")
    labelled_at: datetime
    guide_version: str = GUIDE_VERSION
    schema_version: str = SCHEMA_VERSION
    policy_action: Literal["raise", "cut", "hold", "no_rate_decision"] | None
    change_bp: int | None
    forward_guidance: Stance | None
    inflation_assessment: Stance | None
    labor_market_assessment: Stance | None
    growth_assessment: Stance | None
    balance_sheet: Stance | None
    risk_balance: Literal["inflation", "balanced", "growth"] | None
    dissents_tighter: int = Field(ge=0)
    dissents_easier: int = Field(ge=0)
    dissents_other: int = Field(ge=0)
    unsure: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _check(self) -> HumanLabel:
        known = {*HELP_FR}
        if unknown := set(self.unsure) - known:
            raise ValueError(f"unknown fields marked unsure: {sorted(unknown)}")
        if self.policy_action in ("hold", "no_rate_decision") and self.change_bp not in (0, None):
            raise ValueError("a hold or no rate decision has no change in bp")
        return self


def label_path(root: Path, pass_: str, code: str) -> Path:
    return root / pass_ / f"{code}.json"


def save_label(root: Path, label: HumanLabel) -> Path:
    path = label_path(root, label.pass_, label.code)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(label.model_dump_json(by_alias=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def load_label(root: Path, pass_: str, code: str) -> HumanLabel | None:
    path = label_path(root, pass_, code)
    return HumanLabel.model_validate_json(path.read_text("utf-8")) if path.is_file() else None
