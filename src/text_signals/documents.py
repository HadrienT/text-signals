"""The document model shared by every corpus.

A document is keyed by its **release** date (and time, when the source states one), never by the
date of the event it describes. Market data is joined on the release date only.
"""

from __future__ import annotations

from datetime import date, time
from typing import Literal

from pydantic import BaseModel, ConfigDict

# Where the time of day of a release comes from. "document": stated on the page itself
# (e.g. "For release at 2:00 p.m. EST"). "none": the source gives the date only.
ReleaseTimeSource = Literal["document", "none"]


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str  # stable id, e.g. "fomc-statement-20240131"
    corpus: str  # e.g. "fomc"
    doc_type: str  # e.g. "statement", "minutes", "press_conference"
    event_date: date | None  # date of the event described (e.g. last day of the meeting)
    release_date: date  # calendar date of publication, US Eastern time
    release_time_et: time | None  # time of day of publication, US Eastern time, if known
    release_time_source: ReleaseTimeSource
    unscheduled: bool  # intermeeting action, conference call or notation vote
    url: str
    raw_sha256: str  # sha256 of the raw file as fetched (see the fetch manifest)
    title: str
    text: str
    # Publication status stated by the source when it revises a document after release, e.g. a
    # press conference transcript first posted as "PRELIMINARY" and later as "FINAL".
    source_status: str | None = None
