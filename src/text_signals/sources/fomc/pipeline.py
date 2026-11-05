"""Fetch the FOMC calendar, statements, minutes and press conference transcripts, and write one
``Document`` per release.

Run with ``uv run python -m text_signals.sources.fomc`` (see ``just fetch-fomc``).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from text_signals.documents import Document
from text_signals.fetch import Fetcher
from text_signals.sources.fomc.calendar import (
    CURRENT_CALENDAR_URL,
    HISTORICAL_INDEX_URL,
    CalendarEntry,
    historical_year_urls,
    parse_current_calendar,
    parse_historical_year,
)
from text_signals.sources.fomc.minutes import RELEASE_DATE_CORRECTIONS, parse_minutes
from text_signals.sources.fomc.press_conferences import (
    TRANSCRIPT_URL,
    WITHOUT_PAGE,
    parse_transcript,
)
from text_signals.sources.fomc.statements import parse_statement

log = logging.getLogger(__name__)

# FOMC statements start in February 1994.
FIRST_YEAR = 1994


def fetch_calendar(
    fetcher: Fetcher, *, first_year: int = FIRST_YEAR, refresh: bool = False
) -> list[CalendarEntry]:
    """All meetings from ``first_year`` on, from the historical pages and the current calendar.

    A year that appears on both (during the transition the Fed keeps both pages) is taken from
    the current calendar. ``refresh`` refetches the two index pages to pick up new meetings.
    """
    current_html, _ = fetcher.get(CURRENT_CALENDAR_URL, refresh=refresh)
    current = parse_current_calendar(current_html)
    current_years = {e.year for e in current}

    index_html, _ = fetcher.get(HISTORICAL_INDEX_URL, refresh=refresh)
    entries: list[CalendarEntry] = []
    for year, url in historical_year_urls(index_html, first_year).items():
        if year in current_years:
            continue
        html, _ = fetcher.get(url)
        entries.extend(parse_historical_year(html, year))
    entries.extend(e for e in current if e.year >= first_year)
    return sorted(entries, key=lambda e: e.meeting_date)


def build_statements(fetcher: Fetcher, entries: Iterable[CalendarEntry]) -> list[Document]:
    docs: dict[str, Document] = {}
    for entry in entries:
        for url in entry.statement_urls:
            body, rec = fetcher.get(url)
            parsed = parse_statement(body)
            doc_id = f"fomc-statement-{parsed.release_date:%Y%m%d}"
            if doc_id in docs:
                if docs[doc_id].raw_sha256 == rec.sha256:
                    continue  # the same release linked twice
                n = sum(k == doc_id or k.startswith(f"{doc_id}-") for k in docs)
                doc_id = f"{doc_id}-{n + 1}"
            if parsed.release_date < entry.meeting_date:
                raise ValueError(f"{url}: released {parsed.release_date} before the meeting")
            docs[doc_id] = Document(
                doc_id=doc_id,
                corpus="fomc",
                doc_type="statement",
                event_date=entry.meeting_date,
                release_date=parsed.release_date,
                release_time_et=parsed.release_time_et,
                release_time_source="document" if parsed.release_time_et else "none",
                unscheduled=entry.unscheduled,
                url=url,
                raw_sha256=rec.sha256,
                title=parsed.title,
                text=parsed.text,
            )
    return sorted(docs.values(), key=lambda d: (d.release_date, d.doc_id))


def build_minutes(fetcher: Fetcher, entries: Iterable[CalendarEntry]) -> list[Document]:
    """One document per minutes page, released on the date the calendar gives for it."""
    docs: dict[str, Document] = {}
    seen: set[str] = set()
    for entry in entries:
        if entry.minutes_url is None or entry.minutes_url in seen:
            continue
        seen.add(entry.minutes_url)
        released = entry.minutes_release_date
        if released is None:
            raise ValueError(f"{entry.minutes_url}: no release date in the calendar")
        if entry.meeting_date in RELEASE_DATE_CORRECTIONS:
            released, why = RELEASE_DATE_CORRECTIONS[entry.meeting_date]
            log.info("minutes of %s: release date %s (%s)", entry.meeting_date, released, why)
        if released <= entry.meeting_date:
            raise ValueError(f"{entry.minutes_url}: released before or on the meeting day")
        body, rec = fetcher.get(entry.minutes_url)
        parsed = parse_minutes(body)
        doc_id = f"fomc-minutes-{released:%Y%m%d}"
        if doc_id in docs:
            raise ValueError(f"two minutes released on {released}")
        docs[doc_id] = Document(
            doc_id=doc_id,
            corpus="fomc",
            doc_type="minutes",
            event_date=entry.meeting_date,
            release_date=released,
            release_time_et=None,
            release_time_source="none",
            unscheduled=entry.unscheduled,
            url=entry.minutes_url,
            raw_sha256=rec.sha256,
            title=parsed.title,
            text=parsed.text,
        )
    return sorted(docs.values(), key=lambda d: (d.release_date, d.doc_id))


def build_press_conferences(fetcher: Fetcher, entries: Iterable[CalendarEntry]) -> list[Document]:
    """One document per press conference transcript, released on the day it was held.

    The press conference is held live right after the statement, so its content is public that
    day even though the transcript is posted later. The day is read from the press conference
    page's URL (``fomcpresconfYYYYMMDD.htm``), which is not always the meeting date: the call of
    2 March 2020 was followed by a press conference on 3 March. The transcript's header must
    state the same day.
    """
    docs: list[Document] = []
    for entry in entries:
        if entry.press_conference_url is None:
            if entry.meeting_date in WITHOUT_PAGE:
                held_on = entry.meeting_date
                log.info("press conference of %s: %s", held_on, WITHOUT_PAGE[held_on])
            else:
                if entry.year >= 2019 and not entry.unscheduled and entry.statement_urls:
                    log.warning(
                        "%s %s: scheduled meeting without a press conference link",
                        entry.year,
                        entry.label,
                    )
                continue
        else:
            m = re.search(r"fomcpresconf(\d{8})\.htm$", entry.press_conference_url)
            if m is None:
                raise ValueError(f"{entry.press_conference_url}: no date in the URL")
            held_on = datetime.strptime(m.group(1), "%Y%m%d").date()
            if held_on < entry.meeting_date:
                raise ValueError(f"{entry.press_conference_url}: held before the meeting")
        url = TRANSCRIPT_URL.format(held_on)
        body, rec = fetcher.get(url)
        parsed = parse_transcript(body)
        if parsed.held_on != held_on:
            raise ValueError(f"{url}: transcript dated {parsed.held_on}, page {held_on}")
        docs.append(
            Document(
                doc_id=f"fomc-press-conference-{held_on:%Y%m%d}",
                corpus="fomc",
                doc_type="press_conference",
                event_date=entry.meeting_date,
                release_date=held_on,
                release_time_et=None,
                release_time_source="none",
                unscheduled=entry.unscheduled,
                url=url,
                raw_sha256=rec.sha256,
                title=parsed.title,
                text=parsed.text,
                source_status=parsed.status.lower(),
            )
        )
    return sorted(docs, key=lambda d: (d.release_date, d.doc_id))


def write_jsonl(docs: Iterable[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(doc.model_dump_json() + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[Document]:
    return [
        Document.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run(data_dir: Path, *, delay_s: float = 2.0, refresh: bool = False) -> list[Document]:
    with Fetcher(data_dir / "raw", delay_s=delay_s) as fetcher:
        entries = fetch_calendar(fetcher, refresh=refresh)
        log.info("calendar: %d meetings", len(entries))
        docs = build_statements(fetcher, entries)
        minutes = build_minutes(fetcher, entries)
        pressers = build_press_conferences(fetcher, entries)
    out = data_dir / "interim" / "fomc"
    write_jsonl(docs, out / "statements.jsonl")
    write_jsonl(minutes, out / "minutes.jsonl")
    write_jsonl(pressers, out / "press_conferences.jsonl")
    log.info("statements: %d documents", len(docs))
    log.info("minutes: %d documents", len(minutes))
    preliminary = sum(d.source_status == "preliminary" for d in pressers)
    log.info("press conferences: %d transcripts, %d preliminary", len(pressers), preliminary)
    return docs
