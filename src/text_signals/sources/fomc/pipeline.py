"""Fetch the FOMC calendar and statements, and write one ``Document`` per statement.

Run with ``uv run python -m text_signals.sources.fomc`` (see ``just fetch-fomc``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
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
    write_jsonl(docs, data_dir / "interim" / "fomc" / "statements.jsonl")
    log.info("statements: %d documents", len(docs))
    return docs
