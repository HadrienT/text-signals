"""Parse the FOMC meeting calendars on federalreserve.gov.

Two page layouts exist:

- ``fomccalendars.htm`` lists the last five years or so, one ``div.fomc-meeting`` row per
  meeting;
- ``fomchistoricalYYYY.htm`` (one page per year, linked from ``fomc_historical_year.htm``) lists
  older years, one ``div.panel`` per meeting or conference call, headed by an ``<h5>``.

Both are turned into :class:`CalendarEntry` records. Only links are extracted here; the documents
themselves are parsed elsewhere.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel

BASE_URL = "https://www.federalreserve.gov"
CURRENT_CALENDAR_URL = f"{BASE_URL}/monetarypolicy/fomccalendars.htm"
HISTORICAL_INDEX_URL = f"{BASE_URL}/monetarypolicy/fomc_historical_year.htm"

# Labels that mark a meeting held outside the regular schedule.
_UNSCHEDULED_RE = re.compile(r"conference call|unscheduled|notation vote", re.IGNORECASE)
_RELEASED_RE = re.compile(r"Released\s+([A-Z][a-z]+\.?\s+\d{1,2},\s+\d{4})")


class CalendarEntry(BaseModel):
    year: int
    label: str  # e.g. "January 30-31", "May 9 Conference Call"
    unscheduled: bool
    statement_urls: list[str]
    minutes_url: str | None
    minutes_release_date: date | None
    press_conference_url: str | None

    @property
    def meeting_date(self) -> date:
        """Last day of the meeting, read from the label ("Apr/May 30-1" -> May 1)."""
        return meeting_end_date(self.label, self.year)


_MONTHS = {
    m.lower(): i
    for i, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for m in names
}
_LABEL_RE = re.compile(
    r"^([A-Za-z]+(?:/[A-Za-z]+)?)\.?\s+(\d{1,2})"  # "January 31", "Apr/May 30"
    r"(?:\s*-\s*(?:([A-Za-z]+)\.?\s+)?(\d{1,2}))?"  # "-1", "-February 1"
)


def meeting_end_date(label: str, year: int) -> date:
    """Last day of a meeting from a calendar label.

    Handles "January 30-31", "Apr/May 30-1" (current calendar) and "January 31-February 1"
    (historical pages). ``year`` is the year the calendar lists the meeting under.
    """
    m = _LABEL_RE.match(label.strip())
    if m is None:
        raise ValueError(f"cannot parse meeting label {label!r}")
    months, first_day, end_month, last_day = m.groups()
    month = _MONTHS[(end_month or months.split("/")[-1]).lower()]
    return date(year, month, int(last_day or first_day))


def _abs(href: str) -> str:
    return href if href.startswith("http") else BASE_URL + href


def parse_release_date(text: str) -> date | None:
    """Parse "(Released February 21, 2024)" or "(Released Feb 17, 2010)" anywhere in ``text``."""
    m = _RELEASED_RE.search(text)
    if m is None:
        return None
    raw = m.group(1).replace(".", "")
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    # "Sept 21, 2010" style abbreviations
    month, rest = raw.split(" ", 1)
    return datetime.strptime(f"{month[:3]} {rest}", "%b %d, %Y").date()


def _links(block: Tag) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for a in block.find_all("a", href=True):
        href = str(a["href"]).strip()
        if href:
            out.append((a.get_text(" ", strip=True), _abs(href)))
    return out


def _is_statement(text: str, url: str) -> bool:
    # Current layout: the HTML link under "Statement:" points at .../monetaryYYYYMMDDa.htm.
    # Historical layout: the anchor text is "Statement".
    if re.search(r"/pressreleases/monetary\d{8}a\.htm$", url):
        return True
    return text.strip().lower() == "statement" and url.endswith((".htm", "/"))


def _is_minutes_html(url: str) -> bool:
    return "minutes" in url.lower() and url.lower().endswith(".htm")


def _entry(year: int, label: str, block: Tag) -> CalendarEntry:
    links = _links(block)
    statements = [u for t, u in links if _is_statement(t, u)]
    minutes = next((u for _, u in links if _is_minutes_html(u)), None)
    presconf = next((u for _, u in links if re.search(r"fomcpresconf\d{8}\.htm$", u)), None)
    return CalendarEntry(
        year=year,
        label=label,
        unscheduled=bool(_UNSCHEDULED_RE.search(label)),
        statement_urls=list(dict.fromkeys(statements)),
        minutes_url=minutes,
        minutes_release_date=parse_release_date(block.get_text(" ", strip=True))
        if minutes
        else None,
        press_conference_url=presconf,
    )


def parse_current_calendar(html: str | bytes) -> list[CalendarEntry]:
    soup = BeautifulSoup(html, "lxml")
    entries: list[CalendarEntry] = []
    for panel in soup.select("div.panel"):
        heading = panel.find("h4")
        if heading is None:
            continue
        m = re.match(r"(\d{4}) FOMC Meetings", heading.get_text(strip=True))
        if m is None:
            continue
        year = int(m.group(1))
        for row in panel.select("div.fomc-meeting"):
            month = row.select_one(".fomc-meeting__month")
            day = row.select_one(".fomc-meeting__date")
            if month is None or day is None:
                continue
            label = f"{month.get_text(strip=True)} {day.get_text(' ', strip=True)}"
            entries.append(_entry(year, label, row))
    return entries


def parse_historical_year(html: str | bytes, year: int) -> list[CalendarEntry]:
    soup = BeautifulSoup(html, "lxml")
    entries: list[CalendarEntry] = []
    for panel in soup.select("div.panel"):
        heading = panel.find("h5")
        if heading is None:
            continue
        text = heading.get_text(" ", strip=True)
        m = re.match(rf"(.+?)\s*-\s*{year}$", text)
        if m is None:
            continue
        label = re.sub(r"\s+Meeting$", "", m.group(1))
        entries.append(_entry(year, label, panel))
    return entries


def historical_year_urls(index_html: str | bytes, first_year: int) -> dict[int, str]:
    """Map each year >= ``first_year`` to its historical calendar page URL."""
    soup = BeautifulSoup(index_html, "lxml")
    out: dict[int, str] = {}
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        m = re.search(r"fomchistorical(\d{4})\.htm$", href)
        if m and int(m.group(1)) >= first_year:
            out[int(m.group(1))] = _abs(href)
    return dict(sorted(out.items()))
