from datetime import date
from pathlib import Path

import pytest

from text_signals.sources.fomc.calendar import (
    meeting_end_date,
    parse_current_calendar,
    parse_historical_year,
    parse_release_date,
)

FIXTURES = Path(__file__).parent / "fixtures" / "fomc"


@pytest.mark.parametrize(
    ("label", "year", "expected"),
    [
        ("January 30-31", 2024, date(2024, 1, 31)),
        ("Apr/May 30-1", 2024, date(2024, 5, 1)),
        ("Jan/Feb 31-1", 2023, date(2023, 2, 1)),
        ("January 31-February 1", 1995, date(1995, 2, 1)),
        ("July 31-August 1", 2012, date(2012, 8, 1)),
        ("March 19-20*", 2024, date(2024, 3, 20)),
        ("May 9 Conference Call", 2010, date(2010, 5, 9)),
        ("August 22 (notation vote)", 2025, date(2025, 8, 22)),
        ("Sept 21", 2010, date(2010, 9, 21)),
    ],
)
def test_meeting_end_date(label: str, year: int, expected: date) -> None:
    assert meeting_end_date(label, year) == expected


def test_parse_release_date() -> None:
    assert parse_release_date("(Released February 21, 2024)") == date(2024, 2, 21)
    assert parse_release_date("Minutes (Released Feb 17, 2010):") == date(2010, 2, 17)
    assert parse_release_date("no date here") is None


def test_historical_layout() -> None:
    entries = parse_historical_year((FIXTURES / "calendar_historical_2010.htm").read_bytes(), 2010)
    assert [e.label for e in entries] == [
        "January 26-27",
        "May 9 Conference Call",
        "October 15 Conference Call",
    ]
    jan, may, octo = entries
    assert not jan.unscheduled
    assert jan.meeting_date == date(2010, 1, 27)
    assert jan.statement_urls == [
        "https://www.federalreserve.gov/newsevents/press/monetary/20100127a.htm"
    ]
    assert (
        jan.minutes_url == "https://www.federalreserve.gov/monetarypolicy/fomcminutes20100127.htm"
    )
    assert jan.minutes_release_date == date(2010, 2, 17)
    assert may.unscheduled and len(may.statement_urls) == 1
    # A conference call without a statement (only a transcript, released years later).
    assert octo.unscheduled and octo.statement_urls == [] and octo.minutes_url is None


def test_current_layout() -> None:
    entries = parse_current_calendar((FIXTURES / "calendar_current.htm").read_bytes())
    jan, aprmay, aug = entries
    assert (jan.year, jan.meeting_date) == (2024, date(2024, 1, 31))
    # Only the policy statement, not the implementation note nor the longer-run goals statement.
    assert jan.statement_urls == [
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240131a.htm"
    ]
    assert jan.minutes_release_date == date(2024, 2, 21)
    assert jan.press_conference_url is not None
    assert aprmay.meeting_date == date(2024, 5, 1)
    assert aprmay.minutes_url is None and aprmay.minutes_release_date is None
    assert aug.year == 2025 and aug.unscheduled


def test_minutes_link_without_the_word_minutes_in_its_url() -> None:
    # fomchistorical2008.htm: the June 2008 minutes are at /monetarypolicy/fomc20080625.htm.
    html = """<html><body><div class="panel panel-default">
      <div class="panel-heading"><h5>June 24-25 Meeting - 2008</h5></div>
      <div class="row divided-row panel-body"><div class="col-xs-12 col-md-6">
        <p><a href="/monetarypolicy/2008-fomc-memos.htm">2008 Memos</a></p>
        <p><a href="/newsevents/press/monetary/20080625a.htm">Statement</a></p>
        <p>Minutes (Released Jul 16, 2008): <br />
          <a href="/monetarypolicy/fomc20080625.htm">HTML</a> |
          <a href="/monetarypolicy/files/fomcminutes20080625.pdf">310 KB PDF</a></p>
      </div></div></div></body></html>"""
    [entry] = parse_historical_year(html, 2008)
    assert entry.minutes_url == "https://www.federalreserve.gov/monetarypolicy/fomc20080625.htm"
    assert entry.minutes_release_date == date(2008, 7, 16)
