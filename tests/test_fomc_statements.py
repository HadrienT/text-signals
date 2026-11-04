from datetime import date, time
from pathlib import Path

import pytest

from text_signals.sources.fomc.statements import parse_release_time, parse_statement

FIXTURES = Path(__file__).parent / "fixtures" / "fomc"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("For release at 2:00 p.m. EST", time(14, 0)),
        ("For release at 2:15 p.m. EDT", time(14, 15)),
        ("For release at 8:30 a.m. EST", time(8, 30)),
        ("For release at 12:30 p.m. EST", time(12, 30)),
        ("For immediate release", None),
    ],
)
def test_parse_release_time(text: str, expected: time | None) -> None:
    assert parse_release_time(text) == expected


def test_modern_statement() -> None:
    s = parse_statement((FIXTURES / "statement_modern_2024.htm").read_bytes())
    assert s.release_date == date(2024, 1, 31)
    assert s.release_time_et == time(14, 0)
    assert s.title == "Federal Reserve issues FOMC statement"
    paragraphs = s.text.split("\n\n")
    assert len(paragraphs) == 3
    assert paragraphs[0].startswith("Recent indicators suggest")
    assert "5-1/4 to 5-1/2 percent" in paragraphs[1]
    assert paragraphs[-1].startswith("Voting for the monetary policy action")
    assert (
        "media inquiries" not in s.text
        and "PDF" not in s.text
        and "Implementation Note" not in s.text
    )


def test_legacy_1994_statement() -> None:
    s = parse_statement((FIXTURES / "statement_legacy_1994.htm").read_bytes())
    assert s.release_date == date(1994, 2, 4)
    assert s.release_time_et is None  # "For immediate release": no time of day stated
    paragraphs = s.text.split("\n\n")
    assert len(paragraphs) == 2
    assert paragraphs[0].startswith("Chairman Alan Greenspan announced today")
    assert "Last update" not in s.text and "Home" not in s.text


def test_legacy_1999_tag_soup() -> None:
    s = parse_statement((FIXTURES / "statement_legacy_1999.htm").read_bytes())
    assert s.release_date == date(1999, 5, 18)
    # "Last update: May 18, 1999, 2:15 PM" is a page timestamp, not a release time.
    assert s.release_time_et is None
    intro, body = s.text.split("\n\n")
    assert intro.endswith("Federal Open Market Committee meeting:")
    # The inline <i> does not split the paragraph.
    assert "a buildup of inflationary imbalances." in body
    assert "Monetary policy" not in s.text


def test_empty_page_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_statement(b"<html><body><p>nothing</p></body></html>")
