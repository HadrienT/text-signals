from pathlib import Path

import pytest
from pypdf.errors import PdfReadError

from text_signals.sources.fomc.press_conferences import TRANSCRIPT_URL, parse_transcript

# Pages 1 and 17 of the transcript of 28 October 2026, as first posted (PRELIMINARY).
FIXTURE = Path(__file__).parent / "fixtures" / "fomc" / "presconf_20261028_pages_1_and_17.pdf"


def test_transcript_header_gives_date_and_status() -> None:
    parsed = parse_transcript(FIXTURE.read_bytes())
    assert parsed.held_on.isoformat() == "2026-10-28"
    assert parsed.status == "PRELIMINARY"
    assert parsed.title == "Transcript of Chairman Warsh’s Press Conference"


def test_transcript_has_one_paragraph_per_speaker_turn() -> None:
    turns = parse_transcript(FIXTURE.read_bytes()).text.split("\n\n")
    assert turns[0].startswith("CHAIRMAN WARSH. Good day. This morning the FOMC decided")
    assert turns[-3:] == [
        "CHAIRMAN WARSH. Not to my knowledge.",
        "MICHELLE SMITH. Thank you.",
        "CHAIRMAN WARSH. Thank you guys.",
    ]
    # Running headers and page numbers are removed.
    assert not any("Page 1 of 17" in t or "Press Conference PRELIMINARY" in t for t in turns)


def test_not_a_transcript_is_rejected() -> None:
    with pytest.raises(PdfReadError):
        parse_transcript(b"not a pdf")


def test_transcript_url_from_meeting_date() -> None:
    from datetime import date

    assert TRANSCRIPT_URL.format(date(2011, 4, 27)).endswith("/FOMCpresconf20110427.pdf")
