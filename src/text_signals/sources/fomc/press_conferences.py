"""Parse the Chair's press conference transcripts (PDF, April 2011 onwards).

The transcript of the press conference held after a meeting is a PDF at
``/mediacenter/files/FOMCpresconfYYYYMMDD.pdf``. Recent meeting pages no longer link it, but the
file is served at that address; the URL is built from the meeting date (protocol §3.1).

Every page starts with a running header ("April 27, 2011 Chairman Bernanke's Press Conference
FINAL", then "1 of 26" or "Page 1 of 17"), which is removed. The header also gives the status of
the transcript: the Fed first posts a PRELIMINARY version and replaces it with a FINAL one, so the
status is stored with the document and the file's sha256 pins the version we read.

The text is kept as spoken, with speaker labels ("CHAIRMAN POWELL.", "MICHELLE SMITH."); one
paragraph per speaker turn.
"""

from __future__ import annotations

import io
import re
from datetime import date, datetime

from pydantic import BaseModel
from pypdf import PdfReader

TRANSCRIPT_URL = "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{:%Y%m%d}.pdf"

# Every scheduled meeting since January 2019 is followed by a press conference. Meetings whose
# press conference page is missing from the Fed's site (so the calendar has no link) but whose
# transcript is served, with the evidence (checked 2026-11-05, protocol §3.5).
WITHOUT_PAGE: dict[date, str] = {
    date(2026, 1, 28): (
        "fomcpresconf20260128.htm answers 404; FOMCpresconf20260128.pdf is Chair Powell's FINAL "
        "transcript dated January 28, 2026; USMPD has a press conference at 2:30 p.m. that day"
    ),
}

# Header and title are matched with spaces removed: the PDF text layer sometimes splits a word
# ("P ress Conference", transcript of 18 March 2015).
_HEADER = re.compile(
    r"^([A-Z][a-z]+\d{1,2},\d{4})(.*?PressConference(?:Call)?)(FINAL|PRELIMINARY)$"
)
_PAGE_NUMBER = re.compile(r"^\s*(Page\s+)?\d+\s+of\s+\d+\s*$")
_TITLE = re.compile(r"^Transcriptof.*PressConference(?:Call)?$")
# A speaker label opens a turn: two or more upper-case words ending with a period, e.g.
# "CHAIRMAN POWELL." (one word would also catch acronyms that open a line: "TGA.", "NPR.").
_SPEAKER = re.compile(r"^([A-Z][A-Z’'\-]*(?: [A-Z][A-Z.’'\-]*)+)\.\s")


class ParsedTranscript(BaseModel):
    title: str  # e.g. "Transcript of Chairman Bernanke's Press Conference"
    held_on: date  # the date stated in the running header
    status: str  # "FINAL" or "PRELIMINARY"
    text: str


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def _squeeze(s: str) -> str:
    return re.sub(r"\s+", "", s.replace("\xa0", " "))


def parse_transcript(data: bytes) -> ParsedTranscript:
    pages = [page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages]
    if not pages:
        raise ValueError("transcript PDF without pages")
    first_line = next((ln for ln in pages[0].splitlines() if ln.strip()), "")
    header = _HEADER.match(_squeeze(first_line))
    if header is None:
        raise ValueError(f"transcript without the running header: {first_line!r}")
    held_on = datetime.strptime(header.group(1), "%B%d,%Y").date()
    status = header.group(3)

    title = ""
    lines: list[str] = []
    for page in pages:
        for raw in page.splitlines():
            line = _clean(raw)
            if not line or _PAGE_NUMBER.match(line):
                continue
            m = _HEADER.match(_squeeze(line))
            if m is not None:
                if m.group(3) != status:
                    raise ValueError(f"transcript pages disagree on status: {status}, {m.group(3)}")
                continue
            if _TITLE.match(_squeeze(line)):
                title = title or line
                continue
            if _squeeze(line) == header.group(1) and not lines:
                continue  # the date line under the title
            lines.append(line)

    turns: list[str] = []
    for line in lines:
        if _SPEAKER.match(line) or not turns:
            turns.append(line)
        else:
            turns[-1] = f"{turns[-1]} {line}"
    if not turns:
        raise ValueError("transcript with an empty body")
    return ParsedTranscript(
        title=title or "Transcript of the Chair's Press Conference",
        held_on=held_on,
        status=status,
        text="\n\n".join(turns),
    )
