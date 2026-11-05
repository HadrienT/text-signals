"""Parse the minutes of FOMC meetings (HTML, 1994 onwards).

Three URL layouts (``/fomc/MINUTES/YYYY/YYYYMMDDmin.htm``, ``/fomc/minutes/YYYYMMDD.htm`` and
``/monetarypolicy/fomcminutesYYYYMMDD.htm``) and two page layouts: the modern ``div#article``
and the legacy plain page (``div#content`` or the body). The text is split into paragraphs at
block elements (``<p>``, lists, ``<blockquote>``, table rows, line breaks); the body ends at the
last "Return to top" (legacy pages also repeat it between sections). Page furniture is removed:
the "Return to text" footnote links, the navigation footer and "Last update" lines. Footnotes
are kept, as published.

The minutes pages do not state their release date. It is read from the meeting calendar
("Released March 25, 1994") and was checked, for all 262 minutes, against the Fed's press release
index (2006 on), the "Last update" stamp of each page, USMPD (2001 on) and, where they disagree,
Internet Archive captures. Two calendar dates are wrong and are corrected below (protocol §3.5).
"""

from __future__ import annotations

import re
from datetime import date

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel

_FURNITURE = re.compile(r"^(Last update:|Home\s*\||FOMC$|Return to top$|Accessibility)", re.I)
_RETURN_TO_TEXT = re.compile(r"\s*Return to text\s*$")
_TITLE = re.compile(r"Minutes of the Federal Open Market Committee[^.]*?\d{4}")


# Meeting date -> (release date, evidence). Release dates of the meeting calendar that the
# evidence contradicts; see protocol §3.5 for the full check.
RELEASE_DATE_CORRECTIONS: dict[date, tuple[date, str]] = {
    date(2003, 10, 28): (
        date(2003, 12, 10),
        "calendar says 11 December; the page says 'Last update: December 10, 2003, 2:00 PM' and "
        "USMPD has 10 December 2:00 p.m. (probable, no archive capture of 10-11 December)",
    ),
    date(2004, 9, 21): (
        date(2004, 11, 12),
        "calendar and USMPD say 11 November, a federal holiday with the Treasury market closed; "
        "the page says 'Last update: November 12, 2004, 2:00 PM', and the FOMC page captured by "
        "the Internet Archive has no link to these minutes at 8:58 p.m. ET on 11 November and "
        "one at 7:58 p.m. ET on 12 November",
    ),
}


class ParsedMinutes(BaseModel):
    title: str
    text: str


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


# Elements that start a new paragraph. Table cells do not: a row is one paragraph, and legacy
# pages lay out their whole body in one table whose cells hold the paragraphs.
_BLOCK_TAGS = [
    "p", "li", "dd", "dt", "blockquote", "tr", "br", "div",
    "h1", "h2", "h3", "h4", "h5", "h6", "ol", "ul", "dl", "table",
]  # fmt: skip


def _blocks(container: Tag) -> list[str]:
    """The text of ``container`` split into paragraphs at block elements.

    Legacy pages put text outside ``<p>``: the policy directive in a ``<blockquote>`` inside a
    table cell, the votes directly in the cell, the standing authorizations in lists.
    """
    for tag in container.find_all(_BLOCK_TAGS):
        tag.insert_before("\n\n")
        tag.insert_after("\n\n")
    return [_clean(t) for t in re.split(r"\n\s*\n", container.get_text(" ")) if _clean(t)]


def parse_minutes(html: str | bytes) -> ParsedMinutes:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    container = soup.select_one("div#article") or soup.select_one("div#content") or soup.body
    if container is None:
        raise ValueError("minutes page without a body")
    texts = _blocks(container)
    # Legacy pages repeat "Return to top" between sections; the footer follows the last one.
    tops = [i for i, t in enumerate(texts) if t.lower() == "return to top"]
    if tops:
        texts = texts[: tops[-1]]
    paragraphs: list[str] = []
    for text in texts:
        text = _RETURN_TO_TEXT.sub("", text)
        if text and not _FURNITURE.match(text):
            paragraphs.append(text)
    if not paragraphs:
        raise ValueError("minutes page with an empty body")
    m = _TITLE.search(_clean(container.get_text(" ")))
    title = m.group(0) if m else _clean(soup.title.get_text()) if soup.title else ""
    return ParsedMinutes(title=title, text="\n\n".join(paragraphs))
