"""Parse FOMC statement press releases into text plus release date and time.

Two HTML layouts:

- **modern** (``/newsevents/pressreleases/monetaryYYYYMMDDa.htm``, 2006 onwards, older URLs
  redirect there): a ``div#article`` with ``p.article__time`` (date), ``p.releaseTime``
  ("For release at 2:00 p.m. EST", when stated) and the body paragraphs;
- **legacy** (``/fomc/YYYYMMDDdefault.htm`` and ``/boarddocs/press/...``, 1994-2005): a plain page
  with "Release Date: February 4, 1994", "For immediate release" and the body paragraphs.

The text is kept as published, vote and related actions included. Only page furniture (media
contact line, links to the implementation note, navigation) is removed.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel

_RELEASE_TIME_RE = re.compile(r"For release at\s+(\d{1,2}):(\d{2})\s*([ap])\.?\s*m\.?", re.I)
_RELEASE_DATE_RE = re.compile(r"Release Date:\s*([A-Z][a-z]+\s+\d{1,2},\s*\d{4})")
_BOILERPLATE_RE = re.compile(
    r"^(For media inquiries|Implementation Note issued|Attachment|Related Information|"
    r"Release Date:|For immediate release|For release at|Last update:|Home \|)",
    re.I,
)


_LEGACY_FOOTER_RE = re.compile(r"^(\d{4} Monetary policy|Home(\s*\|.*)?$|Last update:)", re.I)


class ParsedStatement(BaseModel):
    title: str
    release_date: date
    release_time_et: time | None
    text: str


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_release_time(text: str) -> time | None:
    m = _RELEASE_TIME_RE.search(text)
    if m is None:
        return None
    hour, minute, ampm = int(m.group(1)), int(m.group(2)), m.group(3).lower()
    if ampm == "p" and hour != 12:
        hour += 12
    if ampm == "a" and hour == 12:
        hour = 0
    return time(hour, minute)


def _paragraphs(container: Tag) -> list[str]:
    out = []
    for p in container.find_all("p"):
        text = _clean(p.get_text(" ", strip=True))
        if text and not _BOILERPLATE_RE.match(text):
            out.append(text)
    return out


def _parse_modern(article: Tag) -> ParsedStatement:
    date_tag = article.select_one("p.article__time")
    title_tag = article.select_one("h3.title")
    time_tag = article.select_one("p.releaseTime")
    if date_tag is None or title_tag is None:
        raise ValueError("modern statement page without date or title")
    bodies = [
        d
        for d in article.find_all("div", recursive=False)
        if "col-sm-8" in (d.get("class") or []) and "heading" not in (d.get("class") or [])
    ]
    if not bodies:
        raise ValueError("modern statement page without a body")
    return ParsedStatement(
        title=_clean(title_tag.get_text(" ", strip=True)),
        release_date=datetime.strptime(_clean(date_tag.get_text()), "%B %d, %Y").date(),
        release_time_et=parse_release_time(time_tag.get_text(" ")) if time_tag else None,
        text="\n\n".join(_paragraphs(bodies[0])),
    )


def _legacy_paragraphs(container: Tag) -> list[str]:
    """Paragraphs of a legacy page, robust to unclosed ``<p>`` and text placed in table cells.

    Each text node becomes a line; a line is glued to the previous one when the previous one
    does not end a sentence (an inline ``<a>`` or ``<i>`` splits a paragraph into several nodes).
    The body runs from "For immediate release" to the footer.
    """
    lines = [_clean(x) for x in container.get_text("\n").splitlines()]
    lines = [x for x in lines if x]
    start = next((i + 1 for i, x in enumerate(lines) if x.lower() == "for immediate release"), 0)
    paragraphs: list[str] = []
    for line in lines[start:]:
        if _LEGACY_FOOTER_RE.match(line):
            break
        if _BOILERPLATE_RE.match(line):
            continue
        if paragraphs and not paragraphs[-1].endswith((".", ":", ";", '"', ")")):
            paragraphs[-1] = f"{paragraphs[-1]} {line}"
        else:
            paragraphs.append(line)
    return paragraphs


def _parse_legacy(soup: BeautifulSoup) -> ParsedStatement:
    page_text = _clean(soup.get_text(" "))
    m = _RELEASE_DATE_RE.search(page_text)
    if m is None:
        raise ValueError("legacy statement page without a 'Release Date:' line")
    container = soup.select_one("#content") or soup.body
    if container is None:
        raise ValueError("legacy statement page without a body")
    title = _clean(soup.title.get_text()) if soup.title else ""
    return ParsedStatement(
        title=title,
        release_date=datetime.strptime(_clean(m.group(1)), "%B %d, %Y").date(),
        release_time_et=parse_release_time(page_text),
        text="\n\n".join(_legacy_paragraphs(container)),
    )


def parse_statement(html: str | bytes) -> ParsedStatement:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    article = soup.select_one("div#article")
    parsed = _parse_modern(article) if article is not None else _parse_legacy(soup)
    if not parsed.text:
        raise ValueError("statement page with an empty body")
    return parsed
