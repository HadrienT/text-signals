from pathlib import Path

from text_signals.sources.fomc.minutes import parse_minutes

FIXTURES = Path(__file__).parent / "fixtures" / "fomc"


def test_legacy_minutes_end_at_the_last_return_to_top() -> None:
    parsed = parse_minutes((FIXTURES / "minutes_legacy_1994.htm").read_bytes())
    assert parsed.title == (
        "Minutes of the Federal Open Market Committee Meeting of February 3-4, 1994"
    )
    paragraphs = parsed.text.split("\n\n")
    assert paragraphs[:2] == [
        "Minutes of the Federal Open Market Committee",
        "Meeting of February 3-4, 1994",
    ]
    # Hard-wrapped lines inside a paragraph are joined.
    assert paragraphs[2].startswith("A meeting of the Federal Open Market Committee was held in")
    assert "\n" not in paragraphs[2]
    # A "Return to top" between sections is dropped and the text after it is kept.
    assert "The meeting adjourned at 11:45 a.m." in paragraphs
    assert paragraphs[-1] == "Secretary"  # the footer after the last "Return to top" is dropped


def test_legacy_text_outside_paragraph_tags_is_kept() -> None:
    # Minutes of July 1996: the directive is in a <blockquote> inside a layout table, the vote
    # directly in the table cell.
    html = """<html><body><div id="content"><table><tr><td>
      <P>The Committee approved the directive.</P>
      <blockquote>The Federal Open Market Committee seeks monetary and financial
      conditions that will foster price stability.</blockquote>
      By unanimous vote, the minutes were approved.
      <P><a href="#pagetop">Return to top</a></P>
      <P>More text after a section break.</P>
      </td></tr></table>
      <p><a href="#pagetop">Return to top</a></p><p>Home | Monetary policy</p>
    </div></body></html>"""
    assert parse_minutes(html).text.split("\n\n") == [
        "The Committee approved the directive.",
        "The Federal Open Market Committee seeks monetary and financial conditions that will "
        "foster price stability.",
        "By unanimous vote, the minutes were approved.",
        "More text after a section break.",
    ]


def test_modern_minutes_keep_footnotes_without_their_links() -> None:
    parsed = parse_minutes((FIXTURES / "minutes_modern_2026.htm").read_bytes())
    paragraphs = parsed.text.split("\n\n")
    assert paragraphs[3] == "Developments in Financial Markets and Open Market Operations"
    assert paragraphs[4].startswith("The manager started with an overview")
    assert paragraphs[-1] == "1. Attended Tuesday's session only."
    assert "Board of Governors of the Federal Reserve System" not in paragraphs  # footer
