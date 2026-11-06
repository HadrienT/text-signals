from text_signals.sources.fomc.lookahead import anonymise, member_names

VOTE = (
    "Voting for the FOMC monetary policy action were: Alan Greenspan, Chairman; "
    "William J. McDonough, Vice Chairman; Roger W. Ferguson, Jr.; and Gary H. Stern. "
    "Voting against was Jeffrey M. Lacker, who preferred an increase."
)


def test_member_names_come_from_vote_paragraphs_and_chair_phrases() -> None:
    names = member_names([VOTE, "Chairman Alan Greenspan announced today that rates rise."])
    assert {
        "Alan Greenspan",
        "William J. McDonough",
        "Roger W. Ferguson",
        "Gary H. Stern",
        "Jeffrey M. Lacker",
    } <= names
    assert "Vice Chairman" not in names


def test_anonymise_removes_dates_years_and_names_but_not_policy_wording() -> None:
    names = member_names([VOTE])
    text = (
        "Information received since the Committee met in December suggests growth. "
        "The Committee expects to keep the target range at 0 to 1/4 percent through mid-2015 "
        "and will meet on January 28, 2015. Inflation may run below 2 percent. Mr. Lacker "
        "dissented. In May the Committee acted. " + VOTE
    )
    out = anonymise(text, names)
    for gone in ("December", "2015", "January 28", "Greenspan", "Lacker", "McDonough", "Jr."):
        assert gone not in out
    assert "[month]" in out and "[date]" in out and "[year]" in out and "[member]" in out
    assert "0 to 1/4 percent" in out and "below 2 percent" in out
    assert "may run" in out  # the modal verb is not a month
    assert "In [month] the Committee" in out or "In May the Committee" not in out


def test_three_part_names_are_removed_whole() -> None:
    vote = "Voting for the action were: Ben S. Bernanke, Chairman; Sarah Bloom Raskin; and Jeremy C. Stein."
    out = anonymise(vote, member_names([vote]))
    assert "Raskin" not in out and "Bloom" not in out and "Stein" not in out
