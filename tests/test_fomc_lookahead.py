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
    vote = (
        "Voting for the action were: Ben S. Bernanke, Chairman; Sarah Bloom Raskin; "
        "and Jeremy C. Stein."
    )
    out = anonymise(vote, member_names([vote]))
    assert "Raskin" not in out and "Bloom" not in out and "Stein" not in out


def test_cutoff_split_finds_a_change_in_the_effect() -> None:
    import numpy as np
    import pandas as pd

    from text_signals.sources.fomc.lookahead import cutoff_split

    rng = np.random.default_rng(9)
    n = 200
    after = np.arange(n) >= 150
    tone = rng.normal(0, 1, n)
    table = pd.DataFrame(
        {
            "doc_id": [f"d{i:03d}" for i in range(n)],
            "release_date": pd.date_range("1994-01-01", periods=n, freq="45D").astype(str),
            "llm_tone": tone,
            "MP1_bp": rng.normal(0, 5, n),
            "lm_net_negative": rng.normal(0, 0.01, n),
            "finbert_net_negative": rng.normal(0, 0.2, n),
            "in_main_sample": True,
            "llm_after_cutoff": after,
        }
    )
    table["d_UST2Y_30m_bp"] = np.where(after, 4.0, 0.0) * tone + rng.normal(0, 1, n)
    r = cutoff_split(table)
    assert r["n_after"] == 50 and r["interaction_p"] < 0.01
    assert r["beta_after_per_sd"] > r["beta_before_per_sd"] + 2
