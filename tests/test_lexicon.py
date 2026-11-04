import math

from text_signals.lexicon import lm_score, parse_lm_csv, tokenize

HEADER = (
    "Word,Seq_num,Word Count,Word Proportion,Average Proportion,Std Dev,Doc Count,Negative,"
    "Positive,Uncertainty,Litigious,Strong_Modal,Weak_Modal,Constraining,Complexity,Syllables,"
    "Source\n"
)


def _row(word: str, neg: int = 0, pos: int = 0, unc: int = 0) -> str:
    return f"{word},1,1,0,0,0,1,{neg},{pos},{unc},0,0,0,0,0,1,12of12inf\n"


CSV = (
    HEADER
    + _row("DECLINE", neg=2009)
    + _row("WEAK", neg=2009)
    + _row("STRONG", pos=2009)
    + _row("UNCERTAIN", unc=2009)
    + _row("CLOSED", neg=-2020)  # removed from the negative list in 2020
    + _row("THE")
).encode()


def test_parse_lm_csv_keeps_only_positive_years() -> None:
    lm = parse_lm_csv(CSV, sha256="x")
    assert lm.lists["Negative"] == {"DECLINE", "WEAK"}
    assert lm.lists["Positive"] == {"STRONG"}
    assert "CLOSED" in lm.words and "CLOSED" not in lm.lists["Negative"]


def test_tokenize() -> None:
    assert tokenize("Rates rose 25 basis-points; inflation's up.") == [
        "RATES", "ROSE", "BASIS", "POINTS", "INFLATION", "S", "UP",
    ]  # fmt: skip


def test_lm_score() -> None:
    lm = parse_lm_csv(CSV, sha256="x")
    s = lm_score("Growth is weak and the decline is uncertain, but jobs are strong. Closed.", lm)
    assert s.n_tokens == 13
    assert (s.n_negative, s.n_positive, s.n_uncertainty) == (2, 1, 1)
    assert math.isclose(s.net_negative, (2 - 1) / 13)
    assert math.isclose(s.uncertainty, 1 / 13)
    assert math.isnan(lm_score("", lm).net_negative)
