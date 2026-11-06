from datetime import date

from text_signals.chrono_gpt import ANCHORS, CONTINUATION, FALL, RISE
from text_signals.sources.fomc.chrono_scores import assignments, vintage_for
from text_signals.sources.fomc.extract import SampleDoc


def test_each_statement_is_read_by_the_vintage_of_the_year_before() -> None:
    assert vintage_for(1999) is None and vintage_for(1994) is None
    assert vintage_for(2000) == 1999
    assert vintage_for(2024) == 2023
    assert vintage_for(2025) == 2024 and vintage_for(2026) == 2024  # newest available
    docs = [SampleDoc(c, c, date(y, 6, 1), "x") for c, y in [("a", 1999), ("b", 2008), ("c", 2026)]]
    plan = assignments(docs)
    assert {v: [d.code for d in ds] for v, ds in plan.items()} == {2007: ["b"], 2024: ["c"]}


def test_scoring_constants_are_the_ones_specified() -> None:
    import tiktoken

    e = tiktoken.get_encoding("gpt2")
    assert e.encode(" rise") == [RISE] and e.encode(" fall") == [FALL]
    assert CONTINUATION == " As a result, short-term interest rates are likely to"
    assert len(ANCHORS) == 5
