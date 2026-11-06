from pathlib import Path

import pandas as pd
import pytest

from text_signals.sources.fomc.labelling import (
    allocate,
    draw_sample,
    speakable,
    write_listening_pdf,
)


def test_allocation_is_proportional_with_a_floor() -> None:
    sizes = {"a": 20, "b": 80, "c": 8, "d": 32, "e": 49, "f": 40, "g": 18}
    quota = allocate(sizes, 60, 4)
    assert sum(quota.values()) == 60
    assert min(quota.values()) >= 4
    assert quota["b"] > quota["e"] > quota["f"] > quota["d"]


def test_allocation_never_exceeds_a_small_cell() -> None:
    assert allocate({"a": 2, "b": 100}, 10, 4) == {"a": 2, "b": 8}


def _events(n: int = 200) -> pd.DataFrame:
    days = pd.date_range("1994-01-01", periods=n, freq="45D")
    return pd.DataFrame(
        {
            "doc_id": [f"fomc-statement-{d:%Y%m%d}" for d in days],
            "release_date": [f"{d:%Y-%m-%d}" for d in days],
            "zlb": [2009 <= d.year <= 2015 for d in days],
            "in_main_sample": [i % 17 != 0 for i in range(n)],
        }
    )


def test_sample_is_deterministic_and_disjoint() -> None:
    plan = draw_sample(_events())
    assert plan.equals(draw_sample(_events()))
    label = plan[plan.role == "label"]
    practice = plan[plan.role == "practice"]
    assert len(label) == 60 and len(practice) == 5
    assert not set(label.doc_id) & set(practice.doc_id)
    assert label.second_pass.sum() == 15 and not practice.second_pass.any()
    assert list(label.code) == [f"S{i:02d}" for i in range(1, 61)]
    # Codes are shuffled: their order is not the chronological order.
    assert list(label.release_date) != sorted(label.release_date)


def test_listening_pdf_has_one_chapter_per_statement(tmp_path: Path) -> None:
    import pymupdf

    plan = draw_sample(_events())
    texts = {
        d: "Inflation remains elevated.\n\n" + "The Committee decided. " * 120 for d in plan.doc_id
    }
    path = tmp_path / "listen.pdf"
    write_listening_pdf(plan, texts, path)
    doc = pymupdf.open(path)
    toc = doc.get_toc()
    assert len(toc) == 65
    assert toc[0][1] == "Practice statement 1" and toc[5][1] == "Statement S01"
    first = doc[0].get_text()
    assert "Inflation remains elevated." in first
    assert "199" not in first and "200" not in first  # no date anywhere


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("to around 5-1/4 percent", "to around 5 and a quarter percent"),
        ("decline 1/4 percentage point", "decline a quarter percentage point"),
        ("at 1/4 to 1/2 percent", "at a quarter to a half percent"),
        ("to 2-1/4 to 2-1/2 percent", "to 2 and a quarter to 2 and a half percent"),
        ("at 3¾ to 4 percent", "at 3 and three quarters to 4 percent"),
        ("on 12/31 and 2024/25", "on 12/31 and 2024/25"),  # not a Fed fraction
        ("to 2‑1/4 to 2‑1/2 percent", "to 2 and a quarter to 2 and a half percent"),  # U+2011
        ("by a 10 – 2 vote", "by a 10 to 2 vote"),
        ("agency mortgage‑backed securities", "agency mortgage-backed securities"),
    ],
)
def test_fractions_are_spoken_in_words(text: str, spoken: str) -> None:
    assert speakable(text) == spoken
