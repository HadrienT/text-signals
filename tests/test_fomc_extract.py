import json
from datetime import date
from pathlib import Path

import pandas as pd

from text_signals.sources.fomc.extract import SampleDoc, import_reference, write_blind_inputs
from text_signals.sources.fomc.extract_eval import (
    decision_from_change,
    policy_target,
    target_change_bp,
)
from text_signals.sources.fomc.prompts import STATEMENT_V2

TEXT = "The Committee decided today to keep its target. Inflation remains elevated."
EMPTY = {"value": None, "evidence": None}
VALID = {
    "policy_action": {"value": "hold", "change_bp": 0, "evidence": "decided today to keep"},
    "forward_guidance": EMPTY,
    "inflation_assessment": {"value": 1, "evidence": "Inflation remains elevated."},
    "labor_market_assessment": EMPTY,
    "growth_assessment": EMPTY,
    "balance_sheet": EMPTY,
    "risk_balance": EMPTY,
    "dissents": [],
}


def _series(values: dict[str, float]) -> pd.Series:
    return pd.Series(list(values.values()), index=[date.fromisoformat(d) for d in values])


def test_target_is_spliced_at_the_range_start() -> None:
    single = _series({"2008-12-15": 1.0, "2008-12-16": 0.25})  # FRED ends DFEDTAR on 12-15
    upper = _series({"2008-12-16": 0.25, "2008-12-17": 0.25})
    target = policy_target(single, upper)
    assert list(target.index) == [date(2008, 12, 15), date(2008, 12, 16), date(2008, 12, 17)]
    assert target_change_bp([date(2008, 12, 16)], target) == [-75]


def test_change_absorbs_a_one_day_lag_in_the_effective_date() -> None:
    target = _series(
        {"2015-12-15": 0.25, "2015-12-16": 0.25, "2015-12-17": 0.50, "2015-12-18": 0.50}
    )
    assert target_change_bp([date(2015, 12, 16)], target) == [25]
    assert target_change_bp([date(2015, 12, 18)], target) == [0]
    assert target_change_bp([date(2015, 12, 15)], target) == [None]  # nothing before
    assert [decision_from_change(c) for c in (25, -50, 0, None)] == ["raise", "cut", "hold", None]


def test_blind_inputs_hold_the_text_only(tmp_path: Path) -> None:
    doc = SampleDoc("S01", "fomc-statement-20170920", date(2017, 9, 20), TEXT)
    write_blind_inputs([doc], tmp_path)
    assert (tmp_path / "S01.txt").read_text() == TEXT + "\n"
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {"S01.txt", "PROMPT.txt"}
    blob = "".join(p.read_text() for p in tmp_path.iterdir())
    assert "2017" not in blob and "fomc-statement-2017" not in blob
    assert STATEMENT_V2.version in blob


def test_import_reference_validates_and_records(tmp_path: Path) -> None:
    docs = [
        SampleDoc("S01", "d1", date(2017, 9, 20), TEXT),
        SampleDoc("S02", "d2", date(2018, 1, 31), TEXT),
        SampleDoc("S03", "d3", date(2018, 3, 21), TEXT),
    ]
    (tmp_path / "S01.json").write_text(json.dumps(VALID))
    bad = dict(VALID, inflation_assessment={"value": 1, "evidence": None})
    (tmp_path / "S02.json").write_text(json.dumps(bad))
    records = import_reference(tmp_path, docs, "claude-test", out=tmp_path / "out")
    assert [r.doc_id for r in records] == ["d1", "d2"]  # S03 has no file
    assert records[0].output is not None and records[0].error is None
    assert records[1].output is None and records[1].error is not None
    assert records[0].decoding is None and records[0].prompt_version == STATEMENT_V2.version
    assert (tmp_path / "out" / "claude-test.jsonl").read_text().count("\n") == 2


def test_confirmation_sample_avoids_the_evaluation_sample_and_is_fixed() -> None:
    from text_signals.sources.fomc.extract import N_CONFIRM, draw_confirmation

    ids = [f"fomc-statement-2000{m:02d}{d:02d}" for m in range(1, 13) for d in (1, 15, 28)]
    events = pd.DataFrame(
        {
            "doc_id": ids,
            "release_date": [f"2000-{i[-4:-2]}-{i[-2:]}" for i in ids],
            "in_main_sample": [True] * len(ids),
        }
    )
    used = set(ids[:5])
    a = draw_confirmation(events, used)
    b = draw_confirmation(events, used)
    assert len(a) == N_CONFIRM and a.equals(b)
    assert not used & set(a["doc_id"])
    assert list(a["code"][:2]) == ["C01", "C02"]


def test_import_keeps_the_records_of_another_sample(tmp_path: Path) -> None:
    first = [SampleDoc("S01", "d1", date(2017, 9, 20), TEXT)]
    second = [SampleDoc("C01", "d9", date(2018, 1, 31), TEXT)]
    (tmp_path / "S01.json").write_text(json.dumps(VALID))
    (tmp_path / "C01.json").write_text(json.dumps(VALID))
    import_reference(tmp_path, first, "claude-test", out=tmp_path / "out")
    import_reference(tmp_path, second, "claude-test", out=tmp_path / "out")
    import_reference(tmp_path, second, "claude-test", out=tmp_path / "out")  # read again
    lines = (tmp_path / "out" / "claude-test.jsonl").read_text().splitlines()
    assert sorted(json.loads(x)["doc_id"] for x in lines) == ["d1", "d9"]


def test_all_docs_are_in_release_order(tmp_path: Path) -> None:
    from text_signals.sources.fomc.extract import all_docs

    rows = [
        {"doc_id": "fomc-statement-20010103", "release_date": "2001-01-03", "text": "b"},
        {"doc_id": "fomc-statement-19990202", "release_date": "1999-02-02", "text": "a"},
    ]
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    docs = all_docs(path)
    assert [d.code for d in docs] == ["fomc-statement-19990202", "fomc-statement-20010103"]
