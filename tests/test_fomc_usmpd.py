import io
from datetime import date, datetime

import pandas as pd
import pytest

from text_signals.sources.fomc.events import flag_main_sample, join_usmpd
from text_signals.sources.fomc.usmpd import (
    RATE_COLUMNS,
    RETURN_COLUMNS,
    SHEET_FLAGS,
    kuttner_mp1,
    parse_usmpd,
    statement_changes,
)

# Rows copied from USMPD.xlsx, "Statements" sheet, version of 30 October 2026
# (sha256 2b91297c…): date, FF1, FF2 and the published MP1, in percentage points.
REAL_ROWS = [
    (date(1994, 2, 4), 0.14, 0.10001, 0.163333),  # 28-day month: 0.14 * 28 / 24
    (date(2026, 3, 18), 0.0025, 0.0, 0.005962),  # 0.0025 * 31 / 13
    (date(2026, 9, 16), 0.01375, 0.025, 0.029464),  # 0.01375 * 30 / 14
    (date(2026, 7, 29), -0.005, -0.065, -0.065),  # 2 days left: next-month contract
    (date(2010, 6, 23), 0.0, 0.005, 0.005),  # exactly 7 days left: next-month contract
    (date(1999, 8, 24), 0.015, 0.0375, 0.0375),  # 7 days left
]


@pytest.mark.parametrize(("day", "ff1", "ff2", "mp1"), REAL_ROWS)
def test_kuttner_mp1_reproduces_usmpd(day: date, ff1: float, ff2: float, mp1: float) -> None:
    assert kuttner_mp1(day, ff1, ff2) == pytest.approx(mp1, abs=1e-5)


def _sheet(name: str, rows: list[tuple[datetime, dict[str, float]]]) -> pd.DataFrame:
    cols = ["Date", "date_time", *SHEET_FLAGS[name], *RATE_COLUMNS, *RETURN_COLUMNS]
    records = []
    for stamp, values in rows:
        rec: dict[str, object] = {c: 0.0 for c in cols}
        rec["Date"] = datetime(stamp.year, stamp.month, stamp.day)
        rec["date_time"] = stamp
        for flag in SHEET_FLAGS[name]:
            rec[flag] = 0
        rec.update(values)
        records.append(rec)
    return pd.DataFrame(records, columns=cols)


def _workbook(statements: pd.DataFrame, version: str = "Last update: 30 October 2026") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame([[None, "USMPD"], [None, version]]).to_excel(
            writer, sheet_name="README", header=False, index=False
        )
        statements.to_excel(writer, sheet_name="Statements", index=False)
        for name in ("Press Conferences", "Monetary Events", "Minutes"):
            _sheet(name, []).to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()


STATEMENTS = _sheet(
    "Statements",
    [
        (
            datetime(2026, 9, 16, 14, 0),
            {
                "FF1": 0.01375,
                "FF2": 0.025,
                "MP1": 0.029464,
                "MP2": 0.02,
                "UST2Y": 0.055,
                "UST10Y": 0.008,
            },
        ),
        (datetime(2026, 10, 28, 14, 0), {"FF2": -0.015, "MP1": -0.015, "UST2Y": 0.01}),
    ],
)


def test_parse_usmpd_reads_version_and_timestamps() -> None:
    db = parse_usmpd(_workbook(STATEMENTS))
    assert db.version == date(2026, 10, 30)
    s = db.statements
    assert list(s["Date"]) == [date(2026, 9, 16), date(2026, 10, 28)]
    assert s.loc[0, "date_time"] == datetime(2026, 9, 16, 14, 0)
    assert s["Unscheduled"].dtype == bool


def test_parse_usmpd_rejects_a_changed_layout() -> None:
    with pytest.raises(ValueError, match="unexpected columns"):
        parse_usmpd(_workbook(STATEMENTS.drop(columns=["UST30Y"])))


def test_parse_usmpd_requires_a_version() -> None:
    with pytest.raises(ValueError, match="Last update"):
        parse_usmpd(_workbook(STATEMENTS, version="no date here"))


def test_statement_changes_are_in_basis_points() -> None:
    changes = statement_changes(parse_usmpd(_workbook(STATEMENTS)))
    first = changes.iloc[0]
    assert first["d_UST2Y_30m_bp"] == pytest.approx(5.5)
    assert first["d_slope_2s10s_30m_bp"] == pytest.approx(0.8 - 5.5)
    assert first["MP1_bp"] == pytest.approx(first["MP1_recomputed_bp"], abs=1e-3)
    assert first["usmpd_version"] == date(2026, 10, 30)


def test_join_usmpd_matches_on_release_date_only() -> None:
    changes = statement_changes(parse_usmpd(_workbook(STATEMENTS)))
    events = pd.DataFrame(
        {
            "doc_id": ["fomc-statement-20260916", "fomc-statement-20261029"],
            "release_date": [date(2026, 9, 16), date(2026, 10, 29)],
        }
    )
    joined = join_usmpd(events, changes)
    assert joined.loc[0, "d_UST2Y_30m_bp"] == pytest.approx(5.5)
    # A neighbouring USMPD date (28 October) is not matched to a 29 October release.
    assert joined["d_UST2Y_30m_bp"].isna().tolist() == [False, True]


def test_main_sample_needs_usmpd_and_no_chair_speech() -> None:
    events = pd.DataFrame(
        {
            "doc_id": [
                "fomc-statement-20260916",  # matched, clean
                "fomc-statement-20191011",  # no USMPD row
                "fomc-statement-20250822",  # released with Powell's Jackson Hole speech
            ],
            "d_UST2Y_30m_bp": [5.5, float("nan"), -3.0],
        }
    )
    flagged = flag_main_sample(events)
    assert flagged["confounded_chair_speech"].tolist() == [False, False, True]
    assert flagged["in_main_sample"].tolist() == [True, False, False]
