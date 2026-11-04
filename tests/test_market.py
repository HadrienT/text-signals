from datetime import date, time

import pandas as pd
import pytest

from text_signals.market import event_changes, event_window, parse_fred_csv

CSV = b"""observation_date,DGS2
2020-03-12,0.50
2020-03-13,0.49
2020-03-16,0.36
2020-03-17,0.47
2020-03-18,
2020-03-19,0.44
"""


@pytest.fixture
def dgs2() -> pd.Series:
    return parse_fred_csv(CSV, "DGS2")


def test_parse_fred_csv_keeps_closed_days_as_nan(dgs2: pd.Series) -> None:
    assert dgs2.index[0] == date(2020, 3, 12)
    assert dgs2.isna().sum() == 1 and pd.isna(dgs2[date(2020, 3, 18)])


def test_parse_fred_csv_rejects_other_columns() -> None:
    with pytest.raises(ValueError):
        parse_fred_csv(b"observation_date,DGS10\n2020-01-02,1.0\n", "DGS2")


def test_release_before_fixing_counts_the_same_day(dgs2: pd.Series) -> None:
    w = event_window(dgs2, date(2020, 3, 16), time(14, 0))
    assert w is not None
    assert (w.pre_day, w.event_day) == (date(2020, 3, 13), date(2020, 3, 16))
    assert w.change == pytest.approx(0.36 - 0.49)


def test_unknown_time_is_assumed_before_fixing(dgs2: pd.Series) -> None:
    w = event_window(dgs2, date(2020, 3, 17), None)
    assert w is not None and w.event_day == date(2020, 3, 17)


def test_release_after_fixing_rolls_to_next_day(dgs2: pd.Series) -> None:
    w = event_window(dgs2, date(2020, 3, 13), time(17, 0))
    assert w is not None
    assert (w.pre_day, w.event_day) == (date(2020, 3, 13), date(2020, 3, 16))


def test_sunday_release_rolls_to_monday(dgs2: pd.Series) -> None:
    # The FOMC statement of Sunday 15 March 2020, 5:00 p.m. EDT.
    w = event_window(dgs2, date(2020, 3, 15), time(17, 0))
    assert w is not None
    assert (w.pre_day, w.event_day) == (date(2020, 3, 13), date(2020, 3, 16))


def test_closed_days_are_skipped_on_both_sides(dgs2: pd.Series) -> None:
    # Release on a day the bond market is closed: event = next open day, pre = last open day.
    w = event_window(dgs2, date(2020, 3, 18), time(14, 0))
    assert w is not None
    assert (w.pre_day, w.event_day) == (date(2020, 3, 17), date(2020, 3, 19))
    # Release the day after a closed day: the pre-event day skips the closed day.
    w = event_window(dgs2, date(2020, 3, 19), time(14, 0))
    assert w is not None and w.pre_day == date(2020, 3, 17)


def test_out_of_coverage_is_nan_not_dropped(dgs2: pd.Series) -> None:
    frame = event_changes(
        dgs2,
        [("before", date(2020, 3, 12), None), ("ok", date(2020, 3, 17), None),
         ("after", date(2020, 3, 20), None)],
    )  # fmt: skip
    assert list(frame["doc_id"]) == ["before", "ok", "after"]
    assert frame["d_DGS2"].isna().tolist() == [True, False, True]
