"""Daily market series from FRED and their point-in-time join with document releases.

Series are fetched from the keyless ``fredgraph.csv`` endpoint through the cached
:class:`~text_signals.fetch.Fetcher`, so every download is in the provenance manifest.

The join rule (``event_window``) is the one place where a release becomes an event date:

- the **event day** is the first day with an observation that reflects the release: the release
  date itself if the release came before the series' daily fixing time, otherwise the next day
  with an observation (weekends, holidays and after-close releases roll forward);
- the **pre-event day** is the last day with an observation strictly before the event day;
- the change is ``value(event day) - value(pre-event day)``.

A release whose time of day is unknown is assumed to come before the fixing. For FOMC statements
that is the documented practice (2:15 p.m. or 2:00 p.m. ET for scheduled meetings); the pages
that state no time are listed in the protocol.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from datetime import date, time

import pandas as pd
from pydantic import BaseModel

from text_signals.fetch import Fetcher

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# Treasury constant-maturity yields (DGS*) are read from closing bid quotes taken by the New
# York Fed at about 3:30 p.m. ET. A release at or after that time shows up the next day.
TREASURY_FIXING_ET = time(15, 30)


def parse_fred_csv(data: bytes, series_id: str) -> pd.Series:
    """A FRED ``fredgraph.csv`` file as a float Series indexed by ``date``.

    Days on which the market was closed are present with an empty value (or "." in older
    files); they are kept as NaN so that a missing day is distinguishable from a weekend.
    """
    frame = pd.read_csv(io.BytesIO(data), na_values=["."], dtype={series_id: "float64"})
    date_col = frame.columns[0]
    if list(frame.columns) != [date_col, series_id]:
        raise ValueError(f"unexpected FRED columns {list(frame.columns)} for {series_id}")
    series = pd.Series(
        frame[series_id].to_numpy(),
        index=pd.to_datetime(frame[date_col]).dt.date,
        name=series_id,
    )
    series.index.name = "date"
    if not series.index.is_monotonic_increasing or series.index.has_duplicates:
        raise ValueError(f"{series_id}: dates are not strictly increasing")
    return series


def fetch_fred_series(fetcher: Fetcher, series_id: str, *, refresh: bool = False) -> pd.Series:
    body, _ = fetcher.get(FRED_CSV_URL.format(series_id=series_id), refresh=refresh)
    return parse_fred_csv(body, series_id)


class EventWindow(BaseModel):
    release_date: date
    release_time_et: time | None
    event_day: date
    pre_day: date
    value_pre: float
    value_event: float

    @property
    def change(self) -> float:
        return self.value_event - self.value_pre


def event_window(
    series: pd.Series,
    release_date: date,
    release_time_et: time | None,
    *,
    fixing_et: time = TREASURY_FIXING_ET,
) -> EventWindow | None:
    """Pre-event and event-day observations for one release; None if the series does not cover it.

    ``series`` is indexed by ``datetime.date`` and may contain NaN for closed days.
    """
    observed = series.dropna()
    days: list[date] = list(observed.index)
    same_day_counts = release_time_et is None or release_time_et < fixing_et
    event_day = next(
        (d for d in days if d > release_date or (d == release_date and same_day_counts)),
        None,
    )
    if event_day is None:
        return None
    before = [d for d in days if d < event_day]
    if not before:
        return None
    pre_day = before[-1]
    return EventWindow(
        release_date=release_date,
        release_time_et=release_time_et,
        event_day=event_day,
        pre_day=pre_day,
        value_pre=float(observed[pre_day]),
        value_event=float(observed[event_day]),
    )


def event_changes(
    series: pd.Series,
    releases: Iterable[tuple[str, date, time | None]],
    *,
    fixing_et: time = TREASURY_FIXING_ET,
) -> pd.DataFrame:
    """One row per ``(doc_id, release_date, release_time_et)`` with the event-window change.

    Releases outside the series' coverage get NaN rather than being dropped silently.
    """
    rows = []
    for doc_id, release_date, release_time in releases:
        w = event_window(series, release_date, release_time, fixing_et=fixing_et)
        rows.append(
            {
                "doc_id": doc_id,
                "release_date": release_date,
                "event_day": w.event_day if w else None,
                "pre_day": w.pre_day if w else None,
                f"{series.name}_pre": w.value_pre if w else float("nan"),
                f"{series.name}_event": w.value_event if w else float("nan"),
                f"d_{series.name}": w.change if w else float("nan"),
            }
        )
    return pd.DataFrame(rows)
