"""The San Francisco Fed's U.S. Monetary Policy Event-Study Database (USMPD).

USMPD gives, for each FOMC communication event, the change of interest rates and asset prices
over a narrow intraday window, computed by the authors from LSEG tick data (Acosta, Ajello,
Bauer, Loria and Miranda-Agrippino, 2025, FRBSF Working Paper 2025-30). We cannot rebuild these
changes from ticks; we take them as published, pin the file in the provenance manifest and
record the database's own "Last update" date as its version. See the protocol, §3.4.

Units, as stated in the file: rate changes (MP1 to TIPS30Y) in percentage points, stock and
dollar returns (SP500 to USDJPY) in percent. Timestamps are in US Eastern Time and are kept as
naive ``datetime`` values.

The windows run from 10 minutes before the event to 20 minutes after it (statements, minutes),
or 60 minutes after the start of the press conference (press conferences, monetary events).
"""

from __future__ import annotations

import calendar
import io
import re
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from text_signals.fetch import Fetcher, FetchRecord

USMPD_URL = "https://www.frbsf.org/wp-content/uploads/USMPD.xlsx"

RATE_COLUMNS = (
    "MP1",
    "MP2",
    *(f"FF{i}" for i in range(1, 7)),
    *(f"ED{i}" for i in range(1, 9)),
    "OIS1Y",
    "OIS2Y",
    "UST3M",
    "UST6M",
    "UST2Y",
    "UST5Y",
    "UST10Y",
    "UST30Y",
    "TIPS5Y",
    "TIPS10Y",
    "TIPS30Y",
)
RETURN_COLUMNS = ("SP500", "SPFUT", "DXY", "EURUSD", "USDJPY")

# Sheet name -> the indicator columns between ``date_time`` and the market columns.
SHEET_FLAGS: dict[str, tuple[str, ...]] = {
    "Statements": ("SEP", "Unscheduled"),
    "Press Conferences": (),
    "Monetary Events": ("SEP", "Unscheduled", "PC"),
    "Minutes": (),
}

# A meeting in the last days of the month moves the current-month contract too little to be
# scaled up reliably; the next month's contract is used instead (Gürkaynak, Sack and Swanson,
# 2005). The threshold reproduces every MP1 value in USMPD (protocol §3.4).
MP1_LAST_DAYS = 7

_LAST_UPDATE = re.compile(r"Last update:\s*(\d{1,2} \w+ \d{4})")


@dataclass(frozen=True)
class USMPD:
    """The parsed database: its version date and one frame per sheet, keyed by sheet name."""

    version: date
    sheets: dict[str, pd.DataFrame]

    @property
    def statements(self) -> pd.DataFrame:
        return self.sheets["Statements"]

    @property
    def minutes(self) -> pd.DataFrame:
        return self.sheets["Minutes"]


def _parse_version(readme: pd.DataFrame) -> date:
    for value in readme.to_numpy().ravel():
        if isinstance(value, str) and (m := _LAST_UPDATE.search(value)):
            return datetime.strptime(m.group(1), "%d %B %Y").date()
    raise ValueError("USMPD README: no 'Last update' line")


def _parse_sheet(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    expected = ["Date", "date_time", *SHEET_FLAGS[name], *RATE_COLUMNS, *RETURN_COLUMNS]
    if list(frame.columns) != expected:
        raise ValueError(f"USMPD sheet {name!r}: unexpected columns {list(frame.columns)}")
    out = frame.copy()
    stamps = pd.to_datetime(out["date_time"])
    out["date_time"] = [t.to_pydatetime() for t in stamps]
    out["Date"] = pd.to_datetime(out["Date"]).dt.date
    if not (stamps.dt.date == out["Date"]).all():
        raise ValueError(f"USMPD sheet {name!r}: Date and date_time disagree")
    if out["Date"].duplicated().any() or not out["Date"].is_monotonic_increasing:
        raise ValueError(f"USMPD sheet {name!r}: dates are not strictly increasing")
    for flag in SHEET_FLAGS[name]:
        out[flag] = out[flag].astype(bool)
    for col in (*RATE_COLUMNS, *RETURN_COLUMNS):
        out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
    return out.reset_index(drop=True)


def parse_usmpd(data: bytes) -> USMPD:
    """Parse ``USMPD.xlsx``; raise ``ValueError`` if its layout differs from the one we know."""
    book = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None)
    missing = {"README", *SHEET_FLAGS} - set(book)
    if missing:
        raise ValueError(f"USMPD: missing sheets {sorted(missing)}")
    version = _parse_version(book["README"])
    sheets = {}
    for name in SHEET_FLAGS:
        raw = book[name]
        frame = raw.iloc[1:].set_axis([str(c) for c in raw.iloc[0]], axis=1)
        sheets[name] = _parse_sheet(frame, name)
    return USMPD(version=version, sheets=sheets)


def fetch_usmpd(fetcher: Fetcher, *, refresh: bool = False) -> tuple[USMPD, FetchRecord]:
    """The cached database. The file is revised after each meeting: ``refresh`` refetches it."""
    body, rec = fetcher.get(USMPD_URL, refresh=refresh)
    return parse_usmpd(body), rec


def kuttner_mp1(event_date: date, ff1: float, ff2: float) -> float:
    """Surprise in the current decision from fed funds futures (Kuttner, 2001).

    The current-month contract settles on the month's average rate, so a change decided on day
    ``d`` of a ``D``-day month moves it by only ``(D - d) / D`` of the surprise:
    ``MP1 = FF1 * D / (D - d)``. In the last ``MP1_LAST_DAYS`` days of the month the
    next-month contract is used instead: ``MP1 = FF2``. Inputs and output in the same unit.
    """
    days = calendar.monthrange(event_date.year, event_date.month)[1]
    remaining = days - event_date.day
    if remaining <= MP1_LAST_DAYS:
        return ff2
    return ff1 * days / remaining


def recompute_mp1(sheet: pd.DataFrame) -> pd.Series:
    """MP1 recomputed from FF1 and FF2 for every row of a USMPD sheet."""
    values = [
        kuttner_mp1(d, f1, f2)
        for d, f1, f2 in zip(sheet["Date"], sheet["FF1"], sheet["FF2"], strict=True)
    ]
    return pd.Series(values, index=sheet.index, name="MP1_recomputed")


def statement_changes(db: USMPD) -> pd.DataFrame:
    """One row per USMPD statement event, keyed by ``release_date``, in basis points.

    Columns: the event timestamp (ET), the 30-minute changes of the 2- and 10-year yields and of
    the 2s10s slope, MP1, MP1 recomputed by us, MP2, and the database version.
    """
    s = db.statements
    out = pd.DataFrame(
        {
            "release_date": s["Date"],
            "usmpd_date_time": s["date_time"],
            "usmpd_unscheduled": s["Unscheduled"],
            "d_UST2Y_30m_bp": 100 * s["UST2Y"],
            "d_UST10Y_30m_bp": 100 * s["UST10Y"],
            "MP1_bp": 100 * s["MP1"],
            "MP1_recomputed_bp": 100 * recompute_mp1(s),
            "MP2_bp": 100 * s["MP2"],
        }
    )
    out["d_slope_2s10s_30m_bp"] = out["d_UST10Y_30m_bp"] - out["d_UST2Y_30m_bp"]
    out["usmpd_version"] = db.version
    return out
