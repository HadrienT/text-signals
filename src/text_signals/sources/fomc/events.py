"""Join FOMC statements with Treasury yields (point-in-time) and the LM baseline score.

Output: ``data/processed/fomc/statement_events.csv``, one row per statement, with the release,
the event and pre-event days, the yield levels and changes (in basis points) and the LM score.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from text_signals.documents import Document
from text_signals.fetch import Fetcher
from text_signals.lexicon import LMDictionary, fetch_lm_dictionary, lm_score
from text_signals.market import event_changes, fetch_fred_series

log = logging.getLogger(__name__)

SERIES = ("DGS2", "DGS10")


def build_events(
    docs: Sequence[Document], yields: dict[str, pd.Series], lm: LMDictionary
) -> pd.DataFrame:
    releases = [(d.doc_id, d.release_date, d.release_time_et) for d in docs]
    frame = pd.DataFrame(
        {
            "doc_id": [d.doc_id for d in docs],
            "release_date": [d.release_date for d in docs],
            "release_time_et": [d.release_time_et for d in docs],
            "event_date": [d.event_date for d in docs],
            "unscheduled": [d.unscheduled for d in docs],
        }
    )
    for series_id in SERIES:
        changes = event_changes(yields[series_id], releases)
        cols = [f"{series_id}_pre", f"{series_id}_event", f"d_{series_id}"]
        if series_id == SERIES[0]:
            cols = ["event_day", "pre_day", *cols]
        frame = frame.merge(changes[["doc_id", *cols]], on="doc_id", validate="one_to_one")
    # The two series share their trading calendar, so one event window serves both; check it.
    for series_id in SERIES[1:]:
        other = event_changes(yields[series_id], releases)
        if not other["event_day"].equals(frame["event_day"]):
            raise ValueError(f"{series_id} and {SERIES[0]} disagree on event days")
    for series_id in SERIES:
        frame[f"d_{series_id}_bp"] = 100 * frame.pop(f"d_{series_id}")
    frame["d_slope_2s10s_bp"] = frame["d_DGS10_bp"] - frame["d_DGS2_bp"]

    scores = [lm_score(d.text, lm) for d in docs]
    frame["lm_n_tokens"] = [s.n_tokens for s in scores]
    frame["lm_net_negative"] = [s.net_negative for s in scores]
    frame["lm_uncertainty"] = [s.uncertainty for s in scores]
    frame["lm_version"] = lm.version
    return frame


def run(
    docs: Sequence[Document], data_dir: Path, *, delay_s: float = 2.0, refresh: bool = False
) -> pd.DataFrame:
    with Fetcher(data_dir / "raw", delay_s=delay_s) as fetcher:
        yields = {s: fetch_fred_series(fetcher, s, refresh=refresh) for s in SERIES}
        lm = fetch_lm_dictionary(fetcher)
    frame = build_events(docs, yields, lm)
    out = data_dir / "processed" / "fomc" / "statement_events.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    missing = frame["d_DGS2_bp"].isna().sum()
    log.info("events: %d statements, %d without a yield change", len(frame), missing)
    return frame
