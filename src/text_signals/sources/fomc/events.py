"""Join FOMC statements with Treasury yields (point-in-time) and the LM baseline score.

Output: ``data/processed/fomc/statement_events.csv``, one row per statement, with the release,
the event and pre-event days, the daily yield levels and changes (in basis points), the USMPD
30-minute changes and policy surprises (the main window and control, protocol §3.4), and the LM
and FinBERT baseline scores (protocol §5.2).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from text_signals.documents import Document
from text_signals.encoders import (
    FINBERT,
    EncoderScore,
    load_hf_classifier,
    provenance,
    score_documents,
)
from text_signals.fetch import Fetcher
from text_signals.lexicon import LMDictionary, fetch_lm_dictionary, lm_score
from text_signals.market import event_changes, fetch_fred_series
from text_signals.sources.fomc.usmpd import fetch_usmpd, statement_changes

log = logging.getLogger(__name__)

# DGS3MO is the robustness control for the policy surprise (protocol §6.2).
SERIES = ("DGS2", "DGS10", "DGS3MO")

# Upper bound of the federal funds target range, daily from 2008-12-16 (FRED). A statement is at
# the zero lower bound when the range in force on the day before its release is 0 to 0.25 percent.
POLICY_UPPER_SERIES = "DFEDTARU"
ZLB_UPPER_BOUND = 0.25

# Statements released at the same minute as a speech by the Chair: no window, daily or intraday,
# separates the two, and the speech carried the policy news. Excluded from the main sample and
# kept, flagged, for the robustness checks (protocol §3.4, decided 2026-11-05). Sources: the
# Fed's press release and speech indexes (federalreserve.gov/json/ne-press.json, ne-speeches.json).
CONFOUNDED_WITH_CHAIR_SPEECH = {
    "fomc-statement-20200827": "Powell, Jackson Hole, 9:10 AM (powell20200827a.htm)",
    "fomc-statement-20250822": "Powell, Jackson Hole, 10:00 AM (powell20250822a.htm)",
}


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
    # The series share their trading calendar, so one event window serves all; check it.
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


def zlb_flags(release_dates: Sequence[date], upper: pd.Series) -> list[bool]:
    """Whether each release came while the target range was at the zero lower bound.

    The range in force is the last ``upper`` observation strictly before the release date, so
    the statement that moved to the zero lower bound (16 December 2008, 15 March 2020) is not
    flagged and the one that lifted off (16 December 2015, 16 March 2022) is. Before the first
    observation of ``upper`` (a single target, not a range, until 2008) the flag is False.
    """
    observed = upper.dropna()
    flags = []
    for day in release_dates:
        before = observed[observed.index < day]
        flags.append(bool(len(before)) and abs(float(before.iloc[-1]) - ZLB_UPPER_BOUND) < 1e-9)
    return flags


def finbert_columns(scores: Sequence[EncoderScore]) -> pd.DataFrame:
    """FinBERT baseline per document (protocol §5.2, fixed 2026-11-05, before any regression).

    ``finbert_net_negative`` is the mean over sentences of P(negative) − P(positive), so that
    its sign matches ``lm_net_negative``.
    """
    return pd.DataFrame(
        {
            "doc_id": [s.doc_id for s in scores],
            "finbert_n_sentences": [s.n_sentences for s in scores],
            "finbert_n_truncated": [s.n_truncated for s in scores],
            "finbert_positive": [s.mean_probs["positive"] for s in scores],
            "finbert_negative": [s.mean_probs["negative"] for s in scores],
            "finbert_neutral": [s.mean_probs["neutral"] for s in scores],
            "finbert_net_negative": [
                s.mean_probs["negative"] - s.mean_probs["positive"] for s in scores
            ],
            "finbert_revision": [s.revision for s in scores],
        }
    )


def join_usmpd(events: pd.DataFrame, usmpd: pd.DataFrame) -> pd.DataFrame:
    """Add the USMPD statement columns by release date; statements without a row get NaN.

    The match is on the date only: USMPD keys an event by its own timestamp, and a statement
    whose USMPD date differs from our release date is left unmatched on purpose, to be resolved
    by hand and logged in the protocol, never matched to a nearby date.
    """
    return events.merge(usmpd, on="release_date", how="left", validate="many_to_one")


def flag_main_sample(events: pd.DataFrame) -> pd.DataFrame:
    """Add ``confounded_chair_speech`` and ``in_main_sample`` (protocol §3.4 and §6.3).

    A statement is in the main sample when it has a USMPD 30-minute change and was not released
    together with a speech by the Chair. The others stay in the table for the daily robustness
    check.
    """
    out = events.copy()
    out["confounded_chair_speech"] = out["doc_id"].isin(list(CONFOUNDED_WITH_CHAIR_SPEECH))
    out["in_main_sample"] = out["d_UST2Y_30m_bp"].notna() & ~out["confounded_chair_speech"]
    return out


def run(
    docs: Sequence[Document], data_dir: Path, *, delay_s: float = 2.0, refresh: bool = False
) -> pd.DataFrame:
    with Fetcher(data_dir / "raw", delay_s=delay_s) as fetcher:
        yields = {s: fetch_fred_series(fetcher, s, refresh=refresh) for s in SERIES}
        upper = fetch_fred_series(fetcher, POLICY_UPPER_SERIES, refresh=refresh)
        lm = fetch_lm_dictionary(fetcher)
        usmpd, usmpd_rec = fetch_usmpd(fetcher, refresh=refresh)
    changes = statement_changes(usmpd)
    log.info("FinBERT: %s", provenance(FINBERT))
    finbert = score_documents(
        docs,
        FINBERT,
        lambda: load_hf_classifier(FINBERT, data_dir / "raw" / "huggingface"),
        data_dir / "interim" / "fomc" / "encoder_scores" / f"{FINBERT.name}.jsonl",
    )
    mismatch = changes[(changes["MP1_bp"] - changes["MP1_recomputed_bp"]).abs() > 0.01]
    if not mismatch.empty:
        log.warning("MP1 recomputation differs from USMPD on %s", list(mismatch["release_date"]))
    log.info(
        "USMPD version %s (sha256 %s): %d statement events, MP1 recomputed on %d",
        usmpd.version,
        usmpd_rec.sha256[:12],
        len(changes),
        len(changes) - len(mismatch),
    )
    frame = build_events(docs, yields, lm).merge(
        finbert_columns(finbert), on="doc_id", validate="one_to_one"
    )
    frame = flag_main_sample(join_usmpd(frame, changes))
    frame["zlb"] = zlb_flags(list(frame["release_date"]), upper)
    out = data_dir / "processed" / "fomc" / "statement_events.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    missing = frame["d_DGS2_bp"].isna().sum()
    log.info("events: %d statements, %d without a daily yield change", len(frame), missing)
    unmatched = frame.loc[frame["d_UST2Y_30m_bp"].isna(), "doc_id"].tolist()
    log.info("events: %d statements without a USMPD row: %s", len(unmatched), unmatched)
    log.info("events: %d statements in the main sample", int(frame["in_main_sample"].sum()))
    log.info("events: %d statements at the zero lower bound", int(frame["zlb"].sum()))
    return frame
