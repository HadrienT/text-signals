"""A frontier model as an exploratory second reader of the statements (protocol §6.9).

``claude-opus-5-5`` reads each statement in a fresh ``claude -p`` process, under the same
prompt (v2) and schema (v1) as Phi-4, one pass. Readings are cached apart from the reference
reader's (§6.6), which is the same model run another way.

Usage:
  ``python -m text_signals.sources.fomc.claude_reader trial``: the 10-statement trial;
  ``python -m text_signals.sources.fomc.claude_reader compare``: the trial against Phi-4 and FRED.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from text_signals.claude_code import ClaudeCodeReader
from text_signals.llm import ExtractionRecord, extract_all, read_cache
from text_signals.sources.fomc.extract import PROMPT, SampleDoc, all_docs
from text_signals.sources.fomc.extraction import SCHEMA_VERSION, StatementExtraction

MODEL = "claude-opus-5-5"
TRIAL_SEED = 20261110  # protocol §6.9
TRIAL_N = 10
CACHE = Path("data/interim/fomc/extractions/claude-code") / f"{MODEL}.jsonl"
# Outside the repository: Claude Code looks for CLAUDE.md files up the directory tree, so a
# working directory inside the repo would load the project's instructions.
WORKDIR = Path(tempfile.gettempdir()) / "text-signals-claude-reader"


def trial_docs(docs: list[SampleDoc], n: int = TRIAL_N, seed: int = TRIAL_SEED) -> list[SampleDoc]:
    rng = np.random.default_rng(seed)
    ordered = sorted(docs, key=lambda d: d.doc_id)
    picked = sorted(int(i) for i in rng.choice(len(ordered), size=n, replace=False))
    return [ordered[i] for i in picked]


def read(docs: list[SampleDoc]) -> list[ExtractionRecord]:
    reader = ClaudeCodeReader(WORKDIR, model=MODEL)
    started = time.monotonic()

    def run(doc_id: str, text: str) -> ExtractionRecord:
        return reader.extract(
            doc_id=doc_id,
            text=text,
            prompt=PROMPT,
            schema=StatementExtraction,
            schema_version=SCHEMA_VERSION,
        )

    def log(record: ExtractionRecord, cached: bool) -> None:
        status = "ok" if record.error is None else f"INVALID ({record.error[:80]})"
        cost = (record.meta or {}).get("cost_usd_equivalent") or 0
        print(
            f"{record.doc_id} {status}{' (cache)' if cached else ''} "
            f"{record.latency_s or 0:.0f}s ${cost:.3f}  [{time.monotonic() - started:.0f}s]",
            flush=True,
        )

    return extract_all(
        ((d.doc_id, d.text) for d in docs),
        run,
        model=MODEL,
        prompt_version=PROMPT.version,
        schema_version=SCHEMA_VERSION,
        cache_path=CACHE,
        on_record=log,
    )


def compare(docs: list[SampleDoc]) -> pd.DataFrame:
    """Claude and Phi-4 side by side on ``docs``, with the FRED decision (no market data)."""
    from text_signals.fetch import Fetcher
    from text_signals.market import fetch_fred_series
    from text_signals.sources.fomc.extract import procedure
    from text_signals.sources.fomc.extract_eval import (
        decision_from_change,
        load_reader,
        policy_target,
        reader_table,
        target_change_bp,
    )

    claude = {r.doc_id: r for r in read_cache(CACHE).values() if r.output is not None}
    phi = load_reader("phi-4", docs, procedure(1), first=True)
    a = reader_table(claude, docs).set_index("doc_id")
    b = reader_table(phi, docs).set_index("doc_id")
    with Fetcher(Path("data/raw"), delay_s=2.0) as fetcher:
        target = policy_target(
            fetch_fred_series(fetcher, "DFEDTAR"), fetch_fred_series(fetcher, "DFEDTARU")
        )
    change = target_change_bp([d.release_date for d in docs], target)
    out = pd.DataFrame(
        {
            "true_decision": [decision_from_change(c) for c in change],
            "true_bp": change,
            "phi_decision": b["policy_action"].to_numpy(),
            "claude_decision": a["policy_action"].to_numpy(),
            "phi_bp": b["change_bp"].to_numpy(),
            "claude_bp": a["change_bp"].to_numpy(),
            "phi_tone": b["llm_tone"].to_numpy(),
            "claude_tone": a["llm_tone"].to_numpy(),
            "phi_guidance": b["forward_guidance"].to_numpy(),
            "claude_guidance": a["forward_guidance"].to_numpy(),
            "phi_not_verbatim": b["hallucinated"].to_numpy(),
            "claude_not_verbatim": a["hallucinated"].to_numpy(),
        },
        index=[d.doc_id.removeprefix("fomc-statement-") for d in docs],
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=["trial", "compare"])
    args = parser.parse_args()
    docs = trial_docs(all_docs())
    if args.cmd == "trial":
        records = read(docs)
        new = [r for r in records if r.meta]
        cost = sum((r.meta or {}).get("cost_usd_equivalent") or 0 for r in new)
        print(
            f"{len(records)} records, {sum(r.error is not None for r in records)} invalid, "
            f"API-equivalent cost ${cost:.2f}"
        )
        return
    pd.set_option("display.width", 250)
    print(compare(docs).to_string())


if __name__ == "__main__":
    main()
