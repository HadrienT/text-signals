"""The LLM scores of every statement, from the chosen extraction procedure (protocol §5.2, §6.6).

Values come from the **first answer** of the repair run (identical to a single pass); the repaired
answer only audits the evidence. Each row carries the provenance the research rules ask for: the
model, its weights file, prompt and schema versions, and whether the statement was released
after the model's training-data cutoff (look-ahead, §6.7).

Usage: ``python -m text_signals.sources.fomc.llm_scores`` writes
``data/processed/fomc/llm_scores_<model>.csv``.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from text_signals.llm import ExtractionRecord
from text_signals.sources.fomc.extract import REPAIR, all_docs, cache_path, procedure
from text_signals.sources.fomc.extract_eval import load_reader, values_from_repair
from text_signals.sources.fomc.extraction import (
    StatementExtraction,
    guidance_score,
    score_changes,
    tone_score,
)

# Training-data cutoffs stated by the model publishers (protocol §5.2).
TRAINING_CUTOFF = {
    # microsoft/phi-4 model card: "cutoff dates of June 2024 and earlier for publicly available
    # data"; taken as the last day of that month.
    "phi-4": date(2024, 6, 30),
}


def score_table(
    model: str,
    records: dict[str, ExtractionRecord],
    days: dict[str, date],
    from_repair: set[str] | frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """One row per statement, in release order, with the scores and their provenance.

    ``from_repair``: statements whose first answer failed validation, so that their values come
    from the repaired answer.
    """
    cutoff = TRAINING_CUTOFF.get(model)
    rows = []
    for doc_id in sorted(days, key=lambda d: (days[d], d)):
        rec = records.get(doc_id)
        ext = (
            StatementExtraction.model_validate(rec.output)
            if rec is not None and rec.output is not None
            else None
        )
        rows.append(
            {
                "doc_id": doc_id,
                "release_date": days[doc_id].isoformat(),
                "llm_tone": None if ext is None else tone_score(ext),
                "llm_guidance": None if ext is None else guidance_score(ext),
                "llm_valid": ext is not None,
                "llm_model": model,
                "llm_model_path": None if rec is None else rec.model_path,
                "llm_prompt_version": None if rec is None else rec.prompt_version,
                "llm_schema_version": None if rec is None else rec.schema_version,
                "llm_values_from": "repaired (first invalid)"
                if doc_id in from_repair
                else "first answer",
                "llm_after_cutoff": None if cutoff is None else days[doc_id] > cutoff,
            }
        )
    frame = pd.DataFrame(rows)
    frame["d_llm_tone"] = score_changes(
        [None if pd.isna(v) else float(v) for v in frame["llm_tone"]]
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="phi-4")
    args = parser.parse_args()
    docs = all_docs()
    records = load_reader(args.model, docs, procedure(1), first=True)
    days = {d.doc_id: d.release_date for d in docs}
    table = score_table(args.model, records, days, values_from_repair)
    out = Path("data/processed/fomc") / f"llm_scores_{cache_path(args.model).stem}.csv"
    table.to_csv(out, index=False)
    print(
        f"{len(table)} statements, {int((~table['llm_valid']).sum())} without a score, "
        f"{int(table['llm_after_cutoff'].fillna(False).sum())} after the training cutoff, "
        f"{len(values_from_repair)} with values from the repaired answer; "
        f"repair message {REPAIR.version} used for the evidence audit only -> {out}"
    )


if __name__ == "__main__":
    main()
