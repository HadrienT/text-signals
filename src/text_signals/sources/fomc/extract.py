"""Run the statement extraction on the evaluation sample (protocol §6.6).

Two readers fill the same schema with the same prompt:

- a **local model** served by ``llama-server`` (``local``), cached by text sha256, model, prompt
  and schema version, so that a run can stop and resume;
- **Claude**, the reference reader that replaces the hand labels. Claude Code is not an API this
  code can call, so ``blind-inputs`` writes one file per statement holding the text only (no
  date, no URL, no document id, no other model's output), a separate Claude session fills one
  JSON file per statement, and ``import`` validates them and stores them as
  ``ExtractionRecord`` like any other model's output.

Usage: ``python -m text_signals.sources.fomc.extract {local,blind-inputs,import} ...``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from text_signals.encoders import split_sentences
from text_signals.llm import (
    Decoding,
    ExtractionRecord,
    LLMClient,
    extract_all,
    repair_procedure,
    text_sha256,
)
from text_signals.sources.fomc.extraction import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_V2,
    StatementExtraction,
    StatementExtractionV2,
    evidence_problems,
    numbered,
    schema_v1_closed,
    schema_v2_for,
)
from text_signals.sources.fomc.prompts import (
    REPAIR_V1,
    STATEMENT_V2,
    STATEMENT_V3,
    STATEMENT_V4,
)

PROMPT = STATEMENT_V2
REPAIR = REPAIR_V1
PROMPT_V3 = STATEMENT_V3  # with schema v2: evidence by sentence number
PROMPT_V4 = STATEMENT_V4  # with schema v1 and a closed list of sentences as evidence
PLAN = Path("data/labels/fomc_statements_sample_v1.csv")
CONFIRM_PLAN = Path("data/labels/fomc_confirm_sample_v1.csv")
CONFIRM_VERSION = "fomc-confirm-sample-v1"
CONFIRM_SEED = 20261107
N_CONFIRM = 30
EVENTS = Path("data/processed/fomc/statement_events.csv")
STATEMENTS = Path("data/interim/fomc/statements.jsonl")
OUT = Path("data/interim/fomc/extractions")


@dataclass(frozen=True)
class SampleDoc:
    code: str
    doc_id: str
    release_date: date
    text: str


def sample_docs(plan: Path = PLAN, statements: Path = STATEMENTS) -> list[SampleDoc]:
    """The 65 statements of the labelling sample (60 + 5 practice), in code order."""
    texts = {
        d["doc_id"]: d["text"]
        for d in map(json.loads, statements.read_text("utf-8").splitlines())
        if d
    }
    rows = pd.read_csv(plan).sort_values("code")
    return [
        SampleDoc(
            str(r.code),
            str(r.doc_id),
            date.fromisoformat(str(r.release_date)),
            texts[str(r.doc_id)],
        )
        for r in rows.itertuples()
    ]


def procedure(max_repairs: int) -> str:
    return "single-pass" if max_repairs == 0 else repair_procedure(max_repairs, REPAIR)


def draw_confirmation(events: pd.DataFrame, used: set[str]) -> pd.DataFrame:
    """The confirmation sample (protocol §6.6): ``N_CONFIRM`` statements of the main sample drawn
    at random, without strata, from those not in the evaluation sample (``used``), under
    shuffled codes C01, C02, ... Same columns as the labelling plan, ``role`` "confirm"."""
    import random

    pool = sorted(set(events.loc[events["in_main_sample"], "doc_id"]) - used)
    rng = random.Random(CONFIRM_SEED)
    chosen = rng.sample(pool, N_CONFIRM)
    dates = events.set_index("doc_id")["release_date"]
    return pd.DataFrame(
        {
            "code": [f"C{i:02d}" for i in range(1, N_CONFIRM + 1)],
            "doc_id": chosen,
            "release_date": [dates[d] for d in chosen],
            "stratum": "",
            "role": "confirm",
            "second_pass": False,
            "sample_version": CONFIRM_VERSION,
        }
    )


def all_docs(statements: Path = STATEMENTS) -> list[SampleDoc]:
    """Every statement, in release order, its document id as its code (the scoring run)."""
    docs = [json.loads(line) for line in statements.read_text("utf-8").splitlines() if line]
    return sorted(
        (
            SampleDoc(d["doc_id"], d["doc_id"], date.fromisoformat(d["release_date"]), d["text"])
            for d in docs
        ),
        key=lambda d: (d.release_date, d.doc_id),
    )


def cache_path(model: str, out: Path = OUT) -> Path:
    slug = "".join(c if c.isalnum() or c in "-._" else "_" for c in model)
    return out / f"{slug}.jsonl"


def run_local_numbered(
    base_url: str, docs: list[SampleDoc], out: Path = OUT
) -> list[ExtractionRecord]:
    """Schema v2: the statement as numbered sentences, each evidence a sentence number.

    The text the model sees, and whose sha256 keys the cache, is ``numbered(sentences)``.
    """
    client = LLMClient(base_url)
    model, model_path = client.model_info()
    decoding = Decoding()
    sentences = {d.doc_id: split_sentences(d.text) for d in docs}

    def run(doc_id: str, text: str) -> ExtractionRecord:
        return client.extract(
            doc_id=doc_id,
            text=text,
            prompt=PROMPT_V3,
            schema=StatementExtractionV2,
            schema_version=SCHEMA_VERSION_V2,
            decoding=decoding,
            model=model,
            model_path=model_path,
            json_schema=schema_v2_for(len(sentences[doc_id])),
        )

    started = time.monotonic()

    def log(record: ExtractionRecord, cached: bool) -> None:
        status = "ok" if record.error is None else "INVALID"
        print(
            f"{record.doc_id} {status}{' (cache)' if cached else ''} "
            f"{record.latency_s or 0:.1f}s  [{time.monotonic() - started:.0f}s]",
            flush=True,
        )

    return extract_all(
        ((d.doc_id, numbered(sentences[d.doc_id])) for d in docs),
        run,
        model=model,
        prompt_version=PROMPT_V3.version,
        schema_version=SCHEMA_VERSION_V2,
        cache_path=cache_path(model, out),
        on_record=log,
    )


def run_local_closed(
    base_url: str, docs: list[SampleDoc], out: Path = OUT
) -> list[ExtractionRecord]:
    """Schema v1, each evidence restricted by the grammar to one whole sentence of the text."""
    client = LLMClient(base_url)
    model, model_path = client.model_info()
    decoding = Decoding()
    sentences = {d.doc_id: split_sentences(d.text) for d in docs}

    def run(doc_id: str, text: str) -> ExtractionRecord:
        return client.extract(
            doc_id=doc_id,
            text=text,
            prompt=PROMPT_V4,
            schema=StatementExtraction,
            schema_version=SCHEMA_VERSION,
            decoding=decoding,
            model=model,
            model_path=model_path,
            json_schema=schema_v1_closed(sentences[doc_id]),
        )

    started = time.monotonic()

    def log(record: ExtractionRecord, cached: bool) -> None:
        status = "ok" if record.error is None else "INVALID"
        print(
            f"{record.doc_id} {status}{' (cache)' if cached else ''} "
            f"{record.latency_s or 0:.1f}s  [{time.monotonic() - started:.0f}s]",
            flush=True,
        )

    return extract_all(
        ((d.doc_id, d.text) for d in docs),
        run,
        model=model,
        prompt_version=PROMPT_V4.version,
        schema_version=SCHEMA_VERSION,
        cache_path=cache_path(model, out),
        on_record=log,
    )


def run_local(
    base_url: str, docs: list[SampleDoc], out: Path = OUT, *, max_repairs: int = 0
) -> list[ExtractionRecord]:
    """Extract every sample statement; with ``max_repairs`` > 0, through the repair loop."""
    client = LLMClient(base_url)
    model, model_path = client.model_info()
    decoding = Decoding()

    def run(doc_id: str, text: str) -> ExtractionRecord:
        if max_repairs:
            return client.extract_with_repair(
                doc_id=doc_id,
                text=text,
                prompt=PROMPT,
                repair=REPAIR,
                verify=evidence_problems,
                max_repairs=max_repairs,
                schema=StatementExtraction,
                schema_version=SCHEMA_VERSION,
                decoding=decoding,
                model=model,
                model_path=model_path,
            )
        return client.extract(
            doc_id=doc_id,
            text=text,
            prompt=PROMPT,
            schema=StatementExtraction,
            schema_version=SCHEMA_VERSION,
            decoding=decoding,
            model=model,
            model_path=model_path,
        )

    started = time.monotonic()

    def log(record: ExtractionRecord, cached: bool) -> None:
        status = "ok" if record.error is None else "INVALID"
        if record.rejected:
            status += f" after {len(record.rejected)} repair(s), {len(record.problems_left)} left"
        print(
            f"{record.doc_id} {status}{' (cache)' if cached else ''} "
            f"{record.latency_s or 0:.1f}s  [{time.monotonic() - started:.0f}s]",
            flush=True,
        )

    return extract_all(
        ((d.doc_id, d.text) for d in docs),
        run,
        model=model,
        prompt_version=PROMPT.version,
        schema_version=SCHEMA_VERSION,
        cache_path=cache_path(model, out),
        procedure=procedure(max_repairs),
        on_record=log,
    )


def write_blind_inputs(docs: list[SampleDoc], directory: Path) -> None:
    """One ``<code>.txt`` per statement with its text only, plus the prompt and the schema."""
    directory.mkdir(parents=True, exist_ok=True)
    for d in docs:
        (directory / f"{d.code}.txt").write_text(d.text + "\n", encoding="utf-8")
    (directory / "PROMPT.txt").write_text(
        f"{PROMPT.version}\n\n--- system ---\n{PROMPT.system}\n\n--- user ---\n"
        f"{PROMPT.user_template}\n",
        encoding="utf-8",
    )


def import_reference(
    directory: Path, docs: list[SampleDoc], model: str, out: Path = OUT
) -> list[ExtractionRecord]:
    """Read ``<code>.json`` written by a reader without an API and store it as records."""
    records = []
    for d in docs:
        path = directory / f"{d.code}.json"
        if not path.is_file():
            continue
        raw = path.read_text("utf-8")
        output, error = None, None
        try:
            output = StatementExtraction.model_validate_json(raw).model_dump(mode="json")
        except ValidationError as exc:
            error = str(exc)
        records.append(
            ExtractionRecord(
                doc_id=d.doc_id,
                text_sha256=text_sha256(d.text),
                model=model,
                model_path=None,
                prompt_version=PROMPT.version,
                schema_version=SCHEMA_VERSION,
                decoding=None,
                created_at=datetime.fromtimestamp(path.stat().st_mtime, UTC),
                latency_s=None,
                prompt_tokens=None,
                completion_tokens=None,
                raw_output=raw,
                output=output,
                error=error,
            )
        )
    # Merge into the reader's cache: records of other documents (another sample) are kept, and
    # a document read again replaces its earlier record.
    target = cache_path(model, out)
    target.parent.mkdir(parents=True, exist_ok=True)
    new = {r.doc_id for r in records}
    old = target.read_text("utf-8").splitlines() if target.exists() else []
    kept = [
        r
        for r in map(ExtractionRecord.model_validate_json, filter(None, old))
        if r.doc_id not in new
    ]
    merged = [*kept, *records]
    target.write_text("".join(r.model_dump_json() + "\n" for r in merged), encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN, help="which sample to run on")
    parser.add_argument("--all", action="store_true", help="every statement, not a sample")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("draw-confirm")
    local = sub.add_parser("local")
    local.add_argument("--base-url", default="http://127.0.0.1:8000")
    local.add_argument(
        "--repairs", type=int, default=0, help="repair turns when the verifier finds a problem"
    )
    local.add_argument(
        "--numbered", action="store_true", help="schema v2: evidence by sentence number"
    )
    local.add_argument(
        "--closed", action="store_true", help="evidence limited to whole sentences of the text"
    )
    blind = sub.add_parser("blind-inputs")
    blind.add_argument("directory", type=Path)
    imp = sub.add_parser("import")
    imp.add_argument("directory", type=Path)
    imp.add_argument("--model", required=True)
    args = parser.parse_args()
    if args.cmd == "draw-confirm":
        used = set(pd.read_csv(PLAN)["doc_id"])
        plan = draw_confirmation(pd.read_csv(EVENTS), used)
        plan.to_csv(CONFIRM_PLAN, index=False)
        print(f"{len(plan)} statements written to {CONFIRM_PLAN}")
        return
    docs = all_docs() if args.all else sample_docs(args.plan)
    if args.cmd == "local":
        if args.numbered:
            records = run_local_numbered(args.base_url, docs)
        elif args.closed:
            records = run_local_closed(args.base_url, docs)
        else:
            records = run_local(args.base_url, docs, max_repairs=args.repairs)
    elif args.cmd == "blind-inputs":
        write_blind_inputs(docs, args.directory)
        print(f"{len(docs)} statements written to {args.directory}")
        return
    else:
        records = import_reference(args.directory, docs, args.model)
    bad = sum(r.error is not None for r in records)
    print(f"{len(records)} records, {bad} invalid")


if __name__ == "__main__":
    main()
