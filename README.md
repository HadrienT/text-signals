# text-signals

**Extracting investment signals from text corpora with local LLMs, and measuring honestly
whether they hold up.**

Status: early work, set up on 2026-11-04. Results will show up here as they come.

## What this is

Investment research reads a lot of unstructured text: central bank communications, corporate
filings, earnings calls, news. This repository builds a reusable pipeline that turns such a corpus
into structured, auditable signals, and evaluates them on three axes:

1. **Signal.** Does the LLM-derived measure explain or predict a market variable, beyond classic
   text baselines (dictionaries, finance-tuned encoders)?
2. **Reliability.** How accurate is the extraction against hand-labelled data, and do the
   citations of a retrieval agent actually support its answers?
3. **Look-ahead bias.** LLMs have read the future of every historical document in their training
   data. How much of a backtest survives after the model's training cutoff, or once the documents
   are anonymised?

The pipeline is the same for every corpus:

```
fetch → parse (keyed by release timestamp) → score (baselines + LLM extraction)
      → evaluate extraction → join market data point-in-time → event study / out-of-sample test
```

## Case studies

| # | Corpus | Question | Status |
|---|---|---|---|
| 1 | FOMC statements, minutes, press conferences | Does an LLM policy-tone score explain 2-year yield moves on announcement days? | in progress, [protocol](docs/case-studies/fomc.md) |
| – | SEC filings (10-K/10-Q), earnings calls, other central banks… | to be defined | – |

## Models

Open-weight LLMs served locally through an OpenAI-compatible `llama-server`. No document leaves
the machine. This is a deliberate constraint: research teams can rarely send internal material to
a hosted API.

## Layout

```
src/text_signals/   library code (generic pipeline + per-corpus sources)
tests/              pytest
docs/case-studies/  one protocol per case study
data/               local data, not versioned (see data/README.md)
reports/            research reports
notebooks/          exploration only; anything that matters goes into src/
```

## Development

```bash
uv sync
just lint   # ruff + mypy --strict
just test
```
