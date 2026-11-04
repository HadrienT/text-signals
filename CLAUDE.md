# text-signals — notes for Claude Code

A research project: extracting investment signals from text corpora with local LLMs, evaluating
the extraction, testing the signal against market data, and measuring look-ahead bias. The first
case study is FOMC communications; others (SEC filings, earnings calls, other central banks) will
follow. It is also a **portfolio project for a quantitative research internship application**
(AI/LLM tools for investment research). Optimise for a sound method and for clarity to a quant
reader, not for the number of features.

Read the protocol of the current case study first (`docs/case-studies/*.md`). Change it when a
decision changes, and write down why.

## Architecture

- **Generic core** in `src/text_signals/`: the document model (keyed by release timestamp, with
  provenance), the fetch cache and manifest, the LLM extraction client and its cache, the
  evaluation metrics, the point-in-time market join, the event study and the out-of-sample tools.
- **Per-corpus code** in `src/text_signals/sources/<corpus>/` (fetching and parsing) and, where
  needed, a corpus-specific extraction schema. A new corpus should only need a source module, a
  schema and a protocol document.
- Don't build the abstraction ahead of time. Write the FOMC case study first and pull code into
  the core when a second corpus actually needs it.

## Tracking work: GitHub Issues

Repository `HadrienT/text-signals`; `gh` is authenticated. One issue per piece of work (the FOMC
case study is split by week), plus a pinned `📋 Board` issue.

- At the start of a session, run `gh issue list --state open`.
- When an issue is done: `gh issue close <n> --comment "done in <sha>"`, then tick the box in the
  Board issue.
- No handoff markdown files.

## Conventions

- **English** for code, comments, commits, issues and docs. The conversation itself follows the
  maintainer's language (French).
- `uv` for everything; Python 3.12.
- `just lint` (ruff + `mypy --strict`) and `just test` must pass before any commit.
- Work on a branch and open a PR to `main`. End commit messages with
  `Co-Authored-By: Claude <noreply@anthropic.com>`.
- Small, incremental commits that tell the story of the work. Never rewrite or backdate history.
- Notebooks are for exploration only. Anything a result depends on lives in `src/text_signals/` and
  is tested.

## Research rules (non-negotiable)

- **Point-in-time.** Every document is keyed by its *release* timestamp, and market data is joined
  on that timestamp only. Never use a meeting date as an event date.
- **Fix the specification before looking.** Settle the aggregate score formula, the labelling of
  the hand set and the regression specification before seeing the outcome. If one changes after
  the results are known, say so in the plan and in the report.
- **Report negative results.** A null effect with a clean protocol is a valid outcome.
- **Look-ahead.** Always record which model scored which document, the model's training cutoff,
  and whether the meeting falls before or after it.
- **Provenance.** Store raw files as fetched, with URL, fetch time and sha256. Every LLM output is
  stored with the model name, prompt version, schema version and decoding parameters.

## Data

- `data/raw/`, `data/interim/` and `data/processed/` are git-ignored (see `data/README.md`).
  Downloads must be reproducible from code.
- federalreserve.gov and FRED: be polite, with a delay between requests and a descriptive
  User-Agent that does not contain a personal email address. Cache everything.
- FRED needs an API key in `.env` (`FRED_API_KEY`), never committed. The `fredgraph.csv` endpoint
  also works without a key.

## Local LLM

- On this host, `llama-server` (OpenAI-compatible) listens on `127.0.0.1:8000`. It is managed
  from `~/AgenticEnv` (`configs/models.yaml`, `just llm-use <profile>`). Only **one model is
  loaded at a time** and it is shared with the AgenticEnv agent, so switching profiles affects
  other users of the server. Ask before switching.
- Hardware: 2× V100 16 GB and 47 GiB of RAM. Long batch scoring runs need a cache keyed by
  (document sha256, model, prompt version) so that they can be resumed.
- FinBERT and other small encoders run with `transformers` on the GPU, as long as there is VRAM
  left next to `llama-server`.
