# fedspeak — notes for Claude Code

A research project: an LLM tone signal from FOMC communications, its evaluation, and look-ahead
bias. It is also a **portfolio project for a quantitative research internship application**
(AI/LLM tools for investment research). Optimise for a sound method and for clarity to a quant
reader, not for the number of features.

Read `docs/research-plan.md` first. It is the protocol. Change it when a decision changes, and
write down why.

## Tracking work: GitHub Issues

Repository `HadrienT/fedspeak`; `gh` is authenticated. One issue per week of the plan, plus a pinned
`📋 Board` issue.

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
- Notebooks are for exploration only. Anything a result depends on lives in `src/fedspeak/` and
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
