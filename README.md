# text-signals

**Extracting investment signals from text corpora with local LLMs, and measuring honestly
whether they hold up.**

Status: case study 1 (FOMC communications) is complete. Read the
[white paper (PDF)](reports/fomc_whitepaper.pdf) or the longer
[research report](reports/fomc.md); the [protocol](docs/case-studies/fomc.md) records every
decision with its date and reason.

## Main results: FOMC statements and the 2-year yield

A local 14 B model (Phi-4, served by `llama-server`) reads each of 250 FOMC statements
(1994–2026) into a structured extraction with verbatim evidence. Its aggregate tone score is
tested against the 2-year Treasury yield's move in the 30 minutes around the release, controlling
for the rate surprise (MP1) and two text baselines (Loughran–McDonald and FinBERT). The
specification was fixed before any score met market data.

| Question | Result |
|---|---|
| **Does the tone explain the 2-year move? (primary test, H1)** | **Yes: +1.03 bp per SD of tone**, HAC p = 0.004, n = 243 (bootstrap 95% interval 0.36 to 1.77) |
| Robust to HC3, bootstrap, influential points, Huber? | Yes, all significant at 5%; Huber gives 0.73 bp |
| Do the dictionary and FinBERT baselines add anything? | No, once the LLM score is in |
| Out of sample (yearly refits, 2000–2026) | +2.1% R² over the baseline, Clark–West p = 0.003; earned before 2009 and in 2022 |
| Permutation placebo (9,999 shuffles) | p = 0.005 |
| Where it stops | No effect on the daily close; much weaker on press conference days (every meeting since 2019) |
| **Look-ahead bias (H3)** | The model can date anonymised statements, but rewriting a statement's guidance moves its score every time, as much for statements it may remember as for those after its cutoff: no sign that memory drives the score |
| A frontier model (Claude Opus) as a second reader | Cleaner extraction (all evidence verbatim) but the same signal: β 0.90 vs 0.84 for Phi-4 on the same texts |
| Retrieval agent over 608 FOMC documents (40 test questions, run once) | 21 of 32 answerable right, 6 of 8 traps refused, 95% of citations verbatim. It also failed in two instructive ways: exact quotes attached to the wrong meeting, and one answer from memory despite the point-in-time filter |

Negative and failed results are reported alongside: the daily outcome, H2 (change in tone) not
significant after correction, a placebo expectation that failed, a look-ahead-free reader too
weak to use, and an inconclusive regime test.

## What this is

Investment research reads a lot of unstructured text: central bank communications, corporate
filings, earnings calls, news. This repository builds a reusable pipeline that turns such a corpus
into structured, auditable signals, and evaluates them on three axes:

1. **Signal.** Does the LLM-derived measure explain or predict a market variable, beyond classic
   text baselines (dictionaries, finance-tuned encoders)?
2. **Reliability.** How accurate is the extraction, and do the citations of a retrieval agent
   actually support its answers?
3. **Look-ahead bias.** LLMs have read the future of every historical document in their training
   data. How much of a result survives after the model's training cutoff, on anonymised
   documents, or when the text is edited to contradict what the model remembers?

## Research rules

- **Point-in-time.** Every document is keyed by its release timestamp; market data are joined on
  it only. Retrieval has an `as_of` filter for the same reason.
- **Fix the specification before looking.** Score formula, primary test, controls and inference
  were frozen before any score was joined to market data. Changes made after a result was seen
  are labelled as such.
- **Report negative results.**
- **Provenance.** Raw files are stored as fetched, with URL, fetch time and sha256. Every LLM
  output is stored with model, weights file, prompt and schema versions and decoding parameters.

## Pipeline

```
fetch → parse (keyed by release timestamp) → score (baselines + LLM extraction)
      → evaluate the extraction → join market data point-in-time → event study, out of sample,
        placebos, look-ahead tests
      → retrieval index → question answering with checked citations
```

### Requirements

- Python 3.12 with [`uv`](https://docs.astral.sh/uv/), and [`just`](https://github.com/casey/just).
- For the LLM steps: an OpenAI-compatible `llama-server` on `127.0.0.1:8000` serving Phi-4
  (`microsoft/phi-4-gguf`, `phi-4-Q8_0.gguf`). A GPU helps with FinBERT and the retrieval
  encoder, but is not required.
- Optional: a FRED API key in `.env` (copy `.env.example`). The public `fredgraph.csv` endpoint
  also works without one.
- Optional, for the second reader only: Claude Code, logged in.

### Rerun from scratch

```bash
uv sync

# 1. Fetch and parse: FOMC calendar, statements, minutes, press conference transcripts;
#    USMPD intraday changes and surprises; FRED yields and targets; LM and FinBERT scores.
just fetch-fomc

# 2. LLM extraction of all statements with Phi-4 (cached by document, model and prompt).
just score-fomc

# 3. Every result of the report, from data/ and the caches: event study, robustness,
#    secondary tests, out of sample, placebos, look-ahead tests, second reader, retrieval agent.
just results
```

The steps that call a model, each cached and resumable:

```bash
uv run python -m text_signals.sources.fomc.lookahead anonymised-run   # Phi-4 on anonymised texts
uv run python -m text_signals.sources.fomc.lookahead probe-run        # dating probe
uv run python -m text_signals.sources.fomc.counterfactual run         # counterfactual edits (test B)
uv run python -m text_signals.sources.fomc.chrono_scores              # ChronoGPT vintages, CPU
uv run python -m text_signals.sources.fomc.qa index                   # retrieval passages and embeddings
uv run python -m text_signals.sources.fomc.qa_answer run --split test # retrieval agent answers
uv run python -m text_signals.sources.fomc.claude_batch full          # second reader (Claude Code)
```

Downloads are polite (a delay between requests, a descriptive User-Agent) and cached under
`data/raw/` with a manifest. `data/` is not versioned except `data/labels/`, which holds the
inputs that cannot be regenerated: evaluation samples, counterfactual edits, retrieval questions
and judgments.

## Layout

```
src/text_signals/             generic core: documents, fetch cache, LLM client and cache,
                              encoders, event study, out-of-sample tools, retrieval
src/text_signals/sources/fomc corpus-specific: fetching, parsing, schema, study, look-ahead
                              tests, retrieval agent
tests/                        pytest; `just lint` (ruff, mypy --strict) and `just test`
docs/case-studies/            one protocol per case study
reports/                      research reports
data/                         local data (see data/README.md); data/labels/ is versioned
```

## Case studies

| # | Corpus | Question | Status |
|---|---|---|---|
| 1 | FOMC statements, minutes, press conferences | Does an LLM policy-tone score explain 2-year yield moves on announcement days, and does it survive look-ahead tests? | complete: [report](reports/fomc.md), [protocol](docs/case-studies/fomc.md) |
| – | SEC filings, earnings calls, other central banks | to be defined | – |

A new corpus needs a source module, an extraction schema and a protocol; the core is shared.

## Models

Open-weight LLMs served locally, so no document leaves the machine. This is a deliberate
constraint: research teams often cannot send internal material to a hosted API. The study also
measures what a frontier model adds on this task: cleaner, fully verbatim extraction, but no
stronger signal.

## Development

```bash
uv sync
just lint   # ruff + mypy --strict
just test
```

## License

MIT, see [LICENSE](LICENSE).
