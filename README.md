# fedspeak

**Does an LLM reading Federal Reserve communications extract a policy-tone signal that explains
rate moves beyond classic text baselines — and how much of that is look-ahead bias?**

Status: early work, set up on 2026-11-04. Results will show up here as they come.

## Question

FOMC statements, minutes and speeches move rates. Dictionary methods (Loughran–McDonald) and
finance-tuned encoders (FinBERT) score their tone. A general LLM can do more: say *what* changed
in the forward guidance, the inflation and labour-market assessment, and the balance-of-risks
wording since the previous meeting.

This project tests three things:

1. **Signal.** Does an LLM-derived hawkish/dovish score explain the change in the 2-year Treasury
   yield on announcement days, beyond the baselines?
2. **Reliability.** How accurate is the structured extraction against a hand-labelled set, and
   can a retrieval agent answer analyst questions with citations that hold up?
3. **Look-ahead bias.** LLMs have read what happened after the meetings in their training data.
   How much of the in-sample result goes away (a) after the model's training cutoff and (b) when
   documents are anonymised?

Full protocol: [docs/research-plan.md](docs/research-plan.md).

## Data (all public and point-in-time)

- FOMC statements, minutes and press conference transcripts: federalreserve.gov
- Treasury yields (DGS2, DGS10) and other series: FRED
- Loughran–McDonald sentiment dictionary

## Models

Open-weight LLMs served locally through an OpenAI-compatible `llama-server`. No document leaves
the machine. This is a deliberate constraint: research teams can rarely send internal material to
a hosted API.

## Layout

```
src/fedspeak/   library code (ingestion, scoring, evaluation)
tests/          pytest
docs/           research plan and design notes
data/           local data, not versioned (see data/README.md)
reports/        final research report
notebooks/      exploration only; anything that matters goes into src/
```

## Development

```bash
uv sync
just lint   # ruff + mypy --strict
just test
```
