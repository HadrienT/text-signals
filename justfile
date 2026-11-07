# Task runner for text-signals. `just --list` for everything.

set dotenv-load := true

# ruff + mypy --strict: green before any commit.
lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy src tests

# Unit tests.
test:
    uv run pytest

# FOMC statements, Treasury yields and LM scores into data/ (cached; --refresh for new data).
fetch-fomc *args:
    uv run python -m text_signals.sources.fomc {{args}}

# Cache a manually downloaded Loughran-McDonald CSV (when Google Drive refuses scripted downloads).
import-lm file:
    uv run python -m text_signals.lexicon import-lm {{file}}

# Labelling page on the LAN (port 8772): audio from AudioBook's fomc-labelling-v1 workspace.
label *args:
    uv run python -m text_signals.sources.fomc.label_app {{args}}

# LLM extraction of the 65 sample statements with the model llama-server is serving (cached).
extract-local *args:
    uv run python -m text_signals.sources.fomc.extract local {{args}}

# Evaluation of a model's extraction, optionally against a reference reader.
extract-eval model reference="":
    uv run python -m text_signals.sources.fomc.extract_eval {{model}} {{reference}}

# Score all 250 statements with the served model (Phi-4: `just llm-use research`); cached.
score-fomc:
    uv run python -m text_signals.sources.fomc.extract --all local --repairs 1
    uv run python -m text_signals.sources.fomc.llm_scores

# One analysis of the event study: primary | diagnostics | robustness | secondary | oos | placebos.
study cmd:
    uv run python -m text_signals.sources.fomc.study {{cmd}}

# Every result of reports/fomc.md, recomputed from data/ and the LLM caches (no LLM call).
results:
    #!/usr/bin/env bash
    set -euo pipefail
    for cmd in primary diagnostics robustness secondary oos placebos; do
        echo "== study $cmd"; uv run python -m text_signals.sources.fomc.study "$cmd"
    done
    echo "== look-ahead"; uv run python -m text_signals.sources.fomc.lookahead report
    echo "== regime-matched cutoff"; uv run python -m text_signals.sources.fomc.lookahead regime
    echo "== counterfactual edits"; uv run python -m text_signals.sources.fomc.counterfactual report
    echo "== ChronoGPT"; uv run python -m text_signals.sources.fomc.lookahead chrono
    echo "== second reader"; uv run python -m text_signals.sources.fomc.claude_signal
    echo "== retrieval agent"
    uv run python -m text_signals.sources.fomc.qa_eval retrieval --split test
    uv run python -m text_signals.sources.fomc.qa_eval answers --split test
