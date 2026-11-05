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
