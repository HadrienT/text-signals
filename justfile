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

# Fetch the FOMC calendar and statements into data/ (cached; --refresh for new meetings).
fetch-fomc *args:
    uv run python -m text_signals.sources.fomc {{args}}
