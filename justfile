# Task runner for fedspeak. `just --list` for everything.

set dotenv-load := true

# ruff + mypy --strict: green before any commit.
lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy src tests

# Unit tests.
test:
    uv run pytest
