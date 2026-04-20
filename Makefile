.PHONY: install demo test benchmark lint typecheck check all

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	pip install poetry --quiet
	poetry install --no-interaction

# ── Demo ─────────────────────────────────────────────────────────────────────

demo: install
	poetry run python scripts/demo.py

# ── Tests ────────────────────────────────────────────────────────────────────

test:
	poetry run pytest -q

# ── Benchmark ────────────────────────────────────────────────────────────────

benchmark:
	poetry run python -m aac.benchmarks.benchmark_engine

# ── Code quality ─────────────────────────────────────────────────────────────

lint:
	poetry run ruff check src tests

typecheck:
	poetry run mypy src

check: lint typecheck test

# ── Default ──────────────────────────────────────────────────────────────────

all: install check
