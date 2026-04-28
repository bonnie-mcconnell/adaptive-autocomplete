.PHONY: install demo test test-fast benchmark lint typecheck check all run

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	pip install poetry --quiet
	poetry install --no-interaction
	@echo ""
	@echo "Setup complete. To use 'aac' directly, activate the virtualenv:"
	@echo "  source .venv/bin/activate    (Linux/macOS)"
	@echo "  .venv\\Scripts\\activate       (Windows PowerShell)"
	@echo ""
	@echo "Or prefix any command with 'poetry run', e.g.:"
	@echo "  poetry run aac suggest he"

# ── Run (convenience wrapper so 'aac' works without activating the venv) ────

ARGS ?= --help

run:
	poetry run aac $(ARGS)

# ── Demo ─────────────────────────────────────────────────────────────────────

demo: install
	poetry run python scripts/demo.py

# ── Tests ────────────────────────────────────────────────────────────────────

test:
	poetry run pytest

test-fast:
	poetry run pytest -m "not integration"

# ── Benchmark ────────────────────────────────────────────────────────────────

benchmark:
	poetry run python -m aac.benchmarks.benchmark_engine

# ── Code quality ─────────────────────────────────────────────────────────────

lint:
	poetry run ruff check src tests examples scripts

typecheck:
	poetry run mypy src

check: lint typecheck test

# ── Default ──────────────────────────────────────────────────────────────────

all: install check
