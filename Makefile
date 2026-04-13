.PHONY: install demo test benchmark lint typecheck check all

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	pip install poetry --quiet
	poetry install --no-interaction

# ── Demo ─────────────────────────────────────────────────────────────────────

demo: install
	@echo ""
	@echo "── Frequency-ranked suggestions for 'he' ──────────────────────────"
	poetry run aac suggest he
	@echo ""
	@echo "── Score breakdown per suggestion ─────────────────────────────────"
	poetry run aac explain he
	@echo ""
	@echo "── Recording two selections of 'hero' ─────────────────────────────"
	poetry run aac record he hero
	poetry run aac record he hero
	@echo ""
	@echo "── Suggestions after learning (hero rises) ─────────────────────────"
	poetry run aac suggest he
	@echo ""
	@echo "── Typo recovery: 'programing' → 'programming' (production preset) ─"
	poetry run aac --preset production suggest programing

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
