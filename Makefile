.PHONY: install dev-setup demo demo-docker warm test test-fast test-perf benchmark benchmark-save benchmark-diff lint typecheck typecheck-examples pre-commit version-check check all run release release-zip

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

# ── Developer setup ──────────────────────────────────────────────────────────

# Install dependencies + pre-commit hooks. Run once after cloning.
dev-setup: install
	poetry run pre-commit install
	@echo "pre-commit hooks installed."

# Run all pre-commit hooks against every file (useful before a PR).
pre-commit:
	poetry run pre-commit run --all-files

# ── Run (convenience wrapper so 'aac' works without activating the venv) ────

ARGS ?= --help

run:
	poetry run aac $(ARGS)

# ── Demo ─────────────────────────────────────────────────────────────────────

# Run the interactive browser demo (requires Python + installed deps).
demo: install
	poetry run aac demo

# Run the demo via Docker - no Python or pip required locally.
# Opens at http://localhost:5000
demo-docker:
	docker compose up --build

# ── Tests ────────────────────────────────────────────────────────────────────

test:
	poetry run pytest

# Fast subset: skips @pytest.mark.integration tests (CLI subprocess calls),
# test_performance_regression.py (timing-sensitive latency gates), and
# test_property_based.py (Hypothesis fuzzing, thousands of generated inputs).
# Matches the push-event CI step exactly.
test-fast:
	poetry run pytest -m "not integration" --ignore=tests/test_performance_regression.py --ignore=tests/test_property_based.py

# Performance regression gate: enforces concrete latency upper bounds.
# Runs automatically in CI (ubuntu-latest, Python 3.12 only).
# Run locally before a PR to check you haven't introduced a regression.
# --no-cov: 5 perf tests alone produce ~32% coverage, triggering the 85%
# threshold. Coverage is meaningless for timing-only tests.
test-perf:
	poetry run pytest tests/test_performance_regression.py -v --no-cov

# Pre-build all preset engine indexes (SymSpell, trigram). Run once after install
# to avoid the first-call latency spike in compare_presets() and the demo.
# Takes ~8 seconds. After this, compare_presets() and aac compare are instant.
warm:
	poetry run python -c "from aac.presets import warm_cache; print('Building engines...'); warm_cache(); print('Done.')"



# Print p50/p95/p99 latency per preset. Informational; no assertions.
# For latency gates with assertions, use: make test-perf
benchmark:
	poetry run python -m aac.benchmarks.benchmark_engine

# Save current benchmark results as the performance baseline.
# Run this after a deliberate performance change to update the reference point.
# The baseline file (.benchmark_baseline.json) should be committed to the repo.
benchmark-save:
	poetry run python -m aac.benchmarks.benchmark_engine --save

# Compare current performance against the saved baseline.
# Exits with code 1 if any preset is >20% slower than the baseline p99.
# Run this before a PR to catch accidental regressions.
benchmark-diff:
	poetry run python -m aac.benchmarks.benchmark_engine --diff

# ── Code quality ─────────────────────────────────────────────────────────────

lint:
	poetry run ruff check src tests examples scripts

typecheck:
	poetry run mypy src

# Type-check examples separately (excluded from main mypy run to keep CI fast,
# but should be checked periodically - type errors in examples are silent bugs
# that mislead users copying the code). Run before a release.
typecheck-examples:
	poetry run mypy examples --ignore-missing-imports

# Verify that src/aac/__init__.py __version__ matches pyproject.toml version.
# Both must be updated together on a release; this catches the drift.
version-check:
	python scripts/check_version.py

check: lint typecheck test

# ── Default ──────────────────────────────────────────────────────────────────

all: install check

# ── Release ───────────────────────────────────────────────────────────────────

# Build a clean source distribution using git archive.
# Unlike 'zip -r', git archive excludes everything in .gitignore:
# __pycache__, .coverage, .ruff_cache, .pyc files, and build artifacts.
# Output: adaptive-autocomplete-<version>.tar.gz in the project root.
#
# Usage:
#   make release          # create the archive
#   make release-zip      # create a .zip instead (e.g. for GitHub releases)
VERSION := $(shell python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(d['tool']['poetry']['version'])")

release:
	@echo "Building release archive for version $(VERSION)..."
	git archive HEAD --format=tar.gz \
		--prefix=adaptive-autocomplete/ \
		--output=adaptive-autocomplete-$(VERSION).tar.gz
	@echo "Created adaptive-autocomplete-$(VERSION).tar.gz"
	@echo "Contents:"
	@tar -tf adaptive-autocomplete-$(VERSION).tar.gz | head -30
	@echo "  ..."

release-zip:
	@echo "Building release zip for version $(VERSION)..."
	git archive HEAD --format=zip \
		--prefix=adaptive-autocomplete/ \
		--output=adaptive-autocomplete-$(VERSION).zip
	@echo "Created adaptive-autocomplete-$(VERSION).zip"
