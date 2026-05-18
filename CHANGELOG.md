# Changelog

All notable changes to this project are documented here.

This project currently has one Git tag: `v1.0.0`, dated 2026-01-17.

---

## [1.0.4] - 2026-05-17

### Fixed
- Exported `average_precision` from `aac.evaluation` and added a regression
  test for the public export.
- Removed `_scoring.py` helpers from `aac.predictors.__all__`; the helpers stay
  available from `aac.predictors._scoring`.
- Simplified the optimiser's ranker cache key. The ranker cache is already
  separate from the predictor cache, so the `":rankers"` suffix was unnecessary.
- Cleaned up `evaluation.harness`: moved `defaultdict` to the module imports
  and made the prefix-length `"n"` value match its declared `float` type.
- Moved the FastAPI example's `asyncio` import to the top-level import block.
- Removed a stale `# noqa: E402` from the CLI demo import path.
- Included `CHANGELOG.md` in `scripts/check_version.py` so version drift is caught
  alongside `pyproject.toml` and `src/aac/__init__.py`.

### Changed
- Updated release docs and Makefile release targets to build source archives
  with `git archive`.
- Updated README, DESIGN, BENCHMARK, and CONTRIBUTING text for the current
  API and CI environment.

---

## [1.0.1] - 2026-05-12

### Added
- `AutocompleteEngine.predictors`, returning a copy of the engine's
  weighted predictor list. The predictor objects are the live instances, which
  lets callers update mutable predictor state without rebuilding the engine.
- Coverage configuration in `pyproject.toml`.
- Regression tests for engine invariants, evaluation edge cases,
  persistence, presets, `ThreadSafeHistory`, `FrequencyPredictor.add_word()`,
  batch APIs, and confidence calculations.

### Fixed
- Preserved `LearningRanker` parameters during `EngineConfig` serialisation.
- Fixed performance-test collection and disabled coverage gates for the
  dedicated performance regression step.
- Marked platform-specific and unreachable defensive paths with documented
  `# pragma: no cover` comments.

---

## 2026-05-09 to 2026-05-11

These commits updated the package around the completed core and moved package
metadata to `1.0.0`. They are listed separately because this checkout does not
contain a Git tag for them.

### Added
- BENCHMARK and DESIGN docs.
- Docker/browser demo support and updated FastAPI, evaluation,
  contextual history, and custom vocabulary examples.
- Public vocabulary helper exports at the package root.

### Changed
- Reworked CLI demo, record, explain, and benchmark output.
- Tightened engine invariant checks and type annotations.
- Improved ranking explanation docs and JSON storage error handling.
- Regenerated `poetry.lock` after project metadata changes.

### Fixed
- Preserved caller-provided history instances across preset construction.
- Fixed CLI vocabulary argument naming and root vocabulary exports.
- Stabilised CLI parsing and the demo record route.
- Made history length checks constant-time.
- Relaxed performance thresholds for loaded CI runners.

---

## 2026-04-28 to 2026-05-01

### Added
- `AdaptiveSymSpellPredictor`, `ThreadSafeHistory`,
  `ContextualHistory`, `EngineConfig`, `PredictorRegistry`, and vocabulary
  utilities.
- Single-pass `explain()` output with contribution percentages and
  `reset_history()` propagation.
- `History.copy()` for independent snapshots.
- Tests for async APIs, CLI integration, engine config, evaluation,
  explanation ordering, SymSpell, vocabulary utilities, persistence, and
  thread-safe history.

### Changed
- Replaced the BK-tree robust preset with SymSpell; BK-tree remains available
  for comparison.
- Moved frequency and history predictors to log-normalised scoring.
- Excluded exact matches from approximate predictors.
- Updated package classifiers, examples, Makefile targets, README, and release
  docs.

### Fixed
- Fixed a Windows file descriptor leak in `JsonHistoryStore`.
- Fixed tests and docs after the scoring/explanation redesign.

---

## 2026-04-04 to 2026-04-22

### Added
- Real word-frequency data, default vocabulary constants, vocabulary-file
  loading, and word-list support.
- BK-tree and trigram fuzzy predictors with tests and benchmarks.
- JSON history persistence with timestamped `HistoryEntry` objects and
  backwards-compatible loading.
- Hypothesis tests and performance regression checks.
- CHANGELOG, CONTRIBUTING, BENCHMARK, DESIGN, SECURITY, and Makefile
  workflow docs.

### Changed
- Replaced hardcoded demo vocabularies with configurable vocabulary sources.
- Improved history and decay lookups with prefix indexing.
- Improved frequency/history scoring, deterministic sorting, and typo-recovery
  weighting.
- Updated CI for benchmark and test coverage work.

### Fixed
- Fixed incorrect invariant checks and an `explain()` double-counting bug.
- Fixed fake predictor test signatures, predictor imports, vocabulary range
  handling, CLI defaults, and demo docs.
- Fixed JSON persistence with atomic temp-file replacement.
- Fixed shared-history behaviour so learning rankers use the intended history.

---

## [1.0.0] - 2026-01-17

First completed Git release. The project already had a working engine, CLI,
presets, persistence, ranking, tests, and documentation at this point.

### Added
- Core autocomplete engine with weighted predictor aggregation.
- Domain types for completion context, suggestions, scored suggestions, history,
  and feedback.
- Predictors for frequency, history, static prefix, trie prefix, and edit
  distance.
- Ranking layer with score, learning, decay, weighted ranking, contracts, and
  explanations.
- JSON history storage.
- Presets for stateless, default, recency-aware, and robust behaviour.
- CLI commands for suggesting, recording selections, explaining rankings,
  debugging, and inspecting presets.
- Benchmark harness and developer/debug pipeline modules.
- Test suite covering domain models, predictors, ranking invariants,
  explanations, history learning, presets, and smoke/integration paths.
- Package metadata, `py.typed`, Ruff, mypy, pytest, and GitHub Actions CI.
- README describing the project, CLI demo, presets, and architecture.

---

## Before 1.0.0 - 2025-12-13 to 2026-01-16

Development leading up to the first completed Git release.

### Added
- Initial project scaffold, package layout, test layout, CI, linting, typing,
  and pytest configuration.
- First versions of the domain model, engine, predictors, rankers, persistence,
  CLI, presets, debug utilities, and README.

### Changed
- Iterated on typing, imports, predictor protocols, ranking invariants,
  explanation semantics, and CLI output before the `v1.0.0` tag.
