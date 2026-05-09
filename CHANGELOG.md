# Changelog

All notable changes to this project are documented here.

Versions follow [Semantic Versioning](https://semver.org/). This project
was developed iteratively over several months; entries reflect the actual
evolution of the design as I understood it better.

---

## [1.0.0] - 2026-05-05

### Added
- `PredictorAcceptsRecord` protocol in `aac.ranking.contracts` - typed,
  documented contract for predictors that want to receive `record()` callbacks.
  Replaces `getattr` duck typing with a `runtime_checkable` Protocol.
- `aac.predictors._scoring` - shared scoring constants (`FREQ_WEIGHT`,
  `build_freq_scores`, `distance_score`, `edit_confidence`). All distance-based
  predictors import from one source of truth; previously `FREQ_WEIGHT = 0.5` was
  duplicated in `symspell.py` and `trigram.py` and could silently drift.
- `EditDistancePredictor` now accepts `frequencies` and applies the shared
  frequency multiplier, making it composable with SymSpell and Trigram in a
  weighted stack.
- `AutocompleteEngine.suggest_full()` - returns `word`, `count`, and `confidence`
  in a single pipeline pass, replacing the pattern of calling
  `suggest_with_history()` + `suggest_with_confidence()` separately (two runs).
- `aac.evaluation` module - `EvaluationHarness`, `WeightOptimiser`, `QueryLog`,
  `make_query_log_from_history()`, `make_synthetic_query_log()`. Full offline
  evaluation pipeline with P@k, MRR, NDCG, MAP, hit rate, and per-prefix-length
  breakdowns. `WeightOptimiser` supports grid search and coordinate descent with
  index caching so tuning the production preset costs ~2ms per evaluation
  instead of ~10s.
- `aac eval`, `aac tune`, `aac compare` CLI subcommands.
- `--json` and `--confidence` flags on `aac suggest`, `aac explain`, `aac compare`.
- `warm_cache()` in `aac.presets` - pre-builds all preset engines at startup.
- `EngineConfig.diff()` - human-readable diff of two engine configurations.
- `PredictorRegistry` - register third-party predictors by name for
  config-driven reconstruction.
- `docker-compose.yml` + `Dockerfile.demo` - one-command demo without Python.
- `contextual_history_example.py` and `custom_vocabulary_example.py` examples.
- Performance regression tests with concrete latency assertions (stateless p99
  < 5ms, production p99 < 30ms).
- `explain()` ordering agreement tests - for every preset, asserts that
  `explain()` and `suggest()` return the same ordering.
- `ThreadSafeHistory` concurrency tests (5 tests covering write accuracy, read
  consistency under concurrent writes, snapshot integrity).

### Fixed (pre-release audit - all changes made before first public commit)
- `AdaptiveSymSpellPredictor.name` is `"symspell"` so that `explain()`
  `base_components`, `EngineConfig` serialisation, and `WeightOptimiser` weight
  grids are consistent whether the engine uses `SymSpellPredictor` or
  `AdaptiveSymSpellPredictor`. The weight grid key and the predictor's `.name`
  must match - using `"adaptive_symspell"` as a grid key silently matched nothing.
- `demo.run()` lacked a `host` parameter; `HTTPServer` was hardcoded to
  `"127.0.0.1"`, making the container demo unreachable from the host even with
  the port exposed. Fixed with explicit `host` parameter threaded through to
  `HTTPServer` and `_find_free_port()`. CLI `aac demo` gains `--host` flag.
  `Dockerfile.demo` CMD updated to pass `--host 0.0.0.0`.
- `Dockerfile.demo` healthcheck used `curl` (not present in `python:3.12-slim`).
  Replaced with `python -c "import urllib.request; ..."`. Same fix in
  `docker-compose.yml`.
- `Dockerfile.demo` used `poetry install --no-dev` (deprecated in Poetry 1.2).
  Replaced with `--without dev`.
- All `assert isinstance(...)` guards in production code replaced with explicit
  `raise TypeError(...)`. `assert` is stripped by `-O` and provides no runtime
  protection in optimised builds.
- `AutocompleteEngine.describe()` and `ContextualHistory.total_entries()` both
  called `len(h.entries())` - O(n) tuple allocation. Changed to
  `len(h._entries)` (O(1)).
- `History` had no `__repr__`. Now returns `History(entries=N, prefixes=M)`.
- `ThreadSafeHistory.lock` docstring showed a deadlock-causing example. Rewritten
  to label the pattern clearly and explain why it deadlocks.
- `ThreadSafeHistory.__repr__` read `self._entries` after releasing the read-lock.
  Fixed to read inside the protected block.
- `vocabulary_from_file()` used `**kwargs: object` with manual `int()` casts.
  Replaced with explicit typed keyword-only parameters.
- `examples/fastapi_app.py` hardcoded `version="1.0.0"`. Now uses
  `aac.__version__`.
- `publish.yml` did not verify `__init__.py` version consistency.
  Added `python scripts/check_version.py` step.
- `pyproject.toml` development status was `4 - Beta`. Corrected to
  `5 - Production/Stable`.
- PyPI and Downloads badges removed from `README.md` (package not yet on PyPI;
  badges showed "unknown"). Replaced with static Python version badge.
- Benchmark docstring incorrectly stated `explain()` performs 2 history lookups.
  `explain()` runs a single forward pass and does not call `_apply_ranking()`.
- All README code examples verified against actual engine output and corrected:
  `recieve` ranking, `helo` ranking, `explain he` numbers, `compare_presets`
  table, opening example scores, `suggest_with_confidence` context.
- CLI `suggest.run()` called `engine.suggest(text)[:limit]` - computes all
  candidates then slices. Fixed to `engine.suggest(text, limit=limit)`.
- CLI `explain.run()` had the same issue. Fixed.

---


## [0.8.0] - 2026-04-30

### Added
- `EngineConfig.diff()` - compare two engine configs and get a human-readable
  list of what changed. Useful for auditing deployed vs local configs.
- `PredictorRegistry` - third-party predictors can now be registered by name
  so `EngineConfig.build()` can reconstruct custom engines, not just presets.
  Fixes the long-standing gap where `to_config()` worked for preset engines
  but `build()` raised `NotImplementedError` for custom ones.
- `ThreadSafeHistory.snapshot_counts()` and `copy()` overrides - these two
  methods were inherited from `History` without locking, meaning a concurrent
  write during either call could produce an inconsistent result. Both are now
  guarded by the read lock.
- `DecayRanker._rank_now` - stores the timestamp used in `rank()` so
  `explain()` reuses the exact same `datetime`, guaranteeing a cache hit on
  the second pass and avoiding a redundant O(k) history scan.
- Performance regression tests with concrete latency assertions. The benchmark
  script previously reported numbers but made no assertions - a 10× regression
  would have passed CI silently.

### Changed
- `AutocompleteEngine._check_ranker_invariant()` extracted as a shared static
  method. Previously `_apply_ranking()` and `explain()` each had their own copy
  of the invariant check - two places where the same logic could diverge.
- `LearningRanker._sort_key()` extracted as a shared method used by both
  `rank()` and `explain()`. The old `explain()` duplicated the sort lambda
  independently, meaning a change to `rank()`'s ordering logic would not
  automatically propagate to `explain()`.

### Fixed
- `EngineConfig.build()` for custom engines (previously raised
  `NotImplementedError` for any engine not built from a preset).

---

## [0.7.0] - 2026-04-12

### Added
- `EngineConfig` serialisation: `engine.to_config()` produces a
  JSON-serialisable config; `EngineConfig.from_json().build()` reconstructs
  the engine. Preset engines round-trip exactly.
- `ContextualHistory` - history keyed by both prefix and surrounding context
  (e.g. field name or application state). Allows different ranking for the
  same prefix depending on where in the UI the user is typing.
- `suggest_with_history()` - returns `(suggestion, count)` pairs so callers
  can display a selection count badge alongside completions.
- `available_presets()` function - returns the list of preset names without
  instantiating any engines.

### Changed
- Benchmark CI step now runs on every push, not just PRs. Latency numbers are
  printed to the CI log for trend tracking.
- `pyproject.toml` minimum Python bumped from 3.9 to 3.10 to allow
  `match` statements and `X | Y` union types without `from __future__ import`.

---

## [0.6.0] - 2026-03-29

### Added
- `robust` preset - combines all five predictor types with conservative weights.
  Designed for domains where typo recovery matters more than raw latency (e.g.
  search bars, form fields with free-text input).
- `AdaptiveSymSpellPredictor` - dispatches to a tighter index for prefixes of
  length ≤ 2 and the full distance-2 index for longer prefixes. Eliminates the
  candidate explosion that made `production` preset suggestions unstable for
  single-character queries.
- `explain()` API - returns a `RankingExplanation` per suggestion showing
  `base_score`, `history_boost`, `final_score`, and a per-predictor component
  breakdown. The `__post_init__` invariant check on `RankingExplanation`
  makes it impossible to construct an explanation where
  `final_score != base_score + history_boost`.

### Changed
- `production` preset now uses `AdaptiveSymSpellPredictor` instead of
  `SymSpellPredictor` directly.
- Minimum `limit` argument value is now 1 (previously 0 was silently accepted
  and returned an empty list).

---

## [0.5.0] - 2026-03-10

### Added
- `suggest_with_confidence()` - normalises final scores to [0, 1] relative to
  the top candidate. The `max(abs(top_score), 1e-9)` guard prevents division
  by zero for near-zero score distributions.
- `ThreadSafeHistory` - RW-lock pattern using `threading.Condition`. Multiple
  threads can read concurrently; writes wait for all readers to finish. Does
  not rely on CPython GIL atomicity guarantees so it is correct under PyPy
  and free-threaded CPython (PEP 703).
- Contract tests (`tests/contracts/`) - any class implementing the `Predictor`
  protocol is tested against a shared contract that verifies prefix-filtering,
  score non-negativity, and idempotency.

### Changed
- `DecayRanker` caches decayed counts per `(prefix, now)` so that `explain()`
  reuses the computation from `rank()` without a second history scan. Cache
  is always invalidated at the start of `rank()`.
- Error message for ranker set-preservation violation now names the offending
  ranker and lists both the added and removed entries.

---

## [0.4.0] - 2026-02-28

### Added
- `LearningRanker` - promotes suggestions with high selection history above
  their base frequency-weighted score. Uses a `dominance_ratio` parameter to
  prevent a heavily-selected word from drowning out all alternatives.
- `DecayRanker` - applies exponential time-decay to selection counts. Recent
  selections influence ranking more than old ones. Half-life is configurable.
- `recency` preset - `FrequencyPredictor` + `HistoryPredictor` + `DecayRanker`.
- Benchmark script (`make benchmark`) measuring p50/p95/p99 latency per preset.

### Fixed
- `HistoryPredictor` was including entries from all prefixes, not just the
  current prefix. Entries are now filtered by exact prefix match before scoring.

---

## [0.3.0] - 2026-02-08

### Added
- `TrigramPredictor` - uses character trigrams as a pre-filter before edit-
  distance verification, making fuzzy matching viable on larger vocabularies
  (48k words). The trigram threshold `max(1, len(query_trigrams) - max_distance)`
  is conservative; it may miss some true matches at the edges of the distance
  bound, which is documented explicitly.
- `TriePrefixPredictor` - exact prefix matching via a compressed trie. O(|prefix|)
  lookup, O(results) traversal. Used in the `stateless` preset as the primary
  exact-match predictor.
- `production` preset - all four predictor types with tuned weights.
- Property-based tests using Hypothesis (`tests/property/`). Running these
  exposed two floating-point edge cases in the score normaliser that
  example-based tests had missed.

---

## [0.2.1] - 2026-02-20

### Fixed
- `FrequencyPredictor` was resorting on every `predict()` call. Pre-sort
  at construction time (O(V log V) once) so `predict()` is O(max_results).
- `SymSpellPredictor._delete_map` construction did not handle empty vocabulary.

---

## [0.2.0] - 2026-02-15

### Added
- `SymSpellPredictor` - delete-neighbourhood index for O(1) average-case
  fuzzy lookup. Implemented from scratch (no `symspellpy` dependency) to
  understand the algorithm and avoid a 4MB transitive dependency.
- `ScoreRanker` - sorts scored suggestions by descending score after all
  predictors have run. Separating scoring (predictors) from ordering (rankers)
  makes it possible to add a ranker that adjusts order based on criteria other
  than raw score (e.g. recency, context).
- `WeightedPredictor` wrapper - each predictor has a weight applied to its
  scores before aggregation. Lets different predictor types contribute at
  different scales.

### Changed
- `AutocompleteEngine` constructor now takes `predictors: list[WeightedPredictor]`
  instead of a single predictor. This was the main structural rethink: the
  original single-predictor design made combining frequency and typo correction
  awkward.

---

## [0.1.7] - 2026-01-27

### Added
- `JsonHistoryStore` - persistence layer for `History`. Saves and loads from
  a JSON file. Format is a list of `{prefix, value, timestamp}` objects.
- `History.copy()` - returns a plain snapshot for passing to persistence
  layer without holding a lock.

---

## [0.1.6] - 2026-01-26

### Added
- `History` domain class - append-only log of `(prefix, selected_value,
  timestamp)` triples. Separated from `HistoryPredictor` so that multiple
  predictors and rankers can share the same history object.
- `HistoryPredictor` - returns previously-selected values for a prefix,
  scored by selection count relative to the total for that prefix.

---

## [0.1.5] - 2026-01-25

### Added
- `FrequencyPredictor` - scores candidates by log-normalised word frequency.
  `log(1 + freq) / log(1 + max_freq)` normalisation so scores are in [0, 1]
  regardless of the raw frequency distribution.
- Bundled English corpus (48k words with frequency counts derived from the
  Google Books n-gram corpus).

### Changed
- `Suggestion` and `ScoredSuggestion` are now frozen dataclasses.

---

## [0.1.4] - 2026-01-24

### Added
- `CompletionContext` - wraps the query text and optional cursor position.
  Passing a struct instead of a raw string leaves room for context-aware
  predictors without changing the `Predictor` protocol signature.
- `Predictor` protocol - structural typing via `typing.Protocol` so any class
  with a `predict(ctx) -> list[ScoredSuggestion]` method satisfies it.

---

## [0.1.3] - 2026-01-23

### Added
- `AutocompleteEngine` - first working version. Single predictor, no ranking,
  no history. `suggest(text)` returns a list of strings.

---

## [0.1.2] - 2026-01-22

### Added
- `Suggestion` dataclass.
- `ScoredSuggestion` dataclass wrapping `Suggestion` and a `float` score.

---

## [0.1.1] - 2026-01-20

### Added
- Project scaffold: `src/aac/`, `tests/`, `pyproject.toml`, `Makefile`.
- `py.typed` marker for PEP 561 compliance.
- CI via GitHub Actions: lint (ruff), typecheck (mypy --strict), test (pytest).

---

## [0.1.0] - 2026-01-17

### Added
- Initial commit. Empty package. README with the problem statement and
  intended design.
