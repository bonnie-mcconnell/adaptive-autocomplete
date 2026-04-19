# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added

- Property-based tests with Hypothesis covering three core invariants:
  `RankingExplanation` arithmetic, ranker candidate-set preservation, and
  History prefix-index consistency. 11 new tests; 241 total.
- `hypothesis = "^6.100"` added to dev dependencies.

### Fixed

- **`RankingExplanation.merge()` floating-point precision bug** (found by
  Hypothesis): `final_score` was computed as the sum of four independent
  terms (`self.base + other.base + self.boost + other.boost`), which
  rounds differently from the two-term sum used by `__post_init__`
  (`base + boost`). Fixed by computing `merged_base` and `merged_boost`
  separately, then deriving `final_score = merged_base + merged_boost`.
  This matches the invariant check and eliminates the precision divergence.
- **`ScoredSuggestion.trace`**: changed from `list[str]` to `tuple[str, ...]`.
  `frozen=True` prevents reassignment of the field but not mutation of a
  list's contents. Using `tuple` makes the immutability guarantee complete.
  All internal construction sites updated to use tuple literals and `+ (item,)`
  concatenation.
- **`_DEFAULT_VOCABULARY` in `presets.py`**: was loaded at import time,
  triggering a 717 KB JSON parse as a side effect of `import aac.presets`.
  Changed to a lazy loader `_get_default_vocabulary()` backed by
  `lru_cache`; the JSON is parsed once on first actual use.
- **`History` docstring**: "Safe to share across predictors and rankers"
  was ambiguous about write safety. Added explicit thread-safety section:
  reads are safe concurrently; `record()` is not thread-safe and requires
  external locking for multi-threaded writers.
- **`JsonHistoryStore.save()` docstring**: documented Windows non-atomicity.
  On POSIX, `rename()` is atomic. On Windows, `Path.replace()` is not atomic
  when the destination exists (delete-then-rename at the OS level).
- **CHANGELOG date**: `[0.1.0] - 2025` corrected to `[0.1.0] - 2025.
- **CI**: added benchmark step (Python 3.12 only) that uploads results as a
  GitHub Actions artifact, making the README performance table independently
  verifiable without cloning.

## [0.1.0] - 2025

### Added

- `AutocompleteEngine`: composable pipeline of predictors, rankers, and history
- `FrequencyPredictor`: prefix index pre-sorted by frequency at construction; O(max_results) per query
- `HistoryPredictor`: recall-based predictor driven by user selection counts
- `TrigramPredictor`: approximate matching via trigram pre-filter + exact Levenshtein; ~600µs at 48k words
- `EditDistancePredictor`: BK-tree approximate matching; exact recall on small vocabularies
- `TriePrefixPredictor`: trie-backed prefix matching; O(prefix_length) lookup
- `LearningRanker`: count-based history boost at ranking time with configurable dominance bound
- `DecayRanker`: exponential recency decay; half-life configurable per instance
- `ScoreRanker`: pure score sort; deterministic, stable, idempotent
- `RankingExplanation`: per-suggestion score breakdown enforcing `final == base + boost` at construction
- `History`: append-only selection store with O(k) prefix index; all reads prefix-scoped
- `JsonHistoryStore`: full timestamp persistence (ISO 8601); v1 count-only migration support
- Five presets: `stateless`, `default`, `recency`, `production`, `robust`
- CLI: `suggest`, `explain`, `record`, `debug`, `presets` subcommands with history persistence
- 48,032-word English vocabulary (wordfreq-derived)
- 230 tests; 99% coverage; CI on Python 3.10–3.13

### Fixed

- `LearningRanker.rank()` now returns boosted `ScoredSuggestion` objects so
  downstream rankers and `predict_scored()` see post-learning scores
- `AutocompleteEngine.__init__` enforces shared-History identity across all
  learning rankers; mismatched instances raise `ValueError` at construction
- `TrigramPredictor` output is now deterministic across processes
- `TrigramPredictor` accepts optional `frequencies` for frequency-weighted
  tiebreaking - common words rank above rare ones at equal edit distance
- `JsonHistoryStore.save()` is now atomic (temp-file rename)
- Fixed memory figure in `FrequencyPredictor` docstring (16M → ~344k references)

### Design decisions recorded in README

- Prediction/ranking separation and why it came from a testing constraint
- Pre-ranking vs post-ranking explain bug and its regression test
- BK-tree O(n) degradation at scale and trigram index as the fix
- History O(n) scan → O(k) prefix index
- FrequencyPredictor pre-sorted index tradeoff (construction time vs query time)
