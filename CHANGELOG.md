# Changelog

All notable changes to this project are documented here.

## [0.1.0] - 2026

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
- 223 tests; 99% coverage; CI on Python 3.10–3.13

### Design decisions recorded in README

- Prediction/ranking separation and why it came from a testing constraint
- Pre-ranking vs post-ranking explain bug and its regression test
- BK-tree O(n) degradation at scale and trigram index as the fix
- History O(n) scan → O(k) prefix index
- FrequencyPredictor pre-sorted index tradeoff (construction time vs query time)
