# Changelog

All notable changes to this project are documented here.

## [Unreleased]

## [0.3.0] - 2026-04-28

### Fixed (correctness)

- **`scripts/demo.py` displayed `freq=` and `recency=` column labels** after the
  CLI renamed them to `base=` and `boost=`. Demo output contradicted both the CLI
  and the README. Fixed: demo now uses the correct column names.

- **`scripts/demo.py` section 3 `final=` column truncated all non-boosted scores
  to `1`** due to `:5.0f` formatting rounding `0.96`, `0.92`, `0.85`, `0.74` up
  to `1`, making hero (0.74) look identical to help (1.0). Fixed to `:7.2f`.

- **`scripts/demo.py` bar chart was off by one block** due to floating-point
  decay: `DecayRanker` produces `boost=499.9999...` rather than exactly `500.0`.
  `int(499.999/50)` truncates to `9`, not `10`. Fixed with `round()` before
  integer division.

- **`scripts/demo.py` section 2 showed `score==base` and `boost==0` for every
  row** because `default` preset with no history produces no boost anywhere. The
  explain output demonstrated nothing. Fixed: section 2 now uses the `recency`
  preset with two recorded selections, so the boosted row is visible.

- **README `predict_scored` example showed wrong score** for `help` (`0.6544`
  instead of the actual `0.9522` for the custom engine with that vocabulary).

- **README `explain_as_dicts` example showed non-zero `history_boost` with no
  `record_selection()` call in the snippet.** Fixed: added the missing call and
  updated all comment values to match actual output.

- **README CI matrix overstated Windows coverage.** Said "Linux and Windows,
  Python 3.10, 3.11, 3.12, and 3.13" - CI only runs Python 3.11–3.12 on
  Windows. Fixed.

- **`SymSpellPredictor` was missing from the predictor contract test suite.**
  All six other predictors were tested against the shared `PredictorContractTestMixin`
  invariants; SymSpell was not. Added `TestSymSpellPredictorContract`.

- **`Makefile` and CI lint step excluded `scripts/`** - regressions in
  `scripts/demo.py` would not be caught by `make check` or CI. Fixed: both now
  include `scripts` in the `ruff check` invocation.

- **Score dimensional mismatch** (silent, high-impact): `FrequencyPredictor`
  previously emitted raw corpus counts as scores (e.g. 50,000 for "the"). `HistoryPredictor`
  emitted raw selection counts (e.g. 3). Typo predictors emitted scores in `[0, 1]`. Combined
  additively with weights, the weights on `HistoryPredictor` and typo predictors were
  effectively meaningless - a common-word frequency hit dominated by 4–5 orders of magnitude.
  Both predictors now emit **log-normalised scores in (0, 1]**:
  `score = log(1 + freq) / log(1 + max_freq)`. With all predictors in a common scale,
  `weight=1.5` on `HistoryPredictor` means what it says.

- **`LearningRanker` / `DecayRanker` cache invalidation bug**: both rankers cached history
  results keyed by prefix. If history was updated between two `rank()` calls on the same
  prefix (`record_selection()` then `suggest()`), the second `rank()` returned stale results.
  Learning was silently broken. Fixed: both rankers invalidate the cache at the start of
  every `rank()` call, then populate it for reuse within the same pipeline pass.

- **`explain()` base_components wrong with multiple predictors**: `base_components` in
  `explain_as_dicts()` was attributed entirely to the first predictor's name even when
  multiple predictors contributed. With `FrequencyPredictor(weight=1.0) +
  HistoryPredictor(weight=1.5)`, "help" showed `{"frequency": 2.069}` instead of
  `{"frequency": 0.569, "history": 1.5}`. Fixed by introducing `_score_with_breakdown()`
  which accumulates a structured `{predictor_name: weighted_score}` dict during aggregation.
  No string parsing required.

- **`explain()` base_score doubled in multi-ranker composition**: with `ScoreRanker +
  LearningRanker` both active, `base_score` in the merged explanation was doubled
  (e.g. 1.138 instead of 0.569) because `LearningRanker.explain()` claimed the base
  score as its own. The `RankingExplanation` invariant (`final == base + boost`) still held
  numerically, masking the bug. Fixed: `engine.explain()` now derives `base_score` directly
  from `_score_with_breakdown()` (ground truth from the predictor layer) and measures each
  ranker's contribution as a score delta (`post_rank - pre_rank`). Boost rankers
  (`LearningRanker`, `DecayRanker`) emit `base_score=0` from their own `explain()`.

- **Trace string parsing fragility**: the previous fix for `base_components` parsed trace
  strings of the form `"Predictor=NAME, weight=W, raw_score=S"`. Any predictor whose `name`
  contains `", "` would corrupt the parse and silently fall back to wrong attribution.
  Predictor names are arbitrary strings. Fixed by using the structured breakdown from
  `_score_with_breakdown()` directly - no string parsing.

- **`ThreadSafeHistory` GIL reliance**: the previous implementation relied on CPython's GIL
  making `list.append()` atomic for concurrent reads. Not guaranteed under PyPy or
  free-threaded CPython (PEP 703, Python 3.13+). Replaced with a reader-writer lock using
  `threading.Condition`. All read methods acquire the read-lock explicitly.

- **`History.snapshot()` misleading for persistence**: `snapshot()` returned a count-only
  dict (no timestamps). Using it for persistence would silently lose all recency signal.
  Added `snapshot_counts()` as an explicit alias. `snapshot()` now delegates to it with a
  deprecation notice.

- **`LearningRanker.explain()` re-entered `rank()`**: `explain()` called `self.rank()` to
  get ordered results, which invalidated the cache populated during the engine's upstream
  `rank()` call. Fixed: `explain()` replicates the sort key inline without calling `rank()`.

- **`explain_as_dicts()` incomplete schema**: previously returned only four fields. Now
  includes `sources` (all contributing ranker/predictor names), `base_components`
  (per-predictor weighted contributions), and `history_components` (per-ranker boost
  contributions). `base_components` values sum exactly to `base_score`.

- **`ThreadSafeHistory.__repr__`** inherited from `History`, returning `"History(...)"`.
  Added explicit `__repr__` returning `"ThreadSafeHistory(entries=N)"` under a read-lock.

- **`ScoreRanker.explain()` fallback label**: when `ScoredSuggestion.explanation` is `None`
  (manually constructed suggestions), `source` now correctly falls back to `"score"` rather
  than propagating `None`.

### Changed (breaking)

- **`FrequencyPredictor` and `HistoryPredictor` score values**: raw counts → log-normalised
  `(0, 1]`. Ordering preserved. Callers comparing `ScoredSuggestion.score` against hardcoded
  thresholds must update.

- **`WeightedPredictor` rejects `weight <= 0`**: raises `ValueError` at construction.
  Remove zero-weight predictors from the predictor list instead.

- **`ThreadSafeHistory.lock`** returns `threading.Condition` instead of `threading.Lock`.
  `with ts.lock:` still works. `isinstance(ts.lock, threading.Lock)` is now `False`.

- **`explain_as_dicts()` schema extended**: new keys `sources`, `base_components`,
  `history_components` added. Existing keys unchanged.

- **Ranker `explain()` contract**: `LearningRanker.explain()` and `DecayRanker.explain()`
  now emit `base_score=0.0` (boost-ranker contract). Use `engine.explain()` for full
  `base_score + history_boost` breakdowns.

### Added

- `History.snapshot_counts()`: explicit-name alias for `snapshot()`.
- `History.snapshot_history()` on `ThreadSafeHistory`: returns a consistent point-in-time
  plain `History` snapshot under a read-lock, for passing to `JsonHistoryStore.save()`
  without holding a lock during I/O.
- `JsonHistoryStore` now exported from the top-level `aac` package.
- `predict_scored()` documented in the README with an example.
- FastAPI example (`examples/fastapi_app.py`) now referenced in README Quick start section.
- Migration guide in README for 0.2.x → 0.3.0 breaking changes.
- `SymSpellPredictor` added to predictor contract test suite (`TestSymSpellPredictorContract`).
- **`aac history` CLI subcommand** - shows what the engine has learned.
  `aac history` prints a summary of all prefixes (total selections, top completion).
  `aac history <prefix>` shows per-value counts and recency ("5s ago", "2h ago")
  for that prefix, sorted by count descending. 5 tests.
- **`aac.vocabulary` module** with three utilities for building frequency vocabularies
  from common input formats, replacing the requirement to hand-construct `{word: int}` dicts:
  - `vocabulary_from_wordlist(words)` - plain word/phrase list, all words get equal weight.
  - `vocabulary_from_text(text)` - tokenise and count a free-form text corpus.
  - `vocabulary_from_file(path, format=)` - load from a file in either format.
  All three are exported from the top-level `aac` package. 25 tests.
- **`--vocab-path PATH` and `--vocab-format {wordlist,text}` CLI flags** - replace the
  bundled 48k English vocabulary with a user-supplied file. Example:
  `aac --vocab-path commands.txt suggest git`. Missing-file and empty-file errors are
  caught and reported cleanly. 4 CLI tests.
- Tests: score normalisation unit-interval invariant (Hypothesis, 200 examples each
  for `FrequencyPredictor`, `HistoryPredictor`, combined engine), `ThreadSafeHistory`
  concurrent write correctness and read visibility, `snapshot_counts()` consistency,
  `explain_as_dicts()` per-predictor value correctness, `LearningRanker` cache
  invalidation after `record()`, `ScoreRanker.explain()` with `explanation=None`,
  `predict_scored` / `explain` ordering consistency, `base_components` sum invariant.

### Performance

- `explain()` calls predictors once (via `_score_with_breakdown()`), not twice.
- `explain()` calls each ranker's `rank()` twice (once in `_apply_ranking()`, once in
  the delta-attribution loop). For history-scanning rankers (`LearningRanker`,
  `DecayRanker`), this means 2 history lookups per `explain()` vs 1 per `suggest()`.
  Acceptable given the typical call ratio (thousands of `suggest()` per `explain()`).
- Both `suggest()` and `explain()` are now benchmarked for all presets.
- `Makefile` and CI lint scope extended to include `scripts/`.

## [0.2.1] - 2026-02-20

### Fixed

- **`asyncio.get_event_loop()` replaced with `asyncio.get_running_loop()`** in
  all three async methods (`suggest_async`, `explain_async`,
  `record_selection_async`). `get_event_loop()` is deprecated in Python 3.10+
  when called with no running loop and can silently return or create a new loop
  in some contexts. `get_running_loop()` raises `RuntimeError` if called outside
  a running coroutine - making misuse explicit rather than silent - and
  guarantees it returns the actual running loop, not a potentially different one.
  `asyncio` is now imported at module level rather than inside each method body.

- **`FrequencyPredictor.add_word` complexity documented accurately**: the
  docstring previously stated `O(len(word)) per call`. The actual complexity is
  O(B) per prefix bucket, where B is the number of words in that bucket, because
  each bucket must be linearly scanned to find the correct insertion position. The
  previous implementation also allocated a fresh `[-freq for w in bucket]` list
  on every call to use with `bisect`, paying O(B) time *and* O(B) space. Replaced
  with a direct linear scan (zero allocations, same asymptotic cost, honest
  documentation). The docstring now accurately states when this matters and when
  to use an alternative.

- **`import math` moved to module level** in `symspell.py`. Previously placed
  inside `__init__`, causing a module import on every `SymSpellPredictor`
  construction. Standard library imports belong at the top of the file.
  `import asyncio` similarly moved to module level in `engine.py`.

- **`examples/fastapi_app.py` type annotations corrected**: `_engine_instance`
  was typed as `None` at module level with a `# type: ignore[return]` suppressing
  a genuine mypy error on `get_engine()`. Now typed as
  `AutocompleteEngine | None = None` with a proper `-> AutocompleteEngine` return
  annotation. `lifespan()` now has the correct `-> AsyncGenerator[None, None]`
  return type rather than `# type: ignore[type-arg]`. Unused imports
  (`HTTPException`, `JSONResponse`) removed.

- **`test_no_false_negatives_exhaustive_small_vocab` and
  `test_no_false_positives_exhaustive_small_vocab` had a logical gap**: two
  separate `issubset` checks do not prove set equality - a bug returning all
  correct words *plus* extra wrong words passes the false-negative check, and a
  bug returning only a subset passes the false-positive check. Replaced with a
  single `test_exact_set_equality_exhaustive_small_vocab` that uses `==` and
  also verifies score-distance consistency: every returned word's score must
  equal `base_score / (1 + actual_levenshtein_distance)`, catching bugs where
  the right word is found but its distance is recorded incorrectly.

### Tests

- **`tests/predictors/test_symspell_predictor.py`** added: 21 tests covering the
  correctness claim that `SymSpellPredictor` returns exactly the same word set as
  a brute-force linear scan at both `max_distance=1` and `max_distance=2`, over
  both a small hand-curated vocabulary and a 500-word medium vocabulary. Includes
  an exhaustive exact-set-equality check with score-distance consistency
  verification, and specific threat cases for the delete-neighbourhood algorithm:
  transpositions, 1–2 character queries, exact matches, near-duplicate
  vocabularies, and queries longer than all vocabulary words.

  `SymSpellPredictor` claimed "100% recall, no false negatives" in the docstring,
  README, and CHANGELOG. This file is the test that proves that claim. The
  `BKTree` has had an equivalent brute-force correctness test since 0.1.0;
  SymSpell now has one too.

## [0.2.0] - 2026-02-15

### Added

- **`SymSpellPredictor`** (`aac.predictors.symspell`): delete-neighbourhood
  approximate matcher. O(1) average query time at any vocabulary size;
  ~400µs/call at 48k words - 150× faster than BK-tree. Works on 1–3
  character prefixes (unlike TrigramPredictor which requires ≥4). Exported
  from `aac.predictors`.

- **`robust` preset now uses SymSpell**: the `robust` preset was previously
  an BK-tree that degraded to O(n) at 48k+ words (~60ms/call). It now uses
  `SymSpellPredictor` and runs at ~400µs/call with full recall at any scale.
  The old BK-tree implementation is retained as the `bktree` preset for
  benchmarking/comparison.

- **`bktree` preset**: the legacy BK-tree engine extracted to its own named
  preset. Use for benchmarking. Not recommended for production.

- **`suggest(limit=N)` parameter**: `engine.suggest("he", limit=10)` avoids
  constructing the full candidate list when only the top-N are needed.
  Previous pattern `engine.suggest("he")[:10]` still works.

- **Async API**: `suggest_async()`, `explain_async()`, and
  `record_selection_async()` run the synchronous methods in a thread pool
  executor so async web frameworks (FastAPI, aiohttp) are not blocked.

- **`FrequencyPredictor.add_word(word, frequency)`**: update the vocabulary
  of a running engine without rebuilding the index. O(B) per prefix bucket
  where B is the bucket size (see 0.2.1 for corrected complexity documentation).
  Enables domain-specific vocabulary growth at runtime.

- **`ThreadSafeHistory`** (`aac.domain.thread_safe_history`, exported from
  `aac`): a `History` subclass that serialises `record()` calls with a
  threading lock. Safe for concurrent use in multi-threaded web servers.
  Includes `snapshot_history()` for lock-free serialisation and a `lock`
  property for compound atomic operations.

- **`create_engine(history=...)` parameter**: attach a loaded `History` (or
  `ThreadSafeHistory`) directly via `create_engine()`. Previously required
  `get_preset(name).build(history, vocab)` - an internal API not shown in
  the README.

- **`examples/fastapi_app.py`**: complete FastAPI integration showing
  persistent history, async endpoints, thread-safe recording, and graceful
  shutdown. Run with `uvicorn examples.fastapi_app:app`.

- **Persistence documented in README**: the Python API section now shows a
  complete `JsonHistoryStore` + `create_engine(history=...)` pattern.
  Previously the README said "persistence-ready" but showed no code for it.

### Changed

- **`Suggestion` removed from `aac.__all__`**: `suggest()` returns
  `list[str]`; `Suggestion` is internal. Importing it from `aac` no longer
  works - import from `aac.domain.types` directly if needed.

- **Preset CLI choices auto-expand**: the `--preset` argument now uses
  `available_presets()` dynamically, so new presets (`bktree`) appear
  automatically without CLI changes.

- **Benchmark updated**: `make benchmark` now includes `robust` (SymSpell)
  and `bktree` (legacy) side-by-side for direct comparison.

## [0.1.7] - 2026-01-27

### Changed

- **`Suggestion` removed from `aac.__all__`**: `suggest()` now returns
  `list[str]`; `Suggestion` is an internal scoring type that no caller needs
  to import. Removed from the top-level package exports to avoid confusion.

### Fixed

- **`create_engine()` now accepts a `history` parameter**: previously,
  the only way to attach a loaded `History` to a preset engine was
  `get_preset(name).build(history, vocab)` - an internal API not shown in
  the README. `create_engine("production", history=store.load())` now works
  directly. Default is `None` (fresh in-memory history), so existing code
  is unaffected.
- **README Python API section now shows persistence**: the previous version
  said "persistence-ready" but showed no persistence code and never mentioned
  `JsonHistoryStore`. A developer using the README example would build an
  engine whose learning evaporated on every restart with no obvious fix.
  A complete persistence example is now in the Python API section.

## [0.1.6] - 2026-01-26

### Changed

- **`AutocompleteEngine.suggest()` now returns `list[str]` instead of
  `list[Suggestion]`** (breaking change for callers using `.value`): `Suggestion`
  wraps exactly one field and adds no public information over a plain string. The
  previous return type required every caller to write `[s.value for s in
  engine.suggest(text)]` and made natural usage patterns fail silently:
  `"hello" in engine.suggest("helo")` returned `False`, `sorted(engine.suggest(...))`
  raised `TypeError`, and `print(engine.suggest(...))` showed
  `[Suggestion(value='help'), ...]`. All callsites in source, tests, scripts, and
  documentation updated. `Suggestion` remains as an internal type used by
  `predict_scored()` and `ScoredSuggestion`.

### Added

- **`RankingExplanation.__repr__`**: default dataclass repr showed full-precision
  floats and empty internal dicts (`base_components={}`, `history_components={}`),
  making `engine.explain("helo")[0]` unreadable in a REPL or log. New repr:
  `RankingExplanation(value='hello', base=1.4063, boost=+1.5000, final=2.9063)`.
  Regression test added.

## [0.1.5] - 2026-01-25

### Fixed

- **`aac explain` recency column format inconsistency**: the CLI displayed
  `recency= 0.00` (space-prefixed zero) while `scripts/demo.py` and the README
  both showed `recency=+0.00` (explicit sign). Any user comparing CLI output to
  the README would see a character difference on every line with zero boost.
  Fixed by replacing the manual `if boost > 0` branch with Python's `:+.2f`
  format specifier, which always shows the sign - simpler code, consistent
  output. Regression test added (`test_explain_recency_column_always_shows_sign`).

## [0.1.4] - 2026-01-24

### Fixed

- **`BrokenPipeError` when piping CLI output** (`aac suggest he | head -5`): all
  output-producing subcommands (`suggest`, `explain`, `debug`) raised an unhandled
  `BrokenPipeError` traceback to stderr when the pipe consumer closed early. This
  is the single most common Unix CLI usage pattern and the traceback was the first
  thing many users would encounter. Fixed by extracting the dispatch logic into
  `_run()` and wrapping the call in `main()` with `except BrokenPipeError:
  sys.exit(0)`. Regression test added.

### Added

- **Complete argparse help strings for all CLI arguments**: every positional
  argument and option now has a descriptive help string with a concrete example.
  Previously `aac suggest --help` showed `text` with no description, `aac record
  --help` showed `text` and `value` with no description, and `--limit` had no
  default shown. Users had to read the README or source to understand basic usage.

## [0.1.3] - 2026-01-23

### Fixed

- **`MagicMock` imported after use in `test_no_fd_double_close_when_write_raises_after_fdopen`**:
  `MagicMock` was used inside a nested function defined before the import statement.
  This worked at runtime (Python closures resolve names at call time) but was
  misleading and fragile. `MagicMock` and `builtins` are now imported at the top
  of the test method in proper order.
- **`pyproject.toml` keyword `"machine-learning"` removed**: this library does no
  machine learning - it uses algorithmic ranking (frequency counts, edit distance,
  trigram indexing, exponential decay). Replaced with `"information-retrieval"`.
- **Integration tests now carry `@pytest.mark.integration`**: the module docstring
  previously claimed tests were "marked with pytest's integration marker" but no
  marker existed. The marker is now registered in `pyproject.toml` and applied to
  the `TestCliRoundTrip` class. `make test-fast` runs unit tests only (~30s);
  `make test` runs everything (~90s).
- **CI now skips integration tests on push, runs them on pull requests**: every
  push to `main` runs the unit suite only (fast feedback). Pull requests run the
  full suite including integration tests (complete verification before merge).
- **`make test-fast` target added** to Makefile for running unit tests only.
- **Redundant `-q` flag removed from `make test`**: `pyproject.toml` already
  passes `-q` via `addopts`; the Makefile was passing it a second time.

## [0.1.2] - 2026-01-22

### Added

- `aac presets --json` flag outputs all preset metadata as a JSON array for
  programmatic consumption (tooling, scripts, editor integrations).
- `ScoreRanker`, `DecayFunction`, `DecayRanker`, `LearningRanker`, and
  `RankingExplanation` are now exported from `aac.ranking.__init__` with a
  proper `__all__`. Previously only `Ranker` and `ScoreRanker` were listed;
  the others were importable but undiscoverable by IDEs and type checkers.
- `HistoryStore` added to `aac.storage.__all__`. Previously only
  `JsonHistoryStore` was exported.
- `SECURITY.md`: security vulnerability reporting instructions.
- CLI round-trip integration tests (`tests/test_cli_integration.py`): seven
  subprocess-based tests covering history file creation, JSON schema validation,
  learning signal (recorded word rises above equal-frequency unrecorded peer),
  cross-process history persistence via pre-written file, `presets` text/JSON
  output, and typo recovery. These exercise the full stack end-to-end in a way
  that unit tests cannot.
- Two new `TestJsonStoreExceptionCleanup` tests verifying the fd ownership
  semantics of the Windows temp-file cleanup fix: one confirms `os.close(fd)`
  is called when `fdopen` raises (fd not yet owned), the other confirms it is
  **not** called when the write raises after `fdopen` succeeds (double-close
  would be undefined behaviour).
- `make run ARGS="..."` Makefile target as a convenience wrapper for
  `poetry run aac ...`, so `aac` works without manually activating the venv.
- `make install` now prints venv activation instructions.
- CI matrix extended to `windows-latest` (Python 3.11 and 3.12). The Windows
  fd-cleanup bug was caught by a test but not by CI because CI was Linux-only.
  `fail-fast: false` added so all matrix jobs complete even if one fails.

### Fixed

- **Windows temp-file orphan bug** (`JsonHistoryStore.save()`): when
  `os.fdopen(fd)` raised before taking ownership of the raw fd (e.g. the test
  patches it to raise `OSError("disk full")`), the fd remained open. On
  Windows, open files cannot be `unlink()`-ed - the OS holds a mandatory lock
  - so `os.unlink(tmp_path_str)` silently failed and the `.tmp` file was
  orphaned. Fixed by tracking `fd_owned_by_file`; the `except` block now calls
  `os.close(fd)` before `os.unlink()` when `fdopen` did not take ownership.
  The previously failing test `test_temp_file_cleaned_up_on_write_failure` now
  passes. This was a real cross-platform correctness bug, not just a test
  artifact.
- **`ScoreRanker.rank()` was not stable on tied scores**: used `sorted(...,
  key=lambda s: s.score, reverse=True)` with no tiebreaker, so equal-score
  suggestions had non-deterministic relative order depending on dict insertion
  order from `_score()`. Changed to `(-score, original_index)` matching
  `LearningRanker` and `DecayRanker`'s sort contract, making all three rankers
  composable with deterministic output.
- **`HistoryPredictor.predict()` was non-deterministic on equal selection
  counts**: iterated `dict.items()` directly, whose order depends on
  `History.record()` call order - effectively session-dependent for equal-count
  words. Now sorts by `(-count, value)` before emitting results.
- **`FrequencyPredictor` indexed zero-frequency words**: words with `freq <= 0`
  were added to the prefix index and emitted as `score=0.0` candidates,
  polluting output with no-signal entries. Now filtered at index build time.
- **`_levenshtein` private alias removed** from `bk_tree.py`. The comment
  claimed backward compatibility for a `_`-prefixed name - nothing external
  can legitimately depend on a private symbol. Dead code removed.
- **`TrigramIndex._size` attribute removed**: computed at construction, never
  read anywhere in the codebase. Dead code removed.
- **README quick-start block** correctly documents venv activation after
  `make install`. The previous version told users to run `aac suggest he`
  immediately after `make install`, which fails on all platforms until the
  virtualenv is activated or `poetry run` is used.
- **README test count** updated from 261 to 271 (three new unit tests and seven
  integration tests added).
- **README CI description** updated to reflect Windows runners.
- **`Development Status` PyPI classifier** changed from `3 - Alpha` to
  `4 - Beta`. The package has 99%+ coverage, full type annotations, a complete
  CHANGELOG, property-based tests, CI on four Python versions and two
  platforms, and a published release. "Alpha" misstates the maturity.

## [0.1.1] - 2026-01-20

### Added

- **Coverage: 99.58%** (up from 98.32%). Targeted tests for all previously
  uncovered branches: divergent-History ``ValueError`` in ``AutocompleteEngine``
  (both construction paths), non-finite score detection, ``_predict_scored_unranked``,
  ``FrequencyPredictor`` validation, ``Trie._collect`` limit exit, ``TrigramIndex``
  empty-string guard and length-difference pruning, ``TrigramPredictor`` empty-prefix
  guard, and ``JsonHistoryStore`` exception cleanup on write failure.
- Property-based tests with Hypothesis covering four core invariants.
  31 new tests; 261 total.
  - `RankingExplanation` arithmetic (`final_score == base_score + history_boost`)
    holds for all finite non-negative score combinations, including after
    `merge()` and `apply_history_boost()`
  - `LearningRanker` and `DecayRanker` never add or remove candidates, across
    arbitrary suggestion lists and history states
  - History prefix index always agrees with brute-force full scan, regardless
    of insertion order or prefix distribution
  - `apply_history_boost()` invariant preserved after three-way float sum fix
- `hypothesis = "^6.100"` added to dev dependencies.
- ``scripts/demo.py``: self-contained end-to-end demonstration using a
  controlled vocabulary so learning is visible after five selections.
- CI benchmark artifact: benchmark runs on Python 3.12 and uploads results
  as a GitHub Actions artifact, making the README performance table
  independently verifiable.

### Fixed

- **`test_apply_history_boost_preserves_invariant` was not executing**: the test
  was appended as a module-level function with ``self`` as its first parameter.
  pytest does not collect module-level functions with ``self`` - they require a
  class. The test appeared in the source file and in the changelog but silently
  never ran. The entire ``test_property_based.py`` file was rewritten as three
  clean, properly-structured classes; all 12 Hypothesis tests now execute.
- **`record_selection()` used wrong history key** (silent correctness bug):
  ``self._history.record(ctx.text, value)`` recorded selections under the
  raw input string, but ``counts_for_prefix()`` and ``entries_for_prefix()``
  look up by ``ctx.prefix()`` - the normalised, lowercased, last-word form.
  Keys never matched, silently disabling the entire learning system for
  callers using the public ``record_selection()`` API.  Fixed by recording
  under ``ctx.prefix()``.
- **`DecayRanker.rank()` was not stable** and dropped trace entries:
  sorted on ``score`` alone (no tiebreaker), so equal-score suggestions had
  non-deterministic relative order.  Changed to ``(-score, original_index)``
  matching ``LearningRanker``'s contract.  Also: ``DecayRanker`` was
  passing through ``s.trace`` unchanged even when it applied a non-zero
  boost, making ``debug()`` output silent about its contribution.  Now
  appends a ``"DecayRanker boost=..."`` trace entry when ``boost > 0``.
- **`FrequencyPredictor` indexed exact matches unnecessarily**: the prefix
  index was built with ``range(1, len(word) + 1)``, adding every word under
  its own full string as a key.  ``predict()`` then filtered these out per
  call with an ``if word == prefix: continue`` guard.  For a 48k vocabulary
  this added ~48k wasted entries to the index and one comparison per
  candidate per query.  Fixed by using ``range(1, len(word))`` at build
  time; the query-time guard is now unnecessary and removed.
- **`RankingExplanation.apply_history_boost()` floating-point precision bug**
  (also found by Hypothesis): same three-way vs two-way float sum issue as
  ``merge()``.  ``final_score = base + old_boost + new_boost`` rounds
  differently from ``base + (old_boost + new_boost)`` at magnitudes above
  ~1e9.  Fixed by computing ``new_history_boost = old + boost`` first, then
  ``final_score = base + new_history_boost``.
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
- **CI**: added benchmark step (Python 3.12 only) that uploads results as a
  GitHub Actions artifact, making the README performance table independently
  verifiable without cloning.

## [0.1.0] - 2026-01-17

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
