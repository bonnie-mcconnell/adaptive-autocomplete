# adaptive-autocomplete

[![CI](https://github.com/bonnie-mcconnell/adaptive-autocomplete/actions/workflows/ci.yml/badge.svg)](https://github.com/bonnie-mcconnell/adaptive-autocomplete/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![mypy](https://img.shields.io/badge/mypy-strict-blue)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

An autocomplete engine that learns from user selections, recovers typos, and can explain every ranking decision as plain numbers.

```python
from aac.presets import create_engine

engine = create_engine("production")

engine.suggest("programing")
# → ['programming']   # SymSpell finds the 1-deletion correction

engine.record_selection("programing", "programming")
engine.suggest("programing")
# → ['programming']   # same result, now history-boosted to the top

engine.explain("programing")[0]
# RankingExplanation(value='programming', base=1.65, boost=+1.50, final=3.15)
```

---

## How it works

**Prediction and ranking are separate layers.** Predictors are read-only with respect to history: they take a prefix and return scored candidates but never write to `History`. Rankers are read-write: they reorder candidates based on history and recency, and the engine records selections back into `History`. The seam exists because the original monolithic function was untestable - writing a learning test required mocking the full prediction pipeline.

```
text input
    ↓
CompletionContext   - lowercases and normalises prefix
    ↓
Predictors         - frequency (stateless), history (read-only), symspell (stateless), trigram (stateless)
    ↓
Weighted sum       - additive, configurable per predictor
    ↓
Rankers            - reorder and boost; cannot add or remove candidates
    ↓
Suggestions + explanations
```

**`explain()` costs the same as `suggest()`** - one pipeline run. Score deltas are captured as each ranker runs rather than by re-running the pipeline after the fact.

**`RankingExplanation.__post_init__` enforces `final_score == base_score + history_boost`.** The invariant is checked at object construction. If a ranker produces an inconsistent explanation, you get a `ValueError` immediately, not a silent wrong answer downstream.

**Rankers cannot modify the candidate set.** After each ranker step, the engine checks that suggestions-in equals suggestions-out. Violations raise `RuntimeError` naming the offender.

**History has one owner.** The engine holds the `History` reference. `reset_history()` propagates the new instance to all rankers and predictors via typed protocols (`LearnsFromHistory`, `PredictorLearnsFromHistory`). Diverged history references are caught at construction time.

---

## Typo recovery

Three approximate-match predictors, all sharing the same scoring formula (defined in the internal `predictors/_scoring.py` module, not part of the public API):

```
score = (base / (1 + edit_distance)) * (1 + 0.5 * log_freq_score)
```

Distance is dominant. The frequency multiplier orders words within each distance bucket but can't promote a distance-2 match above a distance-1 match. The 0.5 weight was chosen so scores are directly comparable across predictors - a `SymSpellPredictor` result and a `TrigramPredictor` result at the same distance and frequency produce the same number.

**SymSpell** builds a delete-neighbourhood index at startup: for each dictionary word, every variant reachable by 1–2 deletions is stored. Lookups hash the query's delete-neighbourhood and intersect with the stored set. O(1) average, no BK-tree traversal.

**AdaptiveSymSpell** switches to a shorter index for short prefixes (≤3 chars). The full 2-deletion index on a 2-character query generates hundreds of candidates, many nonsensical. The short-prefix variant uses max_distance=1.

**Trigram** builds a character 3-gram inverted index. Jaccard similarity over shared trigrams filters candidates before computing exact edit distance, so the expensive Levenshtein calculation only runs on plausible matches.

---

## Weight optimisation

```python
from aac.evaluation import EvaluationHarness, WeightOptimiser
from aac.evaluation.datasets import make_synthetic_query_log
from aac.data import load_english_frequencies

vocab = list(load_english_frequencies().keys())
query_log = make_synthetic_query_log(vocab, prefix_lengths=[2, 3, 4])

harness = EvaluationHarness(query_log)
opt = WeightOptimiser(harness, verbose=False)

result = opt.coordinate_descent(
    "production",
    weight_grid={"frequency": [0.5, 1.0, 2.0], "symspell": [0.3, 0.5, 1.0]},
)
print(result.best_weights)
# → {'frequency': 2.0, 'symspell': 0.3, 'history': 1.2, 'trigram': 0.4}
# (exact values depend on vocabulary and query distribution)
```

Predictor indexes are built once per preset and cached. Only weight wrappers are recreated per evaluation. For the production preset (SymSpell takes ~5s to build), this reduces 27 evaluations from ~135s to ~0.05s.

Two strategies: **grid search** (exhaustive, optimal on the grid, practical for ≤3 predictors × ≤4 values) and **coordinate descent** (one weight at a time, converges faster, may find a local optimum).

---

## Concurrency

`History` is not thread-safe. For multi-threaded servers, pass `thread_safe=True`:

```python
from pathlib import Path
from aac.presets import create_engine
from aac.storage.json_store import JsonHistoryStore

store = JsonHistoryStore(Path.home() / ".aac_history.json")
engine = create_engine("production", history=store.load(), thread_safe=True)
# engine.history is now ThreadSafeHistory - safe from any number of threads.
```

Or construct `ThreadSafeHistory` directly if you need the reference before building the engine:

```python
from pathlib import Path
from aac.domain.thread_safe_history import ThreadSafeHistory
from aac.storage.json_store import JsonHistoryStore

store = JsonHistoryStore(Path.home() / ".aac_history.json")
history = ThreadSafeHistory(store.load())
engine = create_engine("production", history=history)
```

`ThreadSafeHistory` uses a `threading.Condition` for write serialisation and reference-counted readers for read concurrency. Multiple `suggest()` calls run simultaneously; `record_selection()` waits for active readers to finish. Does not rely on GIL guarantees - safe on CPython, PyPy, and free-threaded 3.13+.

---

## Quick start

```bash
git clone https://github.com/bonnie-mcconnell/adaptive-autocomplete
cd adaptive-autocomplete
make install

source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows PowerShell

make demo          # interactive browser demo (first run takes ~5s while SymSpell index builds)
make test-fast     # fast unit tests (skips integration, property-based, and perf)
make test          # full suite including integration and property-based tests
make benchmark     # latency numbers
```

Or with Docker (no Python install required):

```bash
make demo-docker   # opens at http://localhost:5000
```

---

## Presets

| Preset | Predictors | Ranker | Use when |
|---|---|---|---|
| `stateless` | frequency | score | no history, fast |
| `default` | frequency, history | score | history-aware, no decay |
| `recency` | frequency, history | score + decay | recency matters |
| `production` | frequency, history, symspell, trigram | score + decay | typos + recency |
| `robust` | frequency, history, symspell | score + decay | typo recovery only |

```python
from aac.presets import create_engine, compare_presets

engine = create_engine("production")

# Side-by-side comparison across presets
comparison = compare_presets("programing", presets=["default", "production"])
```

---

## Tests

727 tests: invariant correctness, IR metrics, evaluation harness, concurrency, async API, CLI integration, and property-based fuzzing with Hypothesis.

Key coverage:

- **SymSpell brute-force equivalence**: every result matches a linear scan exactly, across multiple queries and distance thresholds. Catches index bugs invisible to example-based tests.
- **Explain single-pass**: `ranker.rank()` is called exactly once per `explain()` call, verified with a spy. A structural test, not an output test.
- **Ranker invariant**: `RuntimeError` if a ranker adds or removes candidates.
- **History isolation**: `compare_presets()` does not modify the caller's History.
- **Property-based (Hypothesis)**: four core invariants across thousands of generated inputs. Found two floating-point precision bugs that example-based tests missed.

CI runs on Ubuntu and Windows across Python 3.10–3.13.

---

## Project layout

```
src/aac/
├── engine/         - AutocompleteEngine, EngineConfig (serialisation + diff)
├── domain/         - History, ThreadSafeHistory, CompletionContext
├── predictors/     - FrequencyPredictor, HistoryPredictor, SymSpell, Trigram, EditDistance
├── ranking/        - ScoreRanker, LearningRanker, DecayRanker, RankingExplanation
├── storage/        - JsonHistoryStore (atomic write, v1→v2 migration)
├── evaluation/     - EvaluationHarness, WeightOptimiser, IR metrics (MRR, NDCG, AP)
├── benchmarks/     - latency benchmarks with baseline comparison
├── cli/            - aac suggest / explain / record / history / demo
└── presets.py      - named engine configurations
```

---

## Further reading

- [`DESIGN.md`](DESIGN.md) - architecture decisions, tradeoffs, and what I'd change
- [`BENCHMARK.md`](BENCHMARK.md) - latency numbers, CI gates, and how to reproduce
- [`CHANGELOG.md`](CHANGELOG.md) - what changed and why
- [`examples/`](examples/) - usage examples including async FastAPI integration
