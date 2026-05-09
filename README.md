# adaptive-autocomplete

[![CI](https://github.com/bonnie-mcconnell/adaptive-autocomplete/actions/workflows/ci.yml/badge.svg)](https://github.com/bonnie-mcconnell/adaptive-autocomplete/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![mypy](https://img.shields.io/badge/mypy-strict-blue)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

An autocomplete engine that learns from user selections, recovers typos, and can explain every ranking decision in plain numbers.

```python
from aac.presets import create_engine

engine = create_engine("production")

engine.suggest("programing")
# → ['programming', 'program', 'programs', ...]

engine.record_selection("programing", "programming")

engine.suggest("programing")
# → ['programming', ...]   # boosted by history

engine.explain("programing")[0]
# RankingExplanation(value='programming', base=1.65, boost=+1.50, final=3.15)

engine.suggest_with_confidence("prog", limit=5)
# [('program', 1.0), ('programs', 0.89), ('progress', 0.85), ('programme', 0.79), ...]
```

The differentiating feature is `explain()`. Every suggestion comes with a mathematically auditable score breakdown: which predictor contributed what, how much the ranker boosted it, and the relative contribution of each source as a percentage of the final score. `final_score == base_score + history_boost` is enforced as an object-level invariant - it cannot be violated.

---

## Why I built this

I wanted to understand how autocomplete actually works - not at the "it uses machine learning" level, but at the "here is the exact function that decides why 'programming' ranks above 'program' for your specific query" level. Every library I found either abstracted away the ranking logic or required me to bring a pre-trained model. I wanted to build the ranking from first principles and be able to explain every number.

The second reason was the explainability constraint. Most ranking systems are black boxes: you can see the output order but not why. I wanted a system where you could point at any suggestion and ask "why is this here?" and get a precise mathematical answer - not a narrative, a breakdown of numbers that add up correctly. The `RankingExplanation.__post_init__` invariant (`final_score == base_score + history_boost`) is the mechanism that enforces this: if you can't explain the score, you can't construct the object.

The third reason was the zero-dependency constraint. I wanted to implement SymSpell, BK-trees, and trigram indexing from scratch rather than wrapping a library, because that's the only way to fully understand the trade-offs. `AdaptiveSymSpellPredictor` (two indexes dispatched by prefix length) came directly from debugging the single-index version and watching it return hundreds of low-quality candidates for 1-character prefixes.



```bash
pip install adaptive-autocomplete
```

Requires Python 3.10+. No required external dependencies.

---

## Interactive demo

```bash
aac demo
```

Opens a local browser UI showing live suggestions with confidence bars, per-suggestion score breakdowns, selection recording with immediate visible ranking shifts, and a side-by-side preset comparison table. No account, no API key, no internet required.

---

## Quick start

**CLI:**

```bash
aac suggest prog                              # ranked by frequency + learning
aac suggest programing                        # typo recovery → programming
aac suggest prog --preset stateless           # no learning, frequency only
aac suggest prog --confidence                 # show confidence bars
aac suggest prog --json                       # pipe-friendly JSON output
aac explain prog                              # score breakdown per suggestion
aac explain prog --json                       # full per-predictor JSON breakdown
aac compare recieve                           # side-by-side across all presets
aac compare recieve --presets stateless production  # specific presets
aac batch prog hel wor                        # batch suggestions (JSON)
aac batch prog hel wor --limit 5             # limit per prefix
aac record prog programming                   # record a selection; engine learns
aac history prog                              # what the engine has learned
aac presets                                   # list all presets
aac demo                                      # interactive browser UI
```

History persists automatically to `~/.aac_history.json`.

**Python API:**

```python
from aac.presets import create_engine

engine = create_engine("production")     # 48k vocabulary, typo recovery, learning
engine.suggest("helo")                   # → ['help', 'held', 'hell', 'hello', ...]
engine.record_selection("helo", "hello")
engine.explain("helo")                   # → [RankingExplanation(...), ...]
```

---

## What it does

### Typo recovery

The `production` preset uses a hybrid strategy covering all prefix lengths:

- **1–3 character prefixes**: `AdaptiveSymSpellPredictor` at `max_distance=1`. Raw SymSpell at distance 2 on a 2-char prefix like `"he"` matches hundreds of candidates - almost every short word in the vocabulary. Distance 1 catches single-char typos while keeping results clean.
- **4+ character prefixes**: `TrigramPredictor` at `max_distance=2`. Trigram overlap pre-filters to ~20–100 candidates before Levenshtein, giving ~600µs/call at 48k words vs ~60ms for BK-tree at the same distance.
- Both run at all lengths and combine additively.

```
aac suggest programing   → programming          (distance 2 - missing 'm')
aac suggest recieve      → recieved, relieve, believe, receive
aac suggest helo         → help, held, hell, hello
aac suggest he           → her, hey, here, help (clean, not hundreds of noise)
```

### Learning and recency

`record_selection()` writes to an append-only history. `HistoryPredictor` scores candidates by selection frequency. `DecayRanker` applies an exponential half-life so recent selections outweigh old ones - selecting something yesterday matters more than selecting it 50 times six months ago.

```
aac record he hello
aac record he hello
aac record he hello
aac suggest he      → hello now leads
```

### Explained rankings

`explain()` returns one `RankingExplanation` per suggestion:

```
$ aac record he hello && aac record he hello && aac explain he

hello          score=     4.67 (100.0%)  base=     1.67  boost=+3.00
her            score=     0.99 ( 21.2%)  base=     0.99  boost=+0.00
```

Every breakdown shows `base_score` (sum of predictor contributions), `history_boost` (ranker adjustment), `final_score`, and `contribution_pct` (each source as a fraction of final). The invariant `final_score == base_score + history_boost` is enforced in `RankingExplanation.__post_init__` - there is no code path that can violate it.

`base_components` always includes every configured predictor with an explicit score - `0.0` means "predictor ran but this word was below its threshold", never "predictor not configured". This distinction matters for debugging and for tuning weights.

```python
engine.explain("he")[0]
# RankingExplanation(value='hello', base=1.67, boost=+3.00, final=4.67)
# base_components:    {'frequency': 0.47, 'history': 1.20, 'symspell': 0.0, 'trigram': 0.0}
# history_components: {'decay': 3.00}
# contribution_pct:   {'frequency': 0.10, 'history': 0.26, 'decay': 0.64}
```

The `contribution_pct` field makes weight-tuning direct: "decay is contributing 47% of the final score - do I want learning to dominate this much?"

### Confidence scores

```python
# After recording 3 selections of 'programming':
for _ in range(3):
    engine.record_selection("prog", "programming")

for word, conf in engine.suggest_with_confidence("prog", limit=5):
    label = "★" if conf > 0.8 else " "
    print(f"{label} {word:<20} {conf:.0%}")
# ★ programming          100%
#   program               71%
#   programs              56%
```

### Selection counts

```python
for word, count in engine.suggest_with_history("prog", limit=5):
    badge = f"({count})" if count > 0 else ""
    print(f"{word} {badge}")
# programming (3)
# program
# programs
```

### Contextual learning

`ContextualHistory` partitions learning by domain. A selection in "shell" context cannot influence rankings in "search" context - they are independent History instances sharing no state.

```python
from aac.domain.contextual_history import ContextualHistory
from aac.presets import create_engine

ctx = ContextualHistory()

ctx.record("prog", "programming", domain="python")
ctx.record("prog", "progress",    domain="pm")

python_engine = create_engine("production", history=ctx.for_domain("python"))
pm_engine     = create_engine("production", history=ctx.for_domain("pm"))

python_engine.suggest("prog")   # → ['programming', ...]
pm_engine.suggest("prog")       # → ['progress', ...]

# Persist per-domain
from aac.storage.json_store import JsonHistoryStore
for domain, hist in ctx.domains():
    JsonHistoryStore(path / f"{domain}.json").save(hist)
```

### Preset comparison

`compare_presets()` runs `explain()` across multiple engines simultaneously and returns side-by-side score breakdowns. Each engine gets an independent History copy so comparison is always consistent. Engines are cached after first build so repeated calls are instant.

```python
from aac.presets import compare_presets, warm_cache

# Optional: pre-build all engines at startup (takes ~8s, then cached)
warm_cache()

cmp = compare_presets("recieve")                                   # all presets
cmp = compare_presets("recieve", presets=["stateless", "production"])  # specific
print(cmp.to_table())
```

Or from the CLI:

```bash
aac compare recieve
aac compare recieve --presets stateless production
aac compare recieve --json | jq '.rows[0]'
```

```
suggestion         stateless                    production
                rank    base   boost   final   rank    base   boost   final
---------------------------------------------------------------------------
recieved          #1   0.147  +0.000   0.147    #1   0.550  +0.000   0.550
relieve            -       -       -       -     #2   0.202  +0.000   0.202
believe            -       -       -       -     #3   0.152  +0.000   0.152
receive            -       -       -       -     #4   0.146  +0.000   0.146
```

`stateless` returns only the misspelling (`recieved` is in the corpus with a small count). `production` surfaces more candidates via SymSpell and trigram matching, making the ranking recoverable once you record a selection. The table shows exactly why: without typo-recovery predictors, `stateless` cannot surface `receive` at all.

### Serialisable engine config

```python
from aac.engine.config import EngineConfig

# Save config
config = engine.to_config(
    preset="production",
    metadata={"vocabulary": "english_48k", "deployed": "2026-04-30"},
)
with open("engine_config.json", "w") as f:
    f.write(config.to_json())

# Reconstruct on another server - identical behaviour
with open("engine_config.json") as f:
    engine2 = EngineConfig.from_json(f.read()).build()

# Audit differences between two configs
diffs = config_a.diff(config_b)
# → ["predictor 'history' weight: 1.2 → 1.5", "ranker 'decay' params: ..."]
```

---

## Persistent learning

```python
from aac.presets import create_engine
from aac.storage.json_store import JsonHistoryStore
from pathlib import Path

store = JsonHistoryStore(Path("~/.aac_history.json").expanduser())
engine = create_engine("production", history=store.load())

suggestions = engine.suggest("prog")
engine.record_selection("prog", suggestions[0])
store.save(engine.history)   # write to disk - available next session
```

The CLI does this automatically. History is stored with full ISO 8601 timestamps so recency decay works correctly after reload. V1 count-only files migrate automatically.

---

## Custom vocabulary

```python
from aac.vocabulary import vocabulary_from_wordlist, vocabulary_from_text

# Word list - equal weights
vocab = vocabulary_from_wordlist(["git commit", "git push", "git pull"])
engine = create_engine("production", vocabulary=vocab)
engine.suggest("git")   # → ["git commit", "git push", "git pull"]
# Note: CompletionContext.prefix() extracts the last word of the input,
# so "git commit" is matched by prefix "git", not "git c".
# For multi-word completions, query by the prefix of the first word.

# Text corpus - weighted by frequency
vocab = vocabulary_from_text(open("docs.txt").read())

# Mix domain terms with bundled English
from aac.data import load_english_frequencies
combined = {
    **load_english_frequencies(),
    **vocabulary_from_wordlist(["asyncio", "dataclass"], default_frequency=10_000),
}
engine = create_engine("production", vocabulary=combined)
```

Runtime extension - no rebuild required:

```python
from aac.predictors import FrequencyPredictor
predictor = FrequencyPredictor(base_vocab)
predictor.add_word("MyNewSymbol", frequency=5_000)   # visible immediately
```

---

## Presets

**Default: `production`.**

| Preset | Learns | Typo recovery | Notes |
|--------|--------|---------------|-------|
| `stateless` | No | No | Reproducible, highest throughput |
| `default` | Yes | No | Frequency + history |
| `recency` | Yes (decay) | No | Recent selections outweigh old |
| `production` | Yes (decay) | Hybrid SymSpell + trigram | **Recommended** |
| `robust` | Yes (decay) | SymSpell only | Maximum recall, higher memory |

`bktree` is retained for benchmarking but excluded from `available_presets()` - it degrades to O(n) at `max_distance=2` over 48k+ words (~60ms/call vs ~600µs for `production`).

---

## vs alternatives

Compared against real pip-installable libraries that solve adjacent problems:

| | adaptive-autocomplete | [whoosh](https://pypi.org/project/Whoosh/) | [flashtext](https://pypi.org/project/flashtext/) | [pyahocorasick](https://pypi.org/project/pyahocorasick/) |
|---|---|---|---|---|
| Typo recovery | ✓ SymSpell + trigram | ✓ (BM25, no edit distance) | ✗ | ✗ |
| Learns from use | ✓ history + recency decay | ✗ | ✗ | ✗ |
| Explains rankings | ✓ `explain()` per-word | ✗ | ✗ | ✗ |
| Contribution % | ✓ `contribution_pct` | ✗ | ✗ | ✗ |
| Contextual history | ✓ `ContextualHistory` | ✗ | ✗ | ✗ |
| Serialisable config | ✓ `EngineConfig` | partial (index file) | ✗ | ✗ |
| Batch + async API | ✓ | ✗ | ✗ | ✗ |
| Weight optimiser | ✓ coordinate descent | ✗ | ✗ | ✗ |
| ~48k vocab p50 | ~600µs | ~2–8ms (index load) | ~40µs (exact only) | ~30µs (exact only) |
| Zero dependencies | ✓ | ✗ (six, whoosh) | ✓ | ✗ (C extension) |

**When to choose something else**: if you need pure exact-match keyword extraction at maximum throughput, `flashtext` or `pyahocorasick` are faster. If you need full-text search with field weighting and stemming, `whoosh` is the right tool. If you need typo recovery + learned personalisation + explainability in a zero-dependency Python package, nothing else does all three.

---

## API reference

Core surface - full signatures and parameters are in the module docstrings and `examples/`.

```python
from aac.presets import create_engine, compare_presets
from aac.engine.config import EngineConfig
from aac.domain.contextual_history import ContextualHistory

engine = create_engine("production")

# Suggestions
engine.suggest("helo")                           # → ['help', 'held', ...]
engine.suggest("prog", limit=5)
engine.suggest_with_confidence("prog", limit=5)  # → [('programming', 1.0), ...]
engine.suggest_with_history("prog", limit=5)     # → [('programming', 3), ...]
engine.suggest_full("prog", limit=5)             # → [{'word': 'programming', 'count': 3, 'confidence': 1.0}, ...]

# Learning
engine.record_selection("prog", "programming")
engine.reset_history()

# Explanation
engine.explain("prog")           # → [RankingExplanation(...), ...]
engine.explain_as_dicts("prog")  # → [{value, base_score, contribution_pct, ...}]

# Batch
engine.batch_suggest(["prog", "hel", "wor"], limit=5)
engine.batch_explain(["prog", "hel"], limit=5)
await engine.batch_suggest_async([\"prog\", \"hel\"])  # non-blocking; see GIL note in docstring

# Config round-trip
config = engine.to_config(preset="production", metadata={"env": "prod"})
EngineConfig.from_json(config.to_json()).build()
config_a.diff(config_b)

# Contextual learning
ctx = ContextualHistory()
ctx.record("prog", "programming", domain="python")
engine = create_engine("production", history=ctx.for_domain("python"))

# Async (FastAPI, aiohttp)
await engine.suggest_async("prog", limit=5)
await engine.record_selection_async("prog", "programming")
await engine.explain_async("prog")
```

See [`examples/`](examples/) for FastAPI service, custom engine construction, contextual history, and evaluation usage.

---

## Evaluation

The `aac.evaluation` module lets you measure ranking quality and tune weights - something no comparable autocomplete library provides.

### Measure precision, MRR, NDCG against your own query log

```python
from aac.evaluation import EvaluationHarness, make_query_log_from_history
from aac.presets import create_engine

# After recording real user selections:
engine = create_engine("production")
# engine.record_selection("prog", "programming")  # ... many times

# Build a query log from recorded history (selections = ground truth)
harness = EvaluationHarness.from_history(engine.history, k=10, min_count=2)

result = harness.run(engine)
print(result.summary())
# n=142 queries @ k=10: P@k=0.214  MRR=0.847  NDCG=0.891  MAP=0.823  HitRate=0.944

print(result.to_markdown_table())
# | Metric             | Value  |
# | ------------------ | ------ |
# | MRR@k              | 0.847  |
# | NDCG@k             | 0.891  |
# | Hit rate           | 94.4%  |
# ...

# See which queries the engine struggles with
for qr in result.worst_queries(5):
    print(f"  {qr.entry.prefix!r:<15}  MRR={qr.mrr:.3f}  got={qr.ranked[:3]}")
```

Or from the CLI:

```bash
aac eval --from-history                    # uses ~/.aac_history.json
aac eval --from-history --k 5             # evaluate at k=5
aac eval --from-history --markdown        # copy table into README
aac eval --from-history --worst 10        # show 10 hardest queries
aac eval --query-log queries.jsonl        # labelled JSONL file
```

### Compare presets on the same query log

```python
from aac.evaluation.datasets import make_synthetic_query_log
from aac.data import load_english_frequencies
from aac.presets import create_engine

vocab = list(load_english_frequencies().keys())
log = make_synthetic_query_log(vocab[:500], prefix_lengths=[2, 3, 4])
harness = EvaluationHarness(log, k=10)

for preset in ["stateless", "default", "production"]:
    result = harness.run(create_engine(preset))
    print(f"{preset:12s}  MRR={result.mean_mrr:.3f}  NDCG={result.mean_ndcg:.3f}  Hit={result.hit_rate:.1%}")
```

### Automated weight optimisation

```python
from aac.evaluation import EvaluationHarness, WeightOptimiser

opt = WeightOptimiser(harness, metric="mrr")

# Coordinate descent - fast, finds good weights in O(n_predictors × n_weights) evals
result = opt.coordinate_descent(
    base_preset="production",
    weight_grid={
        "frequency":         [0.5, 1.0, 2.0],
        "history":           [0.8, 1.2, 1.6],
        "adaptive_symspell": [0.2, 0.35, 0.5],
        "trigram":           [0.2, 0.4,  0.6],
    },
)
print(result.report())
# WeightOptimiser - coordinate_descent
# Metric:       mrr
# Baseline:     0.8312
# Optimised:    0.8791  (+0.0479, +5.8%)
# Best weights:
#   adaptive_symspell     0.350
#   frequency             1.000
#   history               1.600
#   trigram               0.400
```

Or from the CLI:

```bash
aac tune --from-history                    # coordinate descent, metric=mrr
aac tune --from-history --metric ndcg     # optimise for NDCG instead
aac tune --from-history --strategy grid   # exhaustive grid search (slower)
```

---

## Performance

20,000 `suggest()` calls across 10 prefixes, full 48,032-word vocabulary. Run `make benchmark` to reproduce.

| Preset | avg | p50 | p99 | Why |
|--------|-----|-----|-----|-----|
| `stateless` | ~82µs | ~73µs | ~124µs | `FrequencyPredictor` pre-sorts at construction; `predict()` is a top-N slice, O(max_results) not O(vocab) |
| `default` | ~84µs | ~74µs | ~124µs | Same as `stateless` + O(k) history lookup where k = selections for this prefix |
| `recency` | ~90µs | ~78µs | ~138µs | Adds `DecayRanker` which scans history once per call; cached for `explain()` |
| `robust` | ~400µs | ~360µs | ~900µs | `SymSpellPredictor`: O(max_distance × |q|) index lookups + Levenshtein verification |
| `production` | ~650µs | ~600µs | ~1.5ms | `AdaptiveSymSpellPredictor` (two indexes) + `TrigramPredictor` pre-filter; higher p99 than `robust` because trigram verification adds a second pass on the shortlist |
| `bktree` (48k) | ~60ms | - | - | ⚠ BK-tree triangle-inequality pruning degrades to O(n) at this scale; the search ball at distance=2 covers most of the metric space |

**Memory:** `production` builds two SymSpell indexes (tight + full, ~50MB each) and one trigram index (~30MB). Total footprint ~130MB at 48k vocabulary. `stateless` and `default` are ~10MB. If memory is the constraint, use `default` or bring a smaller vocabulary.

CI enforces performance bounds: `stateless` p99 < 5ms, `production` p99 < 30ms. These are deliberately generous (10–20× the measured values) to avoid flakiness on loaded CI runners while still catching catastrophic regressions.

---

## Architecture

```
User input
    ↓
CompletionContext      (lowercases prefix, normalises whitespace)
    ↓
Predictors            (stateless: frequency, history, symspell, trigram)
    ↓
Weighted aggregation  (additive, per-predictor weights)
    ↓
Rankers               (stateful: score, decay - reorder and boost; cannot add/remove candidates)
    ↓
Suggestions + explanations
```

**Prediction and ranking are separate layers.** Prediction is stateless - it asks what words plausibly complete this prefix and how likely each is. Ranking is stateful - it reorders candidates based on history and recency. The seam exists because the original monolithic function was untestable: writing a test for the learning behaviour required mocking the full prediction pipeline.

**`explain()` is a single forward pass.** Suggestions pass through the ranker chain once; score deltas are captured at each step. Earlier versions re-ran the full pipeline to compute deltas, making `explain()` cost 2× `suggest()`. It now costs the same.

**Enforced invariants:**

- `final_score == base_score + history_boost` in `RankingExplanation.__post_init__`.
- Rankers cannot add or remove candidates. Enforced at runtime with `RuntimeError` naming the offender - not `assert`, disabled under `-O`.
- History has one owner. The engine holds the single `History` instance. `reset_history()` propagates the new instance to all rankers and predictors via typed protocols (`LearnsFromHistory`, `PredictorLearnsFromHistory`).
- `base_components` always contains all configured predictor names. `0.0` means "ran, below threshold" - not "not configured".
- History reads are O(k) via prefix index, not O(n) full scan.

---

## Local development

```bash
git clone https://github.com/bonnie-mcconnell/adaptive-autocomplete
cd adaptive-autocomplete
make install

source .venv/bin/activate   # Linux/macOS

make demo        # interactive browser demo
make test-fast   # unit tests only (~30s)
make test        # full suite (~3 min)
make benchmark   # latency numbers
```

---

## Tests

594 tests across correctness invariants, IR metrics, evaluation harness, concurrency, async API, integration, and property-based fuzzing.

**Correctness guarantees:**
- SymSpell brute-force equivalence: every result matches linear scan exactly across multiple queries and distance thresholds.
- Trigram no false positives: every returned result is within `max_distance` per exact Levenshtein.
- Explain invariant: `final_score == base_score + history_boost` for every suggestion, every preset.
- Explain single-pass: `ranker.rank()` called exactly once per `explain()` call (spy test, not output test).
- Ranker invariant: `RuntimeError` raised if a ranker adds or removes candidates.
- Decay no double-counting: pre-ranking scores are ground truth; ranker deltas captured in forward pass.

**Correctness guarantees (full list in CHANGELOG):**
- `FrequencyPredictor` max_results=100: words ranked 21–100 in frequency now returned; "hello" for prefix "he" verified present.
- `base_components` completeness: all predictor names present; `0.0` distinguishes "below threshold" from "not configured".
- `contribution_pct`: values in (0, 1]; zero-contribution sources omitted; dominant source correct after many selections.
- `History.copy()`: modifications to copy do not affect original; prefix index works correctly in copy.
- `compare_presets()` isolation: caller's History unmodified after comparison; each engine receives independent copy.
- `ContextualHistory`: domain isolation verified; engine built on domain uses domain history; `for_domain()` returns live instance.
- `EngineConfig`: round-trip JSON preserves all fields; `build()` produces engine with identical suggestions; `diff()` detects weight changes, added/removed predictors; bad version raises.

**Property-based (Hypothesis):** four core invariants across thousands of generated inputs, including two floating-point precision bugs that example-based tests missed.

**Integration:** CLI invoked as subprocess for `suggest`, `explain`, `record`, `history`, `presets`.

---

## What I'd change

**The `default` preset ignores time.** Raw selection counts don't decay. Something selected 50 times six months ago outweighs something selected twice yesterday. `recency` and `production` fix this, but the right design bakes time-awareness into the core history model rather than adding it in the ranker. The CHANGELOG has this honest admission about the original design decision.

**Presets obscure composability.** The direct constructor is already the better interface. Presets are convenience wrappers that hide the weighting decisions from users who might benefit from tuning them. I shipped presets first and the constructor second; it should have been the other way around, with presets as thin named constructors over the direct API.

**The confidence score formula is a heuristic.** The hybrid approach (raw normalisation below 4x dominance threshold, rank-based weighting above it) produces intuitively reasonable output but has no principled statistical justification. A cleaner solution would model user selection probability directly - something like a contextual bandit where each suggestion's selection probability is estimated from history and the confidence IS that probability estimate. That is a real algorithm; this is a workaround that happens to produce sensible numbers.
