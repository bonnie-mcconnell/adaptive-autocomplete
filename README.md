# adaptive-autocomplete

[![CI](https://github.com/bonnie-mcconnell/adaptive-autocomplete/actions/workflows/ci.yml/badge.svg)](https://github.com/bonnie-mcconnell/adaptive-autocomplete/actions)
[![PyPI](https://img.shields.io/pypi/v/adaptive-autocomplete)](https://pypi.org/project/adaptive-autocomplete/)
[![Python](https://img.shields.io/pypi/pyversions/adaptive-autocomplete)](https://pypi.org/project/adaptive-autocomplete/)

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
# RankingExplanation(value='programming', base=1.84, boost=+1.50, final=3.34)

engine.suggest_with_confidence("prog", limit=5)
# [('programming', 1.0), ('program', 0.75), ('progress', 0.64), ...]
```

The differentiating feature is `explain()`. Every suggestion comes with a mathematically auditable score breakdown: which predictor contributed what, how much the ranker boosted it, and the relative contribution of each source as a percentage of the final score. `final_score == base_score + history_boost` is enforced as an object-level invariant - it cannot be violated.

---

## Install

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
aac suggest prog                    # ranked by frequency + learning
aac suggest programing              # typo recovery → programming
aac explain prog                    # score breakdown per suggestion
aac record prog programming         # record a selection; engine learns
aac history prog                    # what the engine has learned
aac demo                            # interactive browser UI
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
aac suggest programing   → programming          (distance 2)
aac suggest recieve      → receive, relieve     (transposition + substitution)
aac suggest helo         → help, hello, hell
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

hello          score=     1.87 (100.0%)  base=     0.62  boost=+1.25
her            score=     0.75 ( 40.1%)  base=     0.75  boost=+0.00
```

Every breakdown shows `base_score` (sum of predictor contributions), `history_boost` (ranker adjustment), `final_score`, and `contribution_pct` (each source as a fraction of final). The invariant `final_score == base_score + history_boost` is enforced in `RankingExplanation.__post_init__` - there is no code path that can violate it.

`base_components` always includes every configured predictor with an explicit score - `0.0` means "predictor ran but this word was below its threshold", never "predictor not configured". This distinction matters for debugging and for tuning weights.

```python
engine.explain("he")[0]
# RankingExplanation(value='hello', base=1.67, boost=+1.50, final=3.17)
# base_components:    {'frequency': 0.47, 'history': 1.20, 'symspell': 0.0, 'trigram': 0.0}
# history_components: {'decay': 1.50}
# contribution_pct:   {'frequency': 0.15, 'history': 0.38, 'decay': 0.47}
```

The `contribution_pct` field makes weight-tuning direct: "decay is contributing 47% of the final score - do I want learning to dominate this much?"

### Confidence scores

```python
for word, conf in engine.suggest_with_confidence("prog", limit=5):
    label = "★" if conf > 0.8 else " "
    print(f"{label} {word:<20} {conf:.0%}")
# ★ programming          100%
#   program               75%
#   progress              64%
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

`compare_presets()` runs `explain()` across multiple engines simultaneously and returns side-by-side score breakdowns. Each engine gets an independent copy of the History so comparison is always consistent.

```python
from aac.presets import compare_presets

cmp = compare_presets("recieve", ["stateless", "production"])
print(cmp.to_table())
```

```
suggestion         stateless                    production
                rank    base   boost   final   rank    base   boost   final
---------------------------------------------------------------------------
recieved          #1   0.147  +0.000   0.147    #2   0.526  +0.000   0.526
recieve            -       -       -       -     #1   0.754  +0.000   0.754
relieve            -       -       -       -     #3   0.179  +0.000   0.179
receive            -       -       -       -     #5   0.122  +0.000   0.122
```

`stateless` returns the misspelling as a corpus frequency hit. `production` surfaces the intended word. The table shows exactly why.

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

| | adaptive-autocomplete | trie / prefix tree | BK-tree fuzzy |
|---|---|---|---|
| Typo recovery | ✓ hybrid SymSpell + trigram | ✗ | ✓ slow at scale |
| Learns from use | ✓ history + decay | ✗ | ✗ |
| Explains rankings | ✓ `explain()` | ✗ | ✗ |
| Contribution % | ✓ `contribution_pct` | ✗ | ✗ |
| Contextual learning | ✓ `ContextualHistory` | ✗ | ✗ |
| Serialisable config | ✓ `EngineConfig` | ✗ | ✗ |
| Compare strategies | ✓ `compare_presets()` | ✗ | ✗ |
| ~48k vocab p50 | ~600µs | ~70µs | ~60ms |
| Required dependencies | none | none | none |

If you need pure prefix matching at maximum throughput and never need typo recovery or learning, a trie is faster. If vocabulary is ≤5k words and you need fuzzy matching only, BK-tree is simpler. Otherwise: this.

---

## API reference

```python
from aac.presets import create_engine, compare_presets
from aac.engine.config import EngineConfig
from aac.domain.contextual_history import ContextualHistory

engine = create_engine("production")

# Suggestions
engine.suggest("helo")                           # → ['help', 'held', ...]
engine.suggest("prog", limit=5)                  # limit results
engine.suggest_with_confidence("prog", limit=5)  # → [('programming', 1.0), ...]
engine.suggest_with_history("prog", limit=5)     # → [('programming', 3), ...]

# Learning
engine.record_selection("prog", "programming")
engine.reset_history()           # clear in-memory state; does not touch store

# Explanation
engine.explain("prog")           # → [RankingExplanation(...), ...]
engine.explain_as_dicts("prog")  # → [{value, base_score, contribution_pct, ...}]

# Config
config = engine.to_config(preset="production", metadata={"env": "prod"})
config.to_json()                 # serialise
EngineConfig.from_json(text).build()             # reconstruct
config_a.diff(config_b)          # audit differences

# Comparison
compare_presets("recieve")                           # all presets
compare_presets("recieve", ["stateless", "production"])

# Contextual learning
ctx = ContextualHistory()
ctx.record("prog", "programming", domain="python")
engine = create_engine("production", history=ctx.for_domain("python"))

# Introspection
engine.history                   # live History instance
engine.describe()                # {predictors, rankers, history_entries}

# Async (FastAPI, aiohttp)
await engine.suggest_async("prog", limit=5)
await engine.record_selection_async("prog", "programming")
await engine.explain_async("prog")
```

**`explain_as_dicts()` schema:**

```python
[{
    "value":               "programming",
    "base_score":          1.84,
    "history_boost":       1.50,
    "final_score":         3.34,
    "source":              "history",
    "sources":             ["frequency", "history", "decay"],
    "base_components":     {"frequency": 0.64, "history": 1.20, "symspell": 0.0, "trigram": 0.0},
    "history_components":  {"decay": 1.50},
    "contribution_pct":    {"frequency": 0.19, "history": 0.36, "decay": 0.45},
}]
```

**Custom engine:**

```python
from aac.engine import AutocompleteEngine
from aac.predictors import FrequencyPredictor, HistoryPredictor, AdaptiveSymSpellPredictor
from aac.domain.types import WeightedPredictor
from aac.domain.history import History
from aac.ranking.decay import DecayFunction, DecayRanker
from aac.ranking.score import ScoreRanker

vocab = {"hello": 100, "help": 80, "hero": 50}
history = History()

engine = AutocompleteEngine(
    predictors=[
        WeightedPredictor(FrequencyPredictor(vocab), weight=1.0),
        WeightedPredictor(HistoryPredictor(history), weight=1.5),
        WeightedPredictor(
            AdaptiveSymSpellPredictor(vocab.keys(), max_distance=2, frequencies=vocab),
            weight=0.35,
        ),
    ],
    ranker=[
        ScoreRanker(),
        DecayRanker(history, DecayFunction(half_life_seconds=3600), weight=2.0),
    ],
    history=history,
)
```

See [`examples/fastapi_app.py`](examples/fastapi_app.py) for a FastAPI service with `ThreadSafeHistory` and atomic persistence.

---

## Performance

20,000 `suggest()` calls across 10 prefixes, full 48,032-word vocabulary. Run `make benchmark` to reproduce.

| Preset | avg | p50 | p99 | Notes |
|--------|-----|-----|-----|-------|
| `stateless` | ~82µs | ~73µs | ~124µs | Frequency only |
| `default` | ~84µs | ~74µs | ~124µs | + history predictor |
| `recency` | ~90µs | ~78µs | ~138µs | + decay ranker |
| `production` | ~650µs | ~600µs | ~1.5ms | Hybrid SymSpell + trigram |
| `robust` | ~400µs | ~360µs | ~900µs | SymSpell only |
| `bktree` (48k) | ~60ms | - | - | ⚠ O(n) at this scale |

**Memory:** `production` builds two SymSpell indexes (tight + full, ~50MB each) and one trigram index (~30MB). Total footprint ~130MB at 48k vocabulary. `stateless` and `default` are ~10MB. If memory is the primary constraint, use `default` or bring a smaller vocabulary.

`FrequencyPredictor` pre-sorts its prefix index at construction time (O(v log v) once) so `predict()` is a top-N slice - O(max_results), not O(vocabulary). The default `max_results=100` was raised from 20 in v0.6.0 to prevent silent truncation of words ranked 21–100 in frequency for their prefix.

`TrigramPredictor` beats BK-tree at scale because trigram overlap pre-filters to a candidate shortlist before Levenshtein. At `max_distance=2` over 48k words, BK-tree's triangle-inequality pruning degrades toward O(n) - the search ball covers most of the metric space.

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

449 tests across correctness invariants, integration, and property-based fuzzing.

**Correctness guarantees:**
- SymSpell brute-force equivalence: every result matches linear scan exactly across multiple queries and distance thresholds.
- Trigram no false positives: every returned result is within `max_distance` per exact Levenshtein.
- Explain invariant: `final_score == base_score + history_boost` for every suggestion, every preset.
- Explain single-pass: `ranker.rank()` called exactly once per `explain()` call (spy test, not output test).
- Ranker invariant: `RuntimeError` raised if a ranker adds or removes candidates.
- Decay no double-counting: pre-ranking scores are ground truth; ranker deltas captured in forward pass.

**New in v0.6.0:**
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

## Migrating from 0.5.x to 0.6.0

### Potentially breaking

**`FrequencyPredictor` default `max_results` changed from 20 to 100.** If you were relying on the old truncation behaviour (e.g. to limit memory at a small vocabulary size), pass `max_results=20` explicitly. For most use cases this change improves correctness.

**`RankingExplanation` gains `contribution_pct` field.** Code constructing `RankingExplanation` directly with positional arguments may break - use keyword arguments. Code using `asdict(explanation)` or `to_dict()` will now see the additional key; downstream consumers that validate exact key sets will need updating.

**`explain()` `base_components` now always complete.** If you were checking for the absence of a predictor name as a signal that it didn't contribute, you need to check for `value == 0.0` instead.

**`explain()` `history_components` now always includes all non-score ranker names.** Zero-boost rankers appear with `0.0` rather than being absent.

### Non-breaking additions

- `History.copy()` - independent deep copy.
- `ContextualHistory` - domain-partitioned history.
- `EngineConfig` / `engine.to_config()` / `EngineConfig.from_json().build()` / `config.diff()`.
- `RankingExplanation.contribution_pct` - relative source percentages.
- Demo `/compare` endpoint now instant (engines cached at startup).

---

## What I'd change

**The `default` preset ignores time.** Raw selection counts don't decay. Something selected 50 times six months ago outweighs something selected twice yesterday. `recency` and `production` fix this, but the right design bakes time-awareness into the core history model rather than adding it in the ranker. The CHANGELOG has this honest admission about the original design decision.

**Presets obscure composability.** The direct constructor is already the better interface. Presets are convenience wrappers that hide the weighting decisions from users who might benefit from tuning them. I shipped presets first and the constructor second; it should have been the other way around, with presets as thin named constructors over the direct API.

**`EngineConfig.build()` raises for custom engines.** Reconstructing a custom engine from config requires caller-supplied predictor instances that cannot be inferred from names alone. A full solution would register predictor classes by name (like a plugin registry) so `build()` could reconstruct any engine. The current implementation handles the common case (preset engines) correctly; custom engine reconstruction is left to the caller.
