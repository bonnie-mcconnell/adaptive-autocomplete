# adaptive-autocomplete

![CI](https://github.com/bonnie-mcconnell/adaptive-autocomplete/actions/workflows/ci.yml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/adaptive-autocomplete)](https://pypi.org/project/adaptive-autocomplete/)

Autocomplete engine built from scratch in Python - candidate generation, frequency ranking, typo recovery, learning from user selections, and per-suggestion score explanations.

I built it to understand what ranking infrastructure actually looks like underneath the ML layer: how candidates get generated, scored, merged across predictors, reordered by rankers, and explained back to the caller. The first version was a single function. It became this when I tried to write a test for the learning behaviour and discovered there was no seam to inject a controlled history.

---

## Quick start

**Install from PyPI** (Python 3.10+):

```bash
pip install adaptive-autocomplete
```

```bash
aac suggest he                               # completions ranked by frequency
aac explain he                               # score breakdown per suggestion
aac record he hero                           # record a selection - engine learns
aac --preset production suggest programing   # typo recovery: programing → programming
```

**Clone and run locally** (for benchmarks, tests, and the demo):

```bash
git clone https://github.com/bonnie-mcconnell/adaptive-autocomplete
cd adaptive-autocomplete
make install

# Activate the virtualenv to use 'aac' directly:
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows PowerShell

# Or prefix any command with 'poetry run':
poetry run aac suggest he

make demo        # run the full pipeline end-to-end
make test-fast   # unit tests only (~30s)
make test        # full suite including integration tests (~90s)
make benchmark   # latency numbers against the full 48k vocabulary
```

Requires Python 3.10+.

A FastAPI endpoint example using `ThreadSafeHistory` and `snapshot_history()` is in [`examples/fastapi_app.py`](examples/fastapi_app.py).

---

## Python API

```python
from aac.presets import create_engine

# One line - production preset, 48k vocabulary, fresh in-memory history
engine = create_engine("production")
engine.suggest("helo")           # → ['help', 'held', 'hell', 'hello', ...]
engine.record_selection("helo", "hello")   # engine learns immediately
engine.explain("helo")           # → [RankingExplanation(value='hello', base=1.4063, boost=+1.5000, final=2.9063), ...]
```

**Persisting learning across restarts:**

```python
from aac.presets import create_engine
from aac.storage.json_store import JsonHistoryStore
from pathlib import Path

store = JsonHistoryStore(Path("~/.aac_history.json").expanduser())
engine = create_engine("production", history=store.load())

suggestions = engine.suggest("prog")
engine.record_selection("prog", suggestions[0])
store.save(engine.history)   # write learning to disk - loads back next session
```

Or compose a custom engine directly - no presets required:

```python
from aac.engine import AutocompleteEngine
from aac.predictors import FrequencyPredictor, HistoryPredictor, TrigramPredictor
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
        WeightedPredictor(TrigramPredictor(vocab.keys()), weight=0.4),
    ],
    ranker=[
        ScoreRanker(),
        DecayRanker(history, DecayFunction(half_life_seconds=3600), weight=2.0),
    ],
    history=history,
)
```

**Using your own vocabulary:**

```python
from aac.presets import create_engine
from aac.vocabulary import vocabulary_from_wordlist, vocabulary_from_text, vocabulary_from_file

# From a plain word list - all words get equal weight
vocab = vocabulary_from_wordlist(["git commit", "git push", "git pull", "git status"])
engine = create_engine("production", vocabulary=vocab)
engine.suggest("git")  # → ["git commit", "git push", "git pull", "git status"]

# From a text corpus - words weighted by frequency
with open("corpus.txt") as f:
    vocab = vocabulary_from_text(f.read())
engine = create_engine("default", vocabulary=vocab)

# From a file - one word per line (wordlist) or free-form text
vocab = vocabulary_from_file("commands.txt")                      # wordlist format
vocab = vocabulary_from_file("corpus.txt", format="text")        # text format

# Or via the CLI:
# aac --vocab-path commands.txt suggest git
# aac --vocab-path corpus.txt --vocab-format text suggest prog
```

Mix with the bundled English vocabulary by using a high `default_frequency` so your domain terms surface above common words:

```python
from aac.data import load_english_frequencies

english = dict(load_english_frequencies())
# Note: CompletionContext lowercases all prefixes, so vocabulary keys should be lowercase.
domain = vocabulary_from_wordlist(["asyncio", "dataclass", "typeddict"], default_frequency=10_000)
combined = {**english, **domain}
engine = create_engine("production", vocabulary=combined)
engine.suggest("dat")  # → ["dataclass", "data", "date", ...]
```

```python
from aac.engine import AutocompleteEngine
from aac.predictors import FrequencyPredictor, HistoryPredictor, TrigramPredictor
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
        WeightedPredictor(TrigramPredictor(vocab.keys()), weight=0.4),
    ],
    ranker=[
        ScoreRanker(),
        DecayRanker(history, DecayFunction(half_life_seconds=3600), weight=2.0),
    ],
    history=history,
)
```

**Power-user API - raw scored results:**

```python
from aac.domain.types import CompletionContext

# predict_scored() returns ScoredSuggestion objects with per-suggestion scores,
# explanations, and trace strings. Use this when you want to build your own
# ranking layer on top of the engine's predictor stack, or for testing.
ctx = CompletionContext("hel")
scored = engine.predict_scored(ctx)
for s in scored:
    print(f"{s.value}: score={s.score:.4f}, source={s.explanation.source}")
# → hello: score=1.0000, source=frequency
# → help:  score=0.9522, source=frequency

# explain_as_dicts() for JSON/logging - includes per-ranker component breakdown.
# After engine.record_selection("hel", "hello"), history signal appears in base_components
# and the DecayRanker boost appears in history_components:
engine.record_selection("hel", "hello")
dicts = engine.explain_as_dicts("hel")
# → [{"value": "hello", "base_score": 2.5, "history_boost": 2.0, "final_score": 4.5,
#     "sources": ["frequency", "history", "decay"],
#     "base_components": {"frequency": 1.0, "history": 1.5},
#     "history_components": {"decay": 2.0}}, ...]
```

---

## Migrating from 0.2.x to 0.3.0

### Breaking changes

**Score values changed.** `FrequencyPredictor` and `HistoryPredictor` previously emitted raw counts as scores (e.g. `50000.0` for "the"). They now emit log-normalised scores in `(0, 1]`. This affects `ScoredSuggestion.score` and `RankingExplanation.base_score`. **Suggestion ordering is preserved** - only the numeric values changed.

If you compare scores against hardcoded thresholds, update them. The new formula is `log(1 + freq) / log(1 + max_freq)`.

**`WeightedPredictor` rejects `weight <= 0`.** Previously accepted silently. Now raises `ValueError` at construction. Remove zero-weight predictors from your predictor list instead.

**`ThreadSafeHistory.lock` type changed** from `threading.Lock` to `threading.Condition`. `isinstance(ts.lock, threading.Lock)` is now `False`. If you use `with ts.lock:` for compound operations, it still works - `Condition` supports the context manager protocol.

**`History.snapshot()` is soft-deprecated.** It still works but its docstring now carries a deprecation notice. Use `snapshot_counts()` for the same result, or `JsonHistoryStore.save()` for persistence.

### Non-breaking changes

`JsonHistoryStore` is now exported from the top-level `aac` package: `from aac import JsonHistoryStore`.

`History.snapshot_counts()` added as an explicit-name alias for `snapshot()`.

`engine.explain_as_dicts()` now returns richer dicts with `sources`, `base_components`, and `history_components` fields alongside the existing `value`, `base_score`, `history_boost`, `final_score`.

---

```
$ aac suggest he
her
here
help
head
health
heart
heard
held
hear
hey

$ aac explain he
her            score=     0.75 (100.0%)  base=     0.75  boost=+0.00
here           score=     0.69 ( 92.3%)  base=     0.69  boost=+0.00
help           score=     0.65 ( 87.2%)  base=     0.65  boost=+0.00
head           score=     0.61 ( 81.6%)  base=     0.61  boost=+0.00
health         score=     0.60 ( 80.0%)  base=     0.60  boost=+0.00
...

$ aac --preset production suggest programing
programming
```

Learning works immediately. `make demo` shows it with a small controlled vocabulary where the boost is proportional to the score gap, so movement is visible after a few selections. With a 48k production vocabulary and the `default` preset, a single selection lifts a word above most frequency-only candidates because history scores are log-normalised to the same (0, 1] scale as frequency scores - one selection yields a history score of 1.0, weighted by 1.5, which exceeds the frequency score of most non-top-100 words.

In the `default` preset, learning happens at the **prediction** layer - `HistoryPredictor` emits history-scored candidates weighted and aggregated with frequency scores before ranking. The `recency` and `production` presets apply `DecayRanker` at ranking time instead, which is why their `boost` column is non-zero in `explain` output.

History persists across restarts with full ISO 8601 timestamps, so decay-based presets remain accurate after reload.

---

## The design decision that shaped everything else

The question was whether prediction and ranking should be one operation or two.

On the surface they look like one thing: text goes in, ordered suggestions come out. But they're solving different problems.

**Prediction** asks: what words plausibly complete this prefix, and how likely is each one? It's stateless. A frequency predictor doesn't know what you selected yesterday. A trie predictor doesn't know either. Given the same input, they always return the same output.

**Ranking** asks: given these candidates and their scores, what order should the user see, and how should past behaviour change that? It's stateful and is where learning lives.

Separating them means each layer has a single job. A predictor can be replaced without touching the learning logic. The engine stays thin - it orchestrates, it doesn't contain scoring or ordering logic. And the layers can be tested independently, which matters when debugging why a particular word appeared where it did.

The tradeoff is more code than a single entangled function. It's worth it because prediction and ranking can be tested in complete isolation - something that was impossible when they were the same thing.

---

## Presets

Six operating modes. **If you don't know which to use, use `production`.**

| Preset | Learns | Typo recovery | Vocabulary scale | Use when |
|--------|--------|---------------|-----------------|----------|
| `stateless` | No | No | Any | Reproducible, high-throughput; no state |
| `default` | Yes | No | Any | Learning without typo recovery |
| `recency` | Yes (decay) | No | Any | Recent selections outweigh old ones |
| `production` | Yes (decay) | Yes (trigram) | 48k+ words | **Recommended default** |
| `robust` | Yes (decay) | Yes (SymSpell) | 48k+ words | Short prefixes, max recall |
| `bktree` | Yes (decay) | Yes (BK-tree) | Small only | Benchmarking/comparison only |

`production` uses a trigram index for approximate matching. At `max_distance=2` over 48k words it takes ~600µs/call. The constraint: trigram matching requires prefix length ≥ 4. For 1–3 character prefixes only frequency and history signals apply.

`robust` uses a **SymSpell delete-neighbourhood index** - O(1) average query at any vocabulary size, ~400µs/call at 48k words. Works correctly on 1–3 character prefixes. One-time build cost: ~1.5s at 48k words, ~50MB RAM. Replaced the BK-tree in 0.2.0.

`bktree` retains the original BK-tree for comparison. Degrades to O(n) at `max_distance=2` over large vocabularies (~60ms at 48k words). Use `robust` or `production` for production typo recovery.

---

## Architecture

```
User input
    ↓
CompletionContext  (lowercases prefix, handles cursor position)
    ↓
Predictors  →  scored candidates (stateless, composable)
    ↓
Weighted aggregation
    ↓
Rankers  →  ordered results + learning updates
    ↓
Suggestions + explanations
```

**Enforced invariants:**

**Rankers cannot add or remove candidates.** They can only reorder and rescore. Checked at runtime with a `RuntimeError` naming the offending ranker - not an `assert`, which Python disables under `-O`.

**Explanations must reconcile with scores.** `RankingExplanation` enforces `final_score == base_score + history_boost` in `__post_init__`. If any code produces an inconsistent explanation, it fails at construction, not silently downstream. `merge()` and `apply_history_boost()` compute intermediate sums before passing to the constructor to avoid floating-point precision divergence from multi-term addition - a bug Hypothesis found and that example-based tests missed.

**History has one owner.** The engine holds the single `History` instance. Rankers can read it; only the engine writes to it. This makes the audit trail clean and prevents predictors from recording selections twice.

**History reads are O(k), not O(n).** `History` maintains a prefix index alongside its entry list - a `defaultdict` keyed by prefix, updated on every `record()` call. All prefix-scoped reads (`counts_for_prefix`, `entries_for_prefix`, `DecayRanker`) use the index rather than scanning all entries. Cost: one dict write per record, one extra pointer per entry in memory.

---

## Performance

20,000 `suggest()` calls across 10 query prefixes against the full 48,032-word vocabulary (wordfreq-derived, MIT licensed). Benchmark output from the latest CI run is available as a downloadable artifact in [GitHub Actions](https://github.com/bonnie-mcconnell/adaptive-autocomplete/actions) - click the most recent passing run and download `benchmark-results`. Run `make benchmark` to reproduce locally.

Numbers below are from the CI benchmark artifact (Linux, CPython 3.12). Local results will vary by platform and CPU - run `make benchmark` to measure your own environment.

| Preset | avg | p50 | p99 | Notes |
|--------|-----|-----|-----|-------|
| stateless | ~82µs | ~73µs | ~124µs | Frequency only |
| default | ~84µs | ~74µs | ~124µs | + history |
| recency | ~90µs | ~78µs | ~138µs | + decay ranker |
| production | ~600µs | ~550µs | ~1.4ms | + trigram (≥4 char prefix) |
| robust | ~400µs | ~360µs | ~900µs | SymSpell, all prefix lengths |
| bktree (1k words) | ~3ms | ~3ms | ~11ms | BK-tree, benchmarking only |
| bktree (48k words) | ~60ms | - | - | ⚠ degrades to O(n) |

Run `make benchmark` to reproduce.

**Why `FrequencyPredictor` is fast at scale.** The prefix index is pre-sorted by frequency at construction time. With 2,332 words starting with "t" in a 48k vocabulary, an unsorted index would require sorting all 2,332 candidates on every keystroke. Pre-sorting at build time reduces `predict()` to a top-N slice - O(max_results), not O(matching_words). Construction is slower, but construction happens once.

**Why `TrigramPredictor` beats `BK-tree` at scale.** The BK-tree exploits the triangle inequality to prune subtrees without visiting them. In theory this gives O(log n) average search time. In practice, at `max_distance=2` with short prefixes, the search ball covers most of the vocabulary's metric space and pruning becomes ineffective - degrading toward O(n).

The trigram index takes a different approach: precompute which words share character trigrams with the query, filter to a shortlist of ~20–100 candidates, then run exact Levenshtein only on that shortlist. Shortlist size scales with trigram overlap, not vocabulary size - so latency is effectively independent of vocabulary size above the threshold where trigrams provide discrimination (prefix length ≥ 4).

---

## What I'd change

**The default preset ignores time.** Raw selection counts don't decay. Something selected 50 times six months ago outweighs something selected twice yesterday. The `recency` and `production` presets fix this with exponential decay, but the better design would be building time-awareness into the core history model rather than patching it in the ranker layer. Timestamps are persisted correctly - the naivety is in the `default` ranker, which discards them. I shipped `default` first and built recency on top of it rather than designing the persistence layer with decay in mind from the start.

**I shipped presets before I understood what the right abstraction was.** Presets configure internal wiring - which predictors, which rankers, which weights - but the line between "what belongs in a preset" and "what belongs in a configuration parameter" was never cleanly resolved. The Python API already lets you compose engines directly without presets; they're just convenience wrappers that obscure the composability. Starting over, I'd remove presets entirely and make the direct constructor the only interface. The CHANGELOG has four entries fixing preset-related bugs that wouldn't have existed if the abstraction had been cleaner.

**`FrequencyPredictor` memory usage should be a documented constraint, not an implementation detail.** Pre-sorting the prefix index holds around 344k string references for a 48k-word vocabulary. For a server process this is fine; for an embedded context it should be explicit. I noticed it when profiling and documented it in the docstring, but it should be in the README's performance section so users encounter it before deployment, not after.

**TrigramPredictor's minimum prefix length is a real limitation I documented instead of fixing.** Below 4 characters, trigrams provide poor discrimination and the shortlist degenerates toward the full vocabulary. The fix is a hybrid: trigram matching for 4+ character prefixes, SymSpell for 1–3 character prefixes. The architecture already supports composing both predictors in the same engine - the `robust` preset does exactly this. The `production` preset doesn't, and that's a gap I wrote about rather than closed.

---

## Tests and CI

The test suite covers correctness properties rather than just happy paths:

- **BK-tree**: every query result matches brute-force linear scan exactly, across multiple queries and thresholds
- **Trigram predictor**: no false positives - every returned result must be within `max_distance` per exact Levenshtein; score formula verified per result
- **History prefix index**: indexed results must be identical to brute-force full scan under mixed-prefix workloads
- **Explain invariant**: `final_score == base_score + history_boost` for every suggestion in every preset
- **Ranker invariant**: `RuntimeError` raised if a ranker adds or removes suggestions (not `assert` - disabled under `-O`)
- **Decay double-counting regression**: `explain()` passes pre-ranking scores to each ranker so boosts are not counted twice
- **Persistence round-trip**: timestamps survive serialisation and deserialisation with sub-second accuracy
- **Schema migration**: v1 count-only history files load under v2 with epoch timestamps, treated as maximally stale by decay rankers
- **Predictor contract**: all seven predictor implementations verified against a shared invariant suite
- **Property-based tests (Hypothesis)**: four core invariants verified across thousands of generated inputs - including two floating-point precision bugs found by Hypothesis that example-based tests missed:
  - `RankingExplanation` arithmetic (`final_score == base_score + history_boost`) holds for all finite non-negative score combinations, including after `merge()` and `apply_history_boost()`
  - `LearningRanker` and `DecayRanker` never add or remove candidates, across arbitrary suggestion lists and history states
  - History prefix index always agrees with brute-force full scan, regardless of insertion order or prefix distribution

373 tests (366 unit, 7 integration). Integration tests are marked `@pytest.mark.integration` and invoke the CLI as a subprocess - run with `make test`, skipped with `make test-fast`. CI runs unit tests on every push and the full suite on every pull request, on Linux (Python 3.10–3.13) and Windows (Python 3.11–3.12) via GitHub Actions.

---

## Why I built this

I wanted to understand what a real ranking system looks like from the inside - not the ML layer on top, but the infrastructure underneath it: how candidates get generated, scored, merged, reordered, and explained.

The first version was a single function. It worked until I tried to write a test for the learning behaviour and hit a wall: there was no seam to inject a controlled history, so I couldn't isolate what I was testing. The split into prediction and ranking layers came from that problem, not from a design document.

The `explain()` bug took an embarrassingly long time to find. The invariant `final_score == base_score + history_boost` was satisfied - the numbers added up - but the engine was passing post-ranking scores into each ranker's `explain()` instead of the pre-ranking baseline. `DecayRanker` was explaining a boost it had already applied. The arithmetic checked out; the semantics were wrong. The fix required reasoning about what each number was supposed to *mean*, not just whether it was correct.

There was a worse bug before that one, and I didn't find it by reasoning - a test caught it. `record_selection()` was keying history under `ctx.text` (the raw input string) but all lookups use `ctx.prefix()` - the lowercased, normalised last-word form. The keys never matched. Every call to `record_selection()` silently succeeded, every test that checked the output format passed, and the entire learning system was completely disabled. Nothing crashed, nothing warned, the history file grew correctly on disk. I found it by writing a Hypothesis test that verified the prefix index and the entry list agreed, which triggered on a case where `record` and `lookup` used different key formats. The fix was one character. The silence was the problem - a system that fails loudly is much easier to debug than one that appears to work.

The performance problem was different. I had a prefix index and thought it was fast. Then I switched from a 312-word toy vocabulary to 48,032 real words and the latency went from 65µs to 4ms. The index was correct - O(prefix_length) lookup - but it was returning all matching candidates unsorted, so the ranker was sorting 2,332 words on every keystroke for the prefix "t". Pre-sorting the index at construction time is obvious once you see it. The BK-tree problem was less obvious: it's theoretically O(log n), the implementation is correct, but the pruning guarantee depends on the search ball being small relative to the vocabulary. The triangle inequality pruning works like this: if a node is at distance d from the query, any subtree at key k can only contain matches if |d - k| ≤ threshold - so you can skip entire subtrees. But at `max_distance=2` with 4-character prefixes, the search ball is so large relative to the vocabulary's metric space that almost no subtrees get pruned. The algorithm visits the right nodes, just nearly all of them. That's why the BK-tree in `bk_tree.py` cites Burkhard & Keller (1973) directly - the paper defines the pruning condition, and once you understand it precisely, the degradation is obvious rather than mysterious.
