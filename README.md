# adaptive-autocomplete

![CI](https://github.com/bonnie-mcconnell/adaptive-autocomplete/actions/workflows/ci.yml/badge.svg)

A ranking and suggestion engine that implements the full autocomplete pipeline from scratch: candidate generation, scoring, learning from user selections, and explaining every decision.

The core design separates prediction from ranking. Once they're separate, each layer can be swapped, tested, and reasoned about independently. The same structure applies to any system that generates candidates, scores them, orders them, and needs to explain why.

---

## Quick start

```bash
git clone https://github.com/bonnie-mcconnell/adaptive-autocomplete
cd adaptive-autocomplete
make install

make demo        # run the full pipeline end-to-end
make test        # 223 tests across 4 Python versions
make benchmark   # latency numbers against the full 48k vocabulary
```

Or manually:

```bash
pip install poetry && poetry install

aac suggest he                          # completions ranked by frequency
aac explain he                          # score breakdown per suggestion
aac record he hero                      # record a selection - engine learns
aac --preset production suggest hello   # typo recovery via trigram index
```

Requires Python 3.10+.

---

## Python API

```python
from aac.presets import create_engine

# One line - production preset, 48k vocabulary, persistence-ready
engine = create_engine("production")
engine.suggest("helo")           # → ['help', 'held', 'hell', 'hello', ...]
engine.record_selection("helo", "hello")   # engine learns immediately
engine.explain("helo")           # → [RankingExplanation(...), ...]
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

---

## What it does

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

$ aac record he hero
Recorded selection 'hero' for input 'he'

$ aac record he hero
Recorded selection 'hero' for input 'he'

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
hero

$ aac explain he
her            score= 20000.00 (100.0%)  freq= 20000.00  recency= 0.00
here           score=  9330.00 ( 46.7%)  freq=  9330.00  recency= 0.00
help           score=  5620.00 ( 28.1%)  freq=  5620.00  recency= 0.00
head           score=  3240.00 ( 16.2%)  freq=  3240.00  recency= 0.00
health         score=  2750.00 ( 13.8%)  freq=  2750.00  recency= 0.00
...
$ aac record he hero && aac record he hero && aac explain he
her            score= 20000.00 (100.0%)  freq= 20000.00  recency= 0.00
...
hero           score=   482.00 (  2.4%)  freq=   479.00  recency=+3.00
```

After two selections, `hero` rises into the top 10. Each selection adds 1.5 (the `HistoryPredictor` weight) to its aggregated score, visible in the `recency` column.

In the `default` preset, learning happens at the **prediction** layer - `HistoryPredictor` emits history-scored candidates weighted and aggregated with frequency scores before ranking. That is why the boost appears in `freq` rather than `recency`. The `recency` and `production` presets apply `DecayRanker` at ranking time, which is why their `recency` column is non-zero.

History persists across restarts with full ISO 8601 timestamps, so decay-based presets remain accurate after reload.

---

## The design decision that shaped everything else

The question was whether prediction and ranking should be one operation or two.

On the surface they look like one thing: text goes in, ordered suggestions come out. But they're solving different problems.

**Prediction** asks: what words plausibly complete this prefix, and how likely is each one? It's stateless. A frequency predictor doesn't know what you selected yesterday. A trie predictor doesn't know either. Given the same input, they always return the same output.

**Ranking** asks: given these candidates and their scores, what order should the user see, and how should past behaviour change that? It's stateful and is where learning lives.

Separating them means each layer has a single job. A predictor can be replaced without touching the learning logic. The engine stays thin - it orchestrates, it doesn't contain scoring or ordering logic. And the layers can be tested independently, which matters when debugging why a particular word appeared where it did.

The tradeoff is more code than a single entangled function, which is worth it.

---

## Presets

Five operating modes. **If you don't know which to use, use `production`.**

| Preset | Learns | Typo recovery | Vocabulary scale | Use when |
|--------|--------|---------------|-----------------|----------|
| `stateless` | No | No | Any | Reproducible, high-throughput; no state |
| `default` | Yes | No | Any | Learning without typo recovery |
| `recency` | Yes (decay) | No | Any | Recent selections outweigh old ones |
| `production` | Yes (decay) | Yes (trigram) | 48k+ words | **Recommended default** |
| `robust` | Yes (decay) | Yes (BK-tree) | Small only | Exact recall on curated small vocab |

`production` uses a trigram index for approximate matching. At `max_distance=2` over 48k words it takes ~600µs/call - 100x faster than the BK-tree at the same scale. The constraint: trigram matching requires prefix length ≥ 4. For 1–3 character prefixes only frequency and history signals apply.

`robust` uses a BK-tree, which gives exact recall but degrades to O(n) at `max_distance=2` with short prefixes over large vocabularies. Use it with a curated vocabulary of a few hundred words, or when exact recall on every query is required.

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

**Explanations must reconcile with scores.** `RankingExplanation` enforces `final_score == base_score + history_boost` in `__post_init__`. If any code produces an inconsistent explanation, it fails at construction, not silently downstream.

**History has one owner.** The engine holds the single `History` instance. Rankers can read it; only the engine writes to it. This makes the audit trail clean and prevents predictors from recording selections twice.

**History reads are O(k), not O(n).** `History` maintains a prefix index alongside its entry list - a `defaultdict` keyed by prefix, updated on every `record()` call. All prefix-scoped reads (`counts_for_prefix`, `entries_for_prefix`, `DecayRanker`) use the index rather than scanning all entries. Cost: one dict write per record, one extra pointer per entry in memory.

---

## Performance

20,000 `suggest()` calls across 10 query prefixes against the full 48,032-word vocabulary:

| Preset | avg | p50 | p99 | Notes |
|--------|-----|-----|-----|-------|
| stateless | ~82µs | ~73µs | ~124µs | Frequency only |
| default | ~84µs | ~74µs | ~124µs | + history |
| recency | ~90µs | ~78µs | ~138µs | + decay ranker |
| production | ~600µs | ~550µs | ~1.4ms | + trigram (≥4 char prefix) |
| robust (1k words) | ~3ms | ~3ms | ~11ms | BK-tree, small vocab only |
| robust (48k words) | ~60ms | - | - | ⚠ degrades to O(n) |

Run `make benchmark` to reproduce.

**Why `FrequencyPredictor` is fast at scale.** The prefix index is pre-sorted by frequency at construction time. With 2,332 words starting with "t" in a 48k vocabulary, an unsorted index would require sorting all 2,332 candidates on every keystroke. Pre-sorting at build time reduces `predict()` to a top-N slice - O(max_results), not O(matching_words). Construction is slower, but construction happens once.

**Why `TrigramPredictor` beats `BK-tree` at scale.** The BK-tree exploits the triangle inequality to prune subtrees without visiting them. In theory this gives O(log n) average search time. In practice, at `max_distance=2` with short prefixes, the search ball covers most of the vocabulary's metric space and pruning becomes ineffective - degrading toward O(n).

The trigram index takes a different approach: precompute which words share character trigrams with the query, filter to a shortlist of ~20–100 candidates, then run exact Levenshtein only on that shortlist. Shortlist size scales with trigram overlap, not vocabulary size - so latency is effectively independent of vocabulary size above the threshold where trigrams provide discrimination (prefix length ≥ 4).

---

## What I'd change

**The default preset ignores time.** Raw selection counts don't decay. Something selected 50 times six months ago outweighs something selected twice yesterday. The `recency` and `production` presets fix this with exponential decay, but the better design would be building time-awareness into the core history model. Timestamps are persisted correctly - the naivety is in the `default` ranker, which discards them.

**Presets are a leaky abstraction.** They configure internal wiring - which predictors, which rankers, which weights - but the line between "what's a preset" and "what's a configuration parameter" was never cleanly resolved. Starting over, I'd remove presets entirely and let callers pass predictor and ranker stacks directly. The Python API already supports this; presets are just convenience wrappers.

**`FrequencyPredictor` memory usage should be a documented constraint, not an implementation detail.** Pre-sorting the prefix index holds around 336k string references for a 48k-word vocabulary. For a server process this is fine; for an embedded context it should be explicit.

**TrigramPredictor has a minimum prefix length constraint.** Below 4 characters, trigrams provide poor discrimination and the shortlist degenerates. The right fix is a hybrid: trigram matching above 4 characters, BK-tree on a curated short-word list below it. The architecture supports composing both predictors in the same engine; the preset wiring just hasn't been done.

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
- **Predictor contract**: all six predictor implementations verified against a shared invariant suite

223 tests. CI runs on Python 3.10, 3.11, 3.12, and 3.13 via GitHub Actions.

---

## Why I built this

I wanted to understand what a real ranking system looks like from the inside - not the ML layer on top, but the infrastructure underneath it: how candidates get generated, scored, merged, reordered, and explained.

The first version was a single function. It worked until I tried to write a test for the learning behaviour and hit a wall: there was no seam to inject a controlled history, so I couldn't isolate what I was testing. The split into prediction and ranking layers came from that problem, not from a design document.

The `explain()` bug took an embarrassingly long time to find. The invariant `final_score == base_score + history_boost` was satisfied - the numbers added up - but the engine was passing post-ranking scores into each ranker's `explain()` instead of the pre-ranking baseline. `DecayRanker` was explaining a boost it had already applied. The arithmetic checked out; the semantics were wrong. The fix required reasoning about what each number was supposed to *mean*, not just whether it was correct.

The performance problem was different. I had a prefix index and thought it was fast. Then I switched from a 312-word toy vocabulary to 48,032 real words and the latency went from 65µs to 4ms. The index was correct - O(prefix_length) lookup - but it was returning all matching candidates unsorted, so the ranker was sorting 2,332 words on every keystroke for the prefix "t". Pre-sorting the index at construction time is obvious once you see it. The BK-tree problem was less obvious: it's theoretically O(log n), the implementation is correct, but the pruning guarantee depends on the search ball being small relative to the vocabulary. At `max_distance=2` with short prefixes, the search ball covers most of the metric space and the log n bound falls apart. Understanding that required reading the original 1973 paper, not just measuring the latency.
