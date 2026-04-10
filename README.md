# adaptive-autocomplete

A ranking and suggestion engine that implements the full autocomplete pipeline from scratch: candidate generation, scoring, learning from user selections, and explaining every decision.

The core design separates prediction from ranking. Once they're separate, each layer can be swapped, tested, and reasoned about on its own. The same structure applies to any system that generates candidates, scores them, orders them, and needs to explain why.

---

## Quick start

```bash
git clone https://github.com/bonnie-mcconnell/adaptive-autocomplete
cd adaptive-autocomplete
pip install poetry && poetry install

aac suggest he                      # completions ranked by frequency
aac explain he                      # score breakdown per suggestion
aac record he hero                  # record a selection - engine learns
aac --preset robust suggest helo    # typo recovery via BK-tree
```

Requires Python 3.10+.

---

## What it does

```
$ aac suggest he
her
heap
hello
help
here
heart
heavy
health
heat
hey

$ aac record he hero
Recorded selection 'hero' for input 'he'

$ aac record he hero
Recorded selection 'hero' for input 'he'

$ aac suggest he
her
heap
hello
help
here
heart
heavy
health
heat
hero

$ aac explain he
her          base=  250.00  history=  +0.00  total=  250.00  [source=score]
heap         base=  105.00  history=  +0.00  total=  105.00  [source=score]
hello        base=  105.00  history=  +0.00  total=  105.00  [source=score]
help         base=  104.00  history=  +0.00  total=  104.00  [source=score]
here         base=  103.00  history=  +0.00  total=  103.00  [source=score]
heart        base=   96.00  history=  +0.00  total=   96.00  [source=score]
heavy        base=   95.00  history=  +0.00  total=   95.00  [source=score]
health       base=   87.00  history=  +0.00  total=   87.00  [source=score]
heat         base=   86.00  history=  +0.00  total=   86.00  [source=score]
hero         base=   73.00  history=  +0.00  total=   73.00  [source=score]
```

After two selections, `hero` moves from outside the top 10 to position 10, passing `hey`. Its base score went from 70 to 73 - each selection adds 1.5 (the `HistoryPredictor` weight) to its aggregated score. Record it a few more times and it continues to climb.

In the `default` preset, history learning happens at the **prediction** layer, not the ranking layer. `HistoryPredictor` emits history-scored candidates which are weighted and aggregated with frequency scores before ranking. That is why the boost appears in `base` rather than `history`. The `recency` and `robust` presets apply `DecayRanker` at the ranking layer instead, which is why their `history` column is non-zero and weights recent selections more heavily than old ones.

History persists across restarts with full ISO 8601 timestamps, so decay-based presets remain accurate after reload.

---

## The design decision that shaped everything else

The question was whether prediction and ranking should be one operation or two.

On the surface they look like one thing: text goes in, ordered suggestions come out. But they're solving different problems.

**Prediction** asks: "what words plausibly complete this prefix, and how likely is each one?" It's stateless. A frequency predictor doesn't know what you selected yesterday. A trie predictor doesn't know either. Given the same input, they always return the same output.

**Ranking** asks: "given these candidates and their scores, what order should the user see, and how should past behaviour change that?" It's stateful and is where learning lives.

Separating them means each layer has a single job. A predictor can be replaced without touching the learning logic. The engine stays thin - it orchestrates, it doesn't contain scoring or ordering logic. And the layers can be tested independently, which matters when debugging why a particular word appeared where it did.

The tradeoff is more code than a single entangled function, which is worth it.

---

## Presets

Four operating modes:

| Preset | Learns | Typo recovery | Use when |
|--------|--------|---------------|----------|
| `stateless` | No | No | Reproducible, high-throughput results |
| `default` | Yes | No | General purpose |
| `recency` | Yes (exponential decay) | No | Recent selections should outweigh old ones |
| `robust` | Yes | Yes | Real user input that may contain typos |

`robust` runs approximate string matching via a BK-tree on every query. It recovers from mid-word typos (`helo` → `hello`, `hlep` → `help`) and is the only preset that catches first-character errors (`wello` → `hello`). The cost is ~1500µs vs ~65µs for the others - opt-in, not default.

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

**History has one owner.** The engine holds the single `History` instance. Rankers can read it, only the engine writes to it. This makes the audit trail clean and prevents predictors from recording selections twice.

---

## Performance

60,000 `suggest()` calls across 6 query prefixes:

| Preset | Avg latency |
|--------|-------------|
| stateless | ~65µs |
| default | ~66µs |
| recency | ~68µs |
| robust | ~1500µs |

`robust` uses a BK-tree (Burkhard-Keller, 1973) for approximate string matching. The BK-tree exploits the triangle inequality property of Levenshtein distance: if a node is distance `d` from the query, only children at keys within `[d-t, d+t]` can contain matches. This prunes large portions of the tree without evaluating them.

At `max_distance=2` with short prefixes over this vocabulary, the search visits around 75% of nodes - pruning is weak because the search ball covers most of the metric space. BK-tree performance is strongest when the threshold is small relative to string length. For vocabularies over ~100k words, a trigram index is the right next step: precompute trigram sets per word, use set intersection to find candidates, then run exact edit distance on the shortlist.

---

## What I'd change

**The default preset ignores time.** Raw selection counts don't decay. Something selected 50 times six months ago outweighs something selected twice yesterday. The `recency` preset fixes this with exponential decay, but the better design would be building time-awareness into the core history model rather than routing around it with a preset. Timestamps are persisted correctly - the naivety is in the `default` ranker, which discards them.

**Presets are a leaky abstraction.** They configure internal wiring - which predictors, which rankers, which weights - but the line between "what's a preset" and "what's a configuration parameter" was never cleanly resolved. Starting over, I'd remove presets entirely and let callers pass their own predictor and ranker stacks directly.

**Edit distance at scale needs a trigram index.** The BK-tree is correct and theoretically O(log n), but degrades at high thresholds relative to query length. The implementation covers this vocabulary; it would not cover a 100k-word corpus without a structural change.

---

## Tests and CI

The test suite covers correctness properties rather than just happy paths:

- **BK-tree**: every query result matches brute-force linear scan exactly, across multiple queries and thresholds
- **Explain invariant**: `final_score == base_score + history_boost` for every suggestion in every preset
- **Ranker invariant**: `RuntimeError` is raised if a ranker adds or removes suggestions
- **Decay double-counting regression**: `explain()` passes pre-ranking scores to each ranker so boosts aren't counted twice
- **Persistence round-trip**: timestamps survive serialisation and deserialisation with sub-second accuracy
- **Schema migration**: v1 count-only history files load correctly under v2, with epoch timestamps so decay treats them as maximally stale
- **Predictor contract**: all predictor implementations are verified against a shared invariant suite

CI runs on Python 3.10, 3.11, 3.12, and 3.13 via GitHub Actions.

---

## Build Process

I originally wrote prediction and ranking as one function that took a prefix and returned ordered strings. It worked, until I tried to write a test for the learning behaviour and couldn't, because there was no seam to inject a controlled history. The separation into distinct layers came from that constraint, not from reading about design patterns first.

The other thing that surprised me was how long the `explain()` bug stayed hidden. The invariant `final_score == base_score + history_boost` was satisfied - the numbers added up - but the engine was passing post-ranking scores into each ranker's `explain()` instead of the pre-ranking baseline. So `DecayRanker` was explaining a boost it had already applied, and the invariant check couldn't catch it because it only verified arithmetic, not whether the numbers meant what they were supposed to mean.