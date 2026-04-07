# adaptive-autocomplete

A ranking and suggestion engine built to understand how autocomplete systems actually work - not by calling a library, but by implementing the full pipeline: candidate generation, scoring, learning from user selections, and explaining every decision.

The architecture separates prediction from ranking. Once they're separate, each layer can be swapped, tested, and reasoned about independently. The same structure applies to any system that generates candidates, scores them, orders them, and needs to explain why.

---

## Quick start

```bash
git clone https://github.com/bonnie-mcconnell/adaptive-autocomplete
cd adaptive-autocomplete
pip install poetry && poetry install

aac suggest he          # completions ranked by frequency
aac explain he          # score breakdown per suggestion
aac record he hero      # record a selection - engine learns from it
aac --preset robust suggest helo   # typo recovery via BK-tree
```

Requires Python 3.10+.

---

## What it does

```
$ aac suggest he
her
here
help
hello
head
heart
hear
health

$ aac explain he
her          base= 3900.00  history=    0.00  total= 3900.00  [source=score]
here         base=  970.00  history=    0.00  total=  970.00  [source=score]
help         base=  900.00  history=    0.00  total=  900.00  [source=score]
hello        base=  760.00  history=    0.00  total=  760.00  [source=score]
head         base=  720.00  history=    0.00  total=  720.00  [source=score]

$ aac record he hero
Recorded selection 'hero' for input 'he'

$ aac explain he
her          base= 3900.00  history=    0.00  total= 3900.00  [source=score]
here         base=  970.00  history=    0.00  total=  970.00  [source=score]
help         base=  900.00  history=    0.00  total=  900.00  [source=score]
hello        base=  760.00  history=    0.00  total=  760.00  [source=score]
head         base=  720.00  history=    0.00  total=  720.00  [source=score]
heart        base=  690.00  history=    0.00  total=  690.00  [source=score]
hear         base=  545.00  history=    0.00  total=  545.00  [source=score]
health       base=  435.00  history=    0.00  total=  435.00  [source=score]
heavy        base=  435.00  history=    0.00  total=  435.00  [source=score]
hero         base=  431.50  history=    0.00  total=  431.50  [source=score]
```

`hero` moved from position ~10 to position 10 with a higher score. Its base score increased from 430 to 431.5 - `HistoryPredictor` contributed 1 selection × 1.5 weight. In the `default` preset, learning happens at the prediction layer rather than the ranking layer, so the boost shows up in `base` rather than `history`. The `recency` and `robust` presets use `DecayRanker` instead, which shows a non-zero `history` column and weights recent selections more heavily.

History persists across restarts. Selections are stored with full ISO 8601 timestamps so decay-based presets remain accurate after reload.

---

## The design decision that shaped everything else

The question was whether prediction and ranking should be one operation or two.

On the surface they look like one thing: text goes in, ordered suggestions come out. But they're solving different problems:

**Prediction** asks "what words plausibly complete this prefix, and how likely is each one?" It's stateless. A frequency predictor doesn't know or care what you selected yesterday. A trie predictor doesn't know it either. Given the same input, they always return the same output.

**Ranking** asks "given these candidates and their scores, what order should the user see, and how should past behaviour change that?" It's stateful. It's where learning lives.

Separating them means each layer has a single job. A predictor can be replaced without touching the learning logic. The engine stays thin - it orchestrates, it doesn't contain scoring or ordering logic. And the layers can be tested completely independently, which matters when debugging why a particular word appeared where it did.

The tradeoff is more code than a single entangled function. It's worth it.

---

## Presets

Four operating modes:

| Preset | Learns | Typo recovery | Use when |
|--------|--------|---------------|----------|
| `stateless` | No | No | Reproducible, high-throughput results |
| `default` | Yes | No | General purpose |
| `recency` | Yes (exponential decay) | No | Recent selections should outweigh old ones |
| `robust` | Yes | Yes | Real user input that may contain typos |

`robust` runs approximate string matching via a BK-tree on every query. It recovers from mid-word typos ("helo" → "hello", "hlep" → "help") and is the only preset that catches first-character errors ("wello" → "hello"). The cost is ~1500µs vs ~65µs for the others - opt-in, not default.

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

---

## Performance

60,000 `suggest()` calls per preset, 482-word vocabulary:

| Preset | Avg latency |
|--------|-------------|
| stateless | ~65µs |
| default | ~66µs |
| recency | ~68µs |
| robust | ~1500µs |

`robust` uses a BK-tree (Burkhard-Keller, 1973) for approximate string matching. The BK-tree exploits the triangle inequality property of Levenshtein distance: if a node is distance `d` from the query, only children at keys within `[d-t, d+t]` can contain matches. This prunes large portions of the tree without evaluating them.

In practice at `max_distance=2` with short prefixes, the search ball covers ~75% of the metric space and pruning is weak. BK-tree performance is strongest when the threshold is small relative to string length. For vocabularies over ~100k words, a trigram index is the right approach - precompute trigram sets per word, use set intersection to find candidates, then run exact edit distance on the shortlist.

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
- **Persistence round-trip**: timestamps survive serialisation and deserialisation
- **Schema migration**: v1 count-only history files load correctly under v2
- **Predictor contract**: all five predictor implementations are verified against a shared invariant suite

CI runs on Python 3.10, 3.11, 3.12, and 3.13 via GitHub Actions.

---

## Why I built this

I wanted to understand ranking systems from the inside. There's a big gap between "I know what autocomplete does" and "I can build one that handles learning, explainability, and performance tradeoffs in a principled way." This project is the attempt to close that gap. I learned more building it than I expected to.
