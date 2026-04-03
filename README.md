# adaptive-autocomplete

I built this to understand how autocomplete actually works at the system level, not by calling a library, but by building the full pipeline: candidate generation, scoring, learning from selections, and explaining every decision.

The domain is autocomplete but the architecture is a general ranking pipeline. The same structure applies to search, recommendations, or any system that generates candidates, scores them, orders them, and needs to explain why.

---

## Quick start

```bash
git clone https://github.com/[your-handle]/adaptive-autocomplete
cd adaptive-autocomplete
pip install poetry && poetry install

aac suggest he          # basic completions
aac explain he          # score breakdown per suggestion
aac record he hero      # teach the engine a selection
aac --preset robust suggest helo   # typo recovery
```

Requires Python 3.10+.

---

## What it does

```
$ aac suggest he
hello
help
hero
helium

$ aac explain he
hello  base=115.00  history=  0.00  total=115.00  [source=score]
help   base= 86.00  history=  0.00  total= 86.00  [source=score]
hero   base= 50.00  history=  0.00  total= 50.00  [source=score]

$ aac record he hero
Recorded selection 'hero' for input 'he'

$ aac suggest he
hello    <- hero now ranks higher due to selection history
help
hero
helium
```

---

## The core design decision

The main thing I had to figure out was whether prediction and ranking should be the same thing or different things.

They feel like the same thing. You type a prefix; suggestions appear in order as one operation. But they're actually two separate problems:

- **Prediction** answers "what are plausible completions, and how likely is each one?" It's stateless. It doesn't know or care what you've selected before.
- **Ranking** answers "given these scored candidates, what order should we show them in, and how should past behaviour change that?" It's stateful. It's where learning happens.

Once I separated them, a lot of other things fell into place. A predictor can be completely swapped out without touching the learning logic. The engine itself stays thin. It just coordinates, it doesn't contain scoring or ordering logic. And I can test each layer independently.

The tradeoff is that this is more code than just writing one function that does everything. I think it's worth it. The alternative is a system in which everything is entangled, and you can't change one thing without breaking another.

---

## Presets

Four operating modes because different contexts need different behaviour:

| Preset | Learns | Handles typos | Use when |
|--------|--------|----------------|----------|
| `stateless` | No | No | High-throughput, results must be reproducible |
| `default` | Yes | No | General purpose |
| `recency` | Yes (with decay) | No | Recent selections should count more |
| `robust` | Yes | Yes | Real input with typos |

`robust` runs edit-distance matching per candidate, which is why it's ~5x slower than the others (~145µs vs ~30µs). I made it opt-in rather than default so you don't pay that cost unless you need it.

---

## Performance

60,000 autocomplete calls per preset:

| Preset | Avg latency |
|--------|-------------|
| stateless | ~38µs |
| default | ~30µs |
| recency | ~34µs |
| robust | ~145µs |

One thing I didn't expect: `default` is slightly faster than `stateless`. The history lookup in `default` happens to hit a warm code path that the stateless variant doesn't go through. I verified this across multiple runs: it's consistent, not noise.

These numbers were measured against a 6-word vocabulary. Latency scales with vocabulary size - robust in particular, since edit-distance runs against every word on every call.

---

## Architecture

```
User input
    ↓
CompletionContext
    ↓
Predictors  →  scored candidates (stateless, parallel-safe)
    ↓
Weighted aggregation
    ↓
Rankers  →  ordered results + learning updates
    ↓
Suggestions + explanations
```

A few invariants I enforced throughout:

**If a score exists, it can be explained.** Every suggestion carries a full breakdown: base score, history adjustment, final score, which ranker touched it. I added this early because I needed to see what the pipeline was doing while building it. It ended up being the most useful debugging tool in the project.

**Rankers can't add or remove candidates.** They can only reorder and rescore. This makes the pipeline composable: you can stack rankers without worrying about one of them silently dropping results.

**History is owned by the engine, not the rankers.** Rankers can read history but can't modify it directly. This means the learning state is always in one place and the audit trail is clean.

---

## What I'd change

**The history model is too naive.** Right now it counts raw selections, so something selected 80 times six months ago has the same weight as something selected twice yesterday. The `recency` preset fixes this with exponential decay, but the better solution is to build time-aware history into the core model from the start rather than patching it in a preset.

**The preset system is a leaky abstraction.**  Presets now accept a custom vocabulary, but the deeper problem remains: the line between "what's a preset" and "what's a configuration option" is fuzzy. If I were starting over, I'd remove presets entirely and let callers pass in their own predictor and ranker stack directly. That would make it more flexible and easier to test combinations.

**No persistence.** History resets when the process exits. For a real use case you'd need to serialise history to disk or a database. It's on the TODO list but not implemented.

---

## Tests and CI

Core logic (engine, predictors, rankers, history, explanations) is unit tested. The CLI is intentionally thin and lightly tested as the interesting behaviour lives in the layers below it.

CI runs on every push via GitHub Actions.

---

## Why I built this

I wanted to understand ranking systems from the inside. There's a big gap between "I know what autocomplete does" and "I can build one that handles learning, explainability, and performance tradeoffs in a principled way." This project is the attempt to close that gap. I learned more building it than I expected to.
