# Design notes

Architecture decisions, tradeoffs, and things I'd change. The README covers
what the project does; this covers why it's structured the way it is.

---

## Prediction and ranking are separate layers

The first version was a monolithic function: score predictors, apply history
boost, sort, return. It was untestable - to write a learning test you had to
mock the full prediction pipeline, including the bit you were trying to test.

Splitting into stateless `Predictor` and stateful `Ranker` meant I could
test learning in isolation from prediction. The cost is one extra abstraction
layer and an explicit contract: rankers may reorder and rescore, but may not
add or remove candidates. The engine checks this after every ranker step.

## Why `__post_init__` for the explanation invariant

`RankingExplanation` enforces `final_score == base_score + history_boost` at
construction time. The alternative - an assertion in the test suite - would
catch the violation a test run later, in a different file, with a stack trace
that doesn't point at the ranker that broke it.

Construction-time enforcement means violations surface at the call site,
with the offending ranker visible in the traceback. It also means the
invariant is documented by the type itself, not by a distant test.

## Explainability without a second pipeline pass

The naive approach is: run `suggest()` to get results, then re-run the
pipeline with instrumentation to get scores. That costs 2× and introduces
a race if history changes between the two calls.

The actual approach: capture score deltas inline as each ranker runs. One
pass gives both the final ranked order and each ranker's contribution. The
implementation is in `engine._score_with_breakdown()` and the ranker loop
in `explain()`.

## SymSpell vs BK-tree

BK-tree gives O(log n) average for edit distance queries and needs no
preprocessing. SymSpell gives O(1) average but requires building a delete
neighbourhood index at startup.

For 48k words the BK-tree takes ~600µs per query; SymSpell takes ~400µs and
the index builds in ~2s. At interactive latencies the difference is marginal,
but SymSpell is strictly faster and the 2s startup cost is paid once. The
BK-tree preset (`bktree`) is retained for benchmarking comparison - it shows
the index-free alternative at scale.

## AdaptiveSymSpell: why two indexes

At a 2-character prefix with max_distance=2, a standard SymSpell index
generates hundreds of candidates (every word whose 2-deletion neighbourhood
overlaps with any 2-deletion of "ab"). Most are nonsensical. The short-prefix
variant uses max_distance=1 for prefixes ≤3 chars, which keeps results tight
without a separate codepath - just a different index parametrised at build
time.

## Why the `recency` decay formula uses half-life rather than lambda

`e^(-lambda * t)` and `e^(-ln(2) / half_life * t)` are the same function.
Half-life is human-readable: "selections from more than 3600 seconds ago
count half as much" is immediately checkable. Lambda is not.

## History ownership

The engine holds the single authoritative `History` reference. `LearningRanker`
and `HistoryPredictor` receive the same instance at construction. `reset_history()`
propagates a new instance to all components via typed protocols. This is checked
at construction time - if a ranker or predictor receives a different `History`
than the engine owns, that's a bug, not a configuration choice.

## Why presets default to plain `History`

`History` is not thread-safe. The default is plain `History` because most
uses of this library are single-threaded (scripts, notebooks, local tools),
and wrapping unconditionally would add lock overhead and hide the concurrency
model from users who don't need it.

For multi-threaded use, pass `thread_safe=True` to `create_engine()`, or
construct a `ThreadSafeHistory` and pass it as the `history` argument. The
concurrency model is explicit, not implicit.

## What I'd change if I started over

**The `default` preset ignores time.** Raw selection counts don't decay.
Something selected 50 times six months ago outweighs something selected twice
yesterday. The `recency` and `production` presets fix this with `DecayRanker`,
but the right design bakes time-awareness into the core history model rather
than bolting it on as a ranker. I shipped the simple version first and added
decay second; it should have been the default from the start.

**Presets obscure composability.** The direct constructor is already the
better interface. Presets are convenience wrappers that hide weighting
decisions from users who might benefit from tuning them. I shipped presets
first because they're easier to explain in documentation, but that made the
composable API look like the "advanced" option when it should be the default.

**The confidence score formula is a heuristic.** The hybrid approach - raw
normalisation below 4× dominance, rank-based weighting above it - produces
reasonable output but has no principled statistical basis. A cleaner design
would model selection probability directly, something like a contextual bandit.
That's a real algorithm; this is a workaround that happens to produce sensible
numbers.

**Serialisation uses class-name heuristics.** `EngineConfig.from_json()`
reconstructs predictors and rankers by matching stored type names against
a registry. This works until someone subclasses a predictor without
registering it, at which point config round-trip silently drops it. The
correct fix is structured config objects per component rather than name
strings.
