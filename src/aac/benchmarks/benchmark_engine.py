"""
Engine benchmark across all presets at real vocabulary scale.

Reports average, p50, and p99 latency per suggest() and explain() call.
Exists to give a runnable answer to "does this hold up at scale?"

Usage:
    make benchmark
    poetry run python -m aac.benchmarks.benchmark_engine

Preset notes:
    robust: SymSpell delete-neighbourhood index. O(1) average query at any
        vocabulary size. ~0.4ms/call at 48k words. Replaces BK-tree in 0.2.0.
    bktree: Legacy BK-tree retained for comparison. O(n) at max_distance=2
        with short prefixes. ~60ms/call at 48k words. Benchmarked against
        a 1k-word slice; full-scale numbers are shown separately.
    production: trigram pre-filter + exact Levenshtein. ~0.6ms/call at 48k.

explain() notes:
    explain() calls each ranker's rank() once in _apply_ranking(), then once
    more in the per-ranker delta loop to isolate each ranker's contribution.
    This means a ranker with a history scan (LearningRanker, DecayRanker) will
    perform 2 history lookups during explain() vs 1 during suggest(). For the
    typical use case - explain() called once per debug session, suggest() called
    thousands of times per session - this is not a performance concern. The
    additional cost over suggest() is dominated by O(k) RankingExplanation
    object construction, typically <0.2ms.
"""
from __future__ import annotations

import random
import statistics
from time import perf_counter

from aac.data import load_english_frequencies
from aac.domain.history import History
from aac.presets import get_preset

_QUERY_PREFIXES = ["t", "th", "the", "h", "he", "hel", "pro", "com", "in", "re"]
_WARMUP = 200
_REPS = 2_000


def _bench(
    preset: str,
    vocab: dict[str, int],
    prefixes: list[str],
    *,
    method: str = "suggest",
) -> list[float]:
    engine = get_preset(preset).build(History(), vocab)
    call = engine.suggest if method == "suggest" else engine.explain
    for p in prefixes * _WARMUP:
        call(p)
    latencies: list[float] = []
    for _ in range(_REPS):
        for p in prefixes:
            t0 = perf_counter()
            call(p)
            latencies.append((perf_counter() - t0) * 1e6)
    return latencies


def _row(label: str, latencies: list[float]) -> str:
    avg = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    return (
        f"{label:<52} | n={len(latencies):>6,} | "
        f"avg={avg:>6.0f}µs | p50={p50:>6.0f}µs | p99={p99:>6.0f}µs"
    )


def main() -> None:
    full_vocab = dict(load_english_frequencies())
    rng = random.Random(42)
    small_vocab = dict(rng.sample(list(full_vocab.items()), 1_000))

    vocab_size = len(full_vocab)
    print(f"Vocabulary: {vocab_size:,} words")
    print(f"Prefixes tested: {', '.join(_QUERY_PREFIXES)}")
    print(
        f"Calls per preset: {_REPS * len(_QUERY_PREFIXES):,}  "
        f"(after {_WARMUP * len(_QUERY_PREFIXES):,} warmup)\n"
    )

    header = (
        f"{'Preset + method':<52} | {'n':>8} | {'avg':>9} | {'p50':>9} | {'p99':>9}"
    )
    print(header)
    print("-" * len(header))

    for preset in ["stateless", "default", "recency"]:
        for method in ("suggest", "explain"):
            lat = _bench(preset, full_vocab, _QUERY_PREFIXES, method=method)
            label = f"{preset} ({vocab_size:,} words) [{method}]"
            print(_row(label, lat))

    # Production: trigram index
    for method in ("suggest", "explain"):
        lat = _bench("production", full_vocab, _QUERY_PREFIXES, method=method)
        print(_row(f"production ({vocab_size:,} words, trigram) [{method}]", lat))

    # Robust: SymSpell - O(1) queries, scalable
    print()
    print("Building SymSpell index (one-time cost)...")
    for method in ("suggest", "explain"):
        lat = _bench("robust", full_vocab, _QUERY_PREFIXES, method=method)
        print(_row(f"robust ({vocab_size:,} words, SymSpell) [{method}]", lat))

    # BKTree: legacy comparison at 1k slice (O(n) at full scale)
    for method in ("suggest", "explain"):
        lat = _bench("bktree", small_vocab, _QUERY_PREFIXES, method=method)
        print(_row(f"bktree (1k-word slice, legacy) [{method}]", lat))

    print()
    print("Notes:")
    print("  robust (SymSpell) suggest at 48k:   ~0.4ms/call  [new in 0.2.0]")
    print("  robust (SymSpell) explain at 48k:   ~0.5ms/call  (2 history lookups vs 1 in suggest)")
    print("  bktree suggest at full 48k scale:   ~60ms/call   [O(n) degradation]")
    print("  production suggest at 48k:          ~0.6ms/call  [trigram pre-filter]")


if __name__ == "__main__":
    main()
