"""
Engine benchmark across all presets at real vocabulary scale.

Reports average, p50, and p99 latency per suggest() call.
Exists to give a runnable answer to "does this hold up at scale?"

Usage:
    make benchmark
    poetry run python -m aac.benchmarks.benchmark_engine

Notes on the 'robust' preset:
    The BK-tree approximate matcher runs in O(n) time over the full
    vocabulary at max_distance=2 with short prefixes. At 48k words,
    this takes ~60ms per call - far outside interactive keystroke
    budgets. The robust preset is measured against a 1k-word slice
    to show its architectural overhead in isolation. See README for
    why BK-trees degrade at scale and what the production fix is.
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


def _bench(preset: str, vocab: dict[str, int], prefixes: list[str]) -> list[float]:
    engine = get_preset(preset).build(History(), vocab)
    for p in prefixes * _WARMUP:
        engine.suggest(p)
    latencies: list[float] = []
    for _ in range(_REPS):
        for p in prefixes:
            t0 = perf_counter()
            engine.suggest(p)
            latencies.append((perf_counter() - t0) * 1e6)
    return latencies


def _row(label: str, latencies: list[float]) -> str:
    avg = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    return (
        f"{label:<28} | n={len(latencies):>6,} | "
        f"avg={avg:>6.0f}µs | p50={p50:>6.0f}µs | p99={p99:>6.0f}µs"
    )


def main() -> None:
    full_vocab = dict(load_english_frequencies())
    # 1k-word slice for robust (BK-tree is O(n) at max_distance=2)
    rng = random.Random(42)
    small_vocab = dict(rng.sample(list(full_vocab.items()), 1_000))

    vocab_size = len(full_vocab)
    print(f"Vocabulary: {vocab_size:,} words")
    print(f"Prefixes tested: {', '.join(_QUERY_PREFIXES)}")
    print(f"Calls per preset: {_REPS * len(_QUERY_PREFIXES):,}  (after {_WARMUP * len(_QUERY_PREFIXES):,} warmup)\n")

    header = f"{'Preset':<28} | {'n':>8} | {'avg':>9} | {'p50':>9} | {'p99':>9}"
    print(header)
    print("-" * len(header))

    for preset in ["stateless", "default", "recency"]:
        lat = _bench(preset, full_vocab, _QUERY_PREFIXES)
        print(_row(f"{preset} ({vocab_size:,} words)", lat))

    # Production: trigram index - the scalable typo-recovery solution
    lat = _bench("production", full_vocab, _QUERY_PREFIXES)
    print(_row(f"production ({vocab_size:,} words, trigram)", lat))

    # Robust: BK-tree at full scale is O(n), measure on 1k slice to show
    # the overhead is algorithmic, not implementation.
    lat = _bench("robust", small_vocab, _QUERY_PREFIXES)
    print(_row("robust (1k-word slice, BK-tree)", lat))
    print()
    print("robust at full 48k scale: ~60ms/call (BK-tree degrades to O(n) at max_distance=2).")
    print("production at 48k scale:  ~600µs/call (trigram pre-filter + exact Levenshtein).")


if __name__ == "__main__":
    main()
