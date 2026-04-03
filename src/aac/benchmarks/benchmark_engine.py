from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

from aac.presets import create_engine

TEXTS = ["h", "he", "hel", "help", "hero", "hex"] * 10_000
WARMUP_TEXTS = ["h", "he", "hel"] * 1_000


def run_benchmark(name: str, texts: Iterable[str]) -> float:
    engine = create_engine(name)

    # Warm-up (stabilize caches, allocations, branch predictors)
    for t in WARMUP_TEXTS:
        engine.suggest(t)

    empty_count = 0

    start = perf_counter()
    for t in texts:
        results = engine.suggest(t)

        if not results:
            empty_count += 1

    elapsed = perf_counter() - start

    # Optional diagnostic (do NOT fail)
    if empty_count:
        print(f"  note: {empty_count:,} empty results")

    return elapsed


def main() -> None:
    presets = ["stateless", "default", "recency", "robust"]

    print(f"Benchmarking {len(TEXTS):,} suggest calls (vocabulary: 6 words)\n")

    for name in presets:
        elapsed = run_benchmark(name, TEXTS)
        avg_us = (elapsed / len(TEXTS)) * 1e6

        print(f"{name:10s} | total: {elapsed:6.3f}s | avg: {avg_us:7.2f} µs")


if __name__ == "__main__":
    main()
