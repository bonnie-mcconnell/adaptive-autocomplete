from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

from aac.presets import create_engine

# Six representative query prefixes - three return results, three are exact
# matches excluded by design (completing a complete word adds no information).
TEXTS = ["h", "he", "hel", "help", "hero", "hex"] * 10_000
WARMUP_TEXTS = ["h", "he", "hel"] * 1_000


def run_benchmark(name: str, texts: Iterable[str]) -> float:
    engine = create_engine(name)

    # Warm up: stabilise caches, allocations, branch predictors
    for t in WARMUP_TEXTS:
        engine.suggest(t)

    start = perf_counter()
    for t in texts:
        engine.suggest(t)
    return perf_counter() - start


def main() -> None:
    presets = ["stateless", "default", "recency", "robust"]

    print(f"Benchmarking {len(TEXTS):,} suggest() calls across 6 query prefixes\n")

    for name in presets:
        elapsed = run_benchmark(name, TEXTS)
        avg_us = (elapsed / len(TEXTS)) * 1e6
        print(f"{name:10s} | total: {elapsed:6.3f}s | avg: {avg_us:7.2f} µs/call")


if __name__ == "__main__":
    main()