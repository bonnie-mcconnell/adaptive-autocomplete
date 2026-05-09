"""
Engine benchmark across all presets at real vocabulary scale.

Reports average, p50, and p99 latency per suggest() and explain() call.
Results can be saved to JSON and diffed against a baseline to catch
performance regressions before they reach CI.

Usage:
    make benchmark                      # run and print
    make benchmark-save                 # run and save baseline to .benchmark_baseline.json
    make benchmark-diff                 # run and compare against saved baseline

    poetry run python -m aac.benchmarks.benchmark_engine
    poetry run python -m aac.benchmarks.benchmark_engine --save
    poetry run python -m aac.benchmarks.benchmark_engine --diff

Preset notes:
    robust: SymSpell delete-neighbourhood index. O(1) average query at any
        vocabulary size. ~0.4ms/call at 48k words. Replaces BK-tree in 0.2.0.
    bktree: Legacy BK-tree retained for comparison. O(n) at max_distance=2
        with short prefixes. ~60ms/call at 48k words. Benchmarked against
        a 1k-word slice; full-scale numbers are shown separately.
    production: trigram pre-filter + exact Levenshtein. ~0.6ms/call at 48k.

explain() notes:
    explain() runs a single forward pass through the ranker chain, capturing
    per-ranker score deltas at each step. It does NOT call _apply_ranking()
    separately - that would double the history lookup cost. The implementation:

        1. _score_with_breakdown() → pre-ranking scores + per-predictor breakdown
        2. Single loop: for each ranker, call ranker.rank(), capture delta, advance

    History-reading rankers (LearningRanker, DecayRanker) perform exactly 1
    history lookup during explain(), the same as during suggest(). The
    per-ranker caching (LearningRanker._counts(), DecayRanker._decayed_counts())
    ensures that even if explain() is called immediately after rank(), the
    cache from rank() is reused and no second scan occurs.

    Additional cost over suggest() is O(k) RankingExplanation object
    construction, typically <0.1ms.

Regression bounds:
    CI enforces p99 < 5ms for stateless and < 30ms for production. These are
    10-20x the measured values - deliberately generous to avoid flakiness on
    loaded CI runners while still catching catastrophic regressions.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from time import perf_counter

from aac.data import load_english_frequencies
from aac.domain.history import History
from aac.presets import get_preset

_QUERY_PREFIXES = ["t", "th", "the", "h", "he", "hel", "pro", "com", "in", "re"]
_WARMUP = 200
_REPS = 2_000
_BASELINE_PATH = Path(".benchmark_baseline.json")

# Regression thresholds: p99 latency in microseconds.
# These are set at 10-20× measured values to avoid CI flakiness.
# Update them if you add a new preset or fundamentally change an index.
_REGRESSION_THRESHOLDS: dict[str, float] = {
    "stateless": 5_000,
    "default": 5_000,
    "recency": 8_000,
    "robust": 15_000,
    "production": 30_000,
}


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


def _stats(latencies: list[float]) -> dict[str, float]:
    sorted_lat = sorted(latencies)
    return {
        "n": len(latencies),
        "avg": statistics.mean(latencies),
        "p50": statistics.median(latencies),
        "p99": sorted_lat[int(len(latencies) * 0.99)],
    }


def _row(label: str, latencies: list[float]) -> str:
    s = _stats(latencies)
    return (
        f"{label:<52} | n={s['n']:>6,} | "
        f"avg={s['avg']:>6.0f}µs | p50={s['p50']:>6.0f}µs | p99={s['p99']:>6.0f}µs"
    )


def _run_all(full_vocab: dict[str, int], small_vocab: dict[str, int]) -> dict[str, dict[str, float]]:
    """Run all benchmarks and return {label: stats_dict}."""
    results: dict[str, dict[str, float]] = {}

    for preset in ["stateless", "default", "recency"]:
        for method in ("suggest", "explain"):
            label = f"{preset} [{method}]"
            lat = _bench(preset, full_vocab, _QUERY_PREFIXES, method=method)
            results[label] = _stats(lat)

    for method in ("suggest", "explain"):
        label = f"production [{method}]"
        lat = _bench("production", full_vocab, _QUERY_PREFIXES, method=method)
        results[label] = _stats(lat)

    for method in ("suggest", "explain"):
        label = f"robust [{method}]"
        lat = _bench("robust", full_vocab, _QUERY_PREFIXES, method=method)
        results[label] = _stats(lat)

    for method in ("suggest", "explain"):
        label = f"bktree-1k [{method}]"
        lat = _bench("bktree", small_vocab, _QUERY_PREFIXES, method=method)
        results[label] = _stats(lat)

    return results


def _print_results(results: dict[str, dict[str, float]]) -> None:
    header = (
        f"{'Preset + method':<52} | {'n':>8} | {'avg':>9} | {'p50':>9} | {'p99':>9}"
    )
    print(header)
    print("-" * len(header))
    for label, s in results.items():
        print(
            f"{label:<52} | n={s['n']:>6,.0f} | "
            f"avg={s['avg']:>6.0f}µs | p50={s['p50']:>6.0f}µs | p99={s['p99']:>6.0f}µs"
        )


def _check_regressions(results: dict[str, dict[str, float]]) -> list[str]:
    """
    Return a list of regression messages for any preset exceeding its p99 threshold.

    Thresholds are keyed by the first word of the label (the preset name).
    Only suggest() calls are checked - explain() is expected to be slower
    and is not gated by CI.
    """
    failures: list[str] = []
    for label, s in results.items():
        if "[suggest]" not in label:
            continue
        preset = label.split(" ")[0]
        threshold = _REGRESSION_THRESHOLDS.get(preset)
        if threshold is not None and s["p99"] > threshold:
            failures.append(
                f"REGRESSION: {label}  p99={s['p99']:.0f}µs > threshold {threshold:.0f}µs"
            )
    return failures


def _diff_against_baseline(
    current: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
) -> None:
    """Print a diff table comparing current results against a saved baseline."""
    print("\nPerformance diff vs baseline:")
    print(f"  {'Label':<40} {'baseline p99':>14} {'current p99':>12} {'delta':>10} {'change':>8}")
    print(f"  {'-'*40} {'-'*14} {'-'*12} {'-'*10} {'-'*8}")

    regressions = []
    for label in sorted(set(current) | set(baseline)):
        cur = current.get(label)
        base = baseline.get(label)
        if cur is None:
            print(f"  {label:<40} {'(missing)':>14} {cur['p99']:>10.0f}µs")
            continue
        if base is None:
            print(f"  {label:<40} {'(new)':>14} {cur['p99']:>12.0f}µs")
            continue
        delta = cur["p99"] - base["p99"]
        pct = delta / max(base["p99"], 1.0) * 100
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        flag = " ⚠" if pct > 20 else ""
        if pct > 20:
            regressions.append(f"{label}: +{pct:.1f}%")
        print(
            f"  {label:<40} {base['p99']:>12.0f}µs {cur['p99']:>12.0f}µs "
            f"{delta:>+9.0f}µs {arrow}{pct:>+6.1f}%{flag}"
        )

    if regressions:
        print(f"\n{len(regressions)} regressions (>20% slower than baseline):")
        for r in regressions:
            print(f"  {r}")
        sys.exit(1)
    else:
        print("\nAll within 20% of baseline. ✓")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark adaptive-autocomplete presets.")
    parser.add_argument("--save", action="store_true", help=f"Save results as new baseline to {_BASELINE_PATH}")
    parser.add_argument("--diff", action="store_true", help=f"Compare against baseline in {_BASELINE_PATH}")
    args = parser.parse_args()

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

    print("Building SymSpell/trigram indexes (one-time cost for robust/production)...")
    results = _run_all(full_vocab, small_vocab)
    print()
    _print_results(results)

    # Regression check against hard thresholds
    failures = _check_regressions(results)
    if failures:
        print()
        for f in failures:
            print(f)
        sys.exit(1)

    print()
    print("Notes:")
    print("  robust (SymSpell) suggest at 48k:   ~0.4ms/call  [O(1) index lookup]")
    print("  robust (SymSpell) explain at 48k:   ~0.5ms/call  (1 history lookup, same as suggest)")
    print("  bktree suggest at full 48k scale:   ~60ms/call   [O(n) degradation]")
    print("  production suggest at 48k:          ~0.6ms/call  [trigram pre-filter]")

    if args.save:
        _BASELINE_PATH.write_text(json.dumps(results, indent=2))
        print(f"\nBaseline saved to {_BASELINE_PATH}")

    if args.diff:
        if not _BASELINE_PATH.exists():
            print(f"\nNo baseline found at {_BASELINE_PATH}. Run with --save first.")
            sys.exit(1)
        baseline = json.loads(_BASELINE_PATH.read_text())
        _diff_against_baseline(results, baseline)


if __name__ == "__main__":
    main()
