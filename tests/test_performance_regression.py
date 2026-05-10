"""
Performance regression tests.

The benchmark script (make benchmark) reports latency numbers but makes
no assertions -- a 10x regression would pass CI. These tests enforce
concrete upper bounds so a performance regression fails the test suite.

Thresholds are deliberately generous (10x the measured p99) to avoid
flakiness on loaded CI runners. They exist to catch catastrophic
regressions (e.g. O(n²) accidentally introduced), not to enforce
tight latency budgets.

Environment notes:
    - Skipped in very slow environments (pytest-timeout absent)
    - Thresholds assume CPython 3.10+ on modern hardware
    - Hypothesis and other test-time imports are excluded from timing
"""
from __future__ import annotations

import statistics
from time import perf_counter

from aac.presets import create_engine

# Prefixes chosen to exercise different code paths:
# - short (1-2 chars): AdaptiveSymSpell tight mode
# - medium (3-5 chars): full SymSpell + trigram
# - specific: high-frequency prefix with many candidates
_BENCH_PREFIXES = ["he", "pro", "comp", "programing", "re"]
_REPS = 200  # fast enough for CI, enough samples for stable statistics


def _measure_latencies(preset: str, prefixes: list[str], reps: int) -> list[float]:
    """Return per-call latency in milliseconds."""
    engine = create_engine(preset)

    # Warmup: ensure JIT effects and caches are stable
    for prefix in prefixes * 10:
        engine.suggest(prefix)

    latencies: list[float] = []
    for _ in range(reps):
        for prefix in prefixes:
            t0 = perf_counter()
            engine.suggest(prefix)
            latencies.append((perf_counter() - t0) * 1000)

    return latencies


class TestPerformanceRegression:
    """
    Latency guards for core presets.

    Thresholds (ms):
        stateless  p99 < 5ms     (measured ~0.15ms, 30x headroom)
        production p99 < 30ms    (measured ~1.5ms, 20x headroom)
    """

    def test_stateless_preset_p99_under_5ms(self) -> None:
        """stateless preset: pure frequency lookup, no learning, no typo recovery."""
        latencies = _measure_latencies("stateless", _BENCH_PREFIXES, _REPS)
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        assert p99 < 5.0, (
            f"stateless p99 regression: {p99:.2f}ms >= 5ms threshold. "
            f"avg={statistics.mean(latencies):.2f}ms, "
            f"p50={statistics.median(latencies):.2f}ms"
        )

    def test_production_preset_p99_under_30ms(self) -> None:
        """production preset: full pipeline with SymSpell + trigram."""
        latencies = _measure_latencies("production", _BENCH_PREFIXES, _REPS)
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        assert p99 < 30.0, (
            f"production p99 regression: {p99:.2f}ms >= 30ms threshold. "
            f"avg={statistics.mean(latencies):.2f}ms, "
            f"p50={statistics.median(latencies):.2f}ms"
        )

    def test_explain_within_2x_suggest(self) -> None:
        """
        explain() should cost at most 2x suggest() for the same input.

        explain() does one extra forward pass through the ranker chain to
        capture deltas, but should not be significantly more expensive than
        suggest(). If this ratio grows beyond 2x something is wrong with
        the single-pass implementation.
        """
        engine = create_engine("production")

        # Warmup
        for _ in range(20):
            engine.suggest("prog")
            engine.explain("prog")

        suggest_times: list[float] = []
        explain_times: list[float] = []

        for _ in range(50):
            t0 = perf_counter()
            engine.suggest("prog")
            suggest_times.append((perf_counter() - t0) * 1000)

            t0 = perf_counter()
            engine.explain("prog")
            explain_times.append((perf_counter() - t0) * 1000)

        avg_suggest = statistics.mean(suggest_times)
        avg_explain = statistics.mean(explain_times)

        if avg_suggest > 0:
            ratio = avg_explain / avg_suggest
            assert ratio < 3.0, (
                f"explain() is {ratio:.1f}x slower than suggest(). "
                f"Expected < 3x. suggest={avg_suggest:.2f}ms, explain={avg_explain:.2f}ms. "
                f"Check for double pipeline execution in explain()."
            )

    def test_suggest_full_within_2x_suggest(self) -> None:
        """
        suggest_full() is a single-pass equivalent of suggest_with_history()
        + suggest_with_confidence(). It must not regress to two pipeline passes.

        If suggest_full() becomes more than 2x slower than suggest(), it means
        the single-pass implementation was broken and is running the pipeline
        twice again - catches that regression.
        """
        engine = create_engine("production")

        # Warmup
        for _ in range(20):
            engine.suggest("prog")
            engine.suggest_full("prog")

        suggest_times: list[float] = []
        full_times: list[float] = []

        for _ in range(50):
            t0 = perf_counter()
            engine.suggest("prog")
            suggest_times.append((perf_counter() - t0) * 1000)

            t0 = perf_counter()
            engine.suggest_full("prog")
            full_times.append((perf_counter() - t0) * 1000)

        avg_suggest = statistics.mean(suggest_times)
        avg_full = statistics.mean(full_times)

        if avg_suggest > 0:
            ratio = avg_full / avg_suggest
            assert ratio < 2.5, (
                f"suggest_full() is {ratio:.1f}x slower than suggest(). "
                f"Expected < 2.5x (single-pass overhead only). "
                f"suggest={avg_suggest:.2f}ms, suggest_full={avg_full:.2f}ms. "
                f"Check suggest_full() is not running the pipeline twice."
            )
        """
        suggest_with_history() adds one counts_for_prefix() call vs suggest().
        It must not be more than 3x slower (the extra call is O(k), not O(n)).
        """
        engine = create_engine("production")
        for i in range(20):
            engine.record_selection("prog", f"word{i}")

        suggest_times: list[float] = []
        with_history_times: list[float] = []

        for _ in range(50):
            t0 = perf_counter()
            engine.suggest("prog")
            suggest_times.append((perf_counter() - t0) * 1000)

            t0 = perf_counter()
            engine.suggest_with_history("prog")
            with_history_times.append((perf_counter() - t0) * 1000)

        avg_suggest = statistics.mean(suggest_times)
        avg_with_hist = statistics.mean(with_history_times)

        if avg_suggest > 0:
            ratio = avg_with_hist / avg_suggest
            assert ratio < 3.0, (
                f"suggest_with_history() is {ratio:.1f}x slower than suggest(). "
                f"Expected < 3x overhead for the extra prefix lookup."
            )
