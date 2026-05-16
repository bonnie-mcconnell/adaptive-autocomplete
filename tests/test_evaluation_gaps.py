"""
Coverage for remaining gaps in the evaluation module.

Tests are organized by submodule. Each test targets a specific uncovered
line and has a documented reason why that line represents a real contract:

metrics.py
  - ndcg_at_k: k=0 guard (line 119), zero-IDCG guard (line 145),
    average_precision empty-relevant guard (line 165)

harness.py
  - best_queries() (line 141), n_queries property (line 187),
    k property (line 192), from_history() return path (line 229),
    constructor empty-log ValueError (line 171-175),
    run() with zero-MRR prefix (verifies no-raise on no-match)

datasets.py
  - QueryLogEntry empty-prefix validation (line 50),
    QueryLogEntry empty-relevant validation (line 52),
    make_query_log_from_history all-below-min_count (line 120),
    max_entries subsampling (lines 138-139),
    prefix_lengths=None default (line 175),
    load_query_log_from_jsonl blank/comment skip (line 235)

optimiser.py
  - OptimisationResult.report() no-weights-changed path (line 116)
  - verbose=True paths in grid_search and coordinate_descent
    (lines 337-341, 359-363, 366-367, 418-423, 440-441, 444-448, 456, 460-461)
  - _get_cached_ranker_templates verbose path (lines 179, 185)
  - LearningRanker rebuild branch (lines 227-248)
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from aac.evaluation.datasets import (
    QueryLogEntry,
    load_jsonl,
    make_query_log_from_history,
    make_synthetic_query_log,
)
from aac.evaluation.harness import EvaluationHarness
from aac.evaluation.metrics import average_precision, ndcg_at_k
from aac.evaluation.optimiser import OptimisationResult, WeightOptimiser

# ---------------------------------------------------------------------------
# Module-level constants (computed once at import time, not per test)
# ---------------------------------------------------------------------------

def _build_small_vocab() -> list[str]:
    """Load the 50 most frequent English words for use as test vocabulary."""
    from aac.data import load_english_frequencies
    return list(load_english_frequencies().keys())[:50]


_SMALL_VOCAB: list[str] = _build_small_vocab()


def _make_harness(prefix_lengths: list[int] | None = None) -> EvaluationHarness:
    log = make_synthetic_query_log(
        _SMALL_VOCAB,
        prefix_lengths=prefix_lengths or [2, 3],
    )
    return EvaluationHarness(log, k=5)


# ===========================================================================
# metrics.py gaps
# ===========================================================================

class TestNdcgAtKGuards:
    """Tests for the three uncovered guard paths in ndcg_at_k."""

    def test_k_zero_returns_zero(self) -> None:
        """
        ndcg_at_k with k=0 must return 0.0 (line 119: early return).

        Calling with k=0 is a degenerate but valid input - the caller is
        asking for NDCG over an empty top-0 list. The correct answer is 0.0,
        not a ZeroDivisionError or silent NaN.
        """
        result = ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=0)
        assert result == pytest.approx(0.0), (
            "ndcg_at_k(k=0) must return 0.0, not raise or return NaN"
        )

    def test_empty_relevant_returns_zero(self) -> None:
        """
        ndcg_at_k with relevant=set() must return 0.0 (line 119: early return).

        An empty relevant set means there is no ground truth - every result
        is irrelevant. The metric is undefined; we return 0.0 by convention
        rather than raising, matching the behaviour of P@k and R@k.
        """
        result = ndcg_at_k(["a", "b", "c"], set(), k=3)
        assert result == pytest.approx(0.0), (
            "ndcg_at_k with empty relevant set must return 0.0"
        )

    def test_zero_idcg_returns_zero(self) -> None:
        """
        When IDCG is 0.0, ndcg_at_k must return 0.0 rather than dividing by
        zero (line 145: `if idcg == 0.0: return 0.0`).

        IDCG is 0 when _dcg(ideal_items) = 0. This happens when relevant
        items exist but none has any gain - which with binary relevance
        (grades=None) means every item in `relevant` has _gain=1 and dcg>0,
        so this path is only reachable with grades where all values are 0.

        Concretely: passing grades={item: 0.0} for all relevant items gives
        every relevant item gain=0, making both DCG and IDCG zero.
        """
        result = ndcg_at_k(
            ["a", "b"],
            {"a"},
            k=2,
            grades={"a": 0.0},  # explicit zero grade → IDCG = 0
        )
        assert result == pytest.approx(0.0), (
            "ndcg_at_k must return 0.0 when IDCG=0 (all grades are zero)"
        )


class TestAveragePrecisionGuards:
    def test_empty_relevant_returns_zero(self) -> None:
        """
        average_precision with relevant=set() must return 0.0 (line 165).

        This mirrors the same guard in precision_at_k, recall_at_k, and
        mrr_at_k. Consistent behaviour across all metric functions means
        callers can safely use an empty relevant set without special-casing.
        """
        result = average_precision(["a", "b", "c"], set(), k=3)
        assert result == pytest.approx(0.0), (
            "average_precision with empty relevant set must return 0.0"
        )


# ===========================================================================
# harness.py gaps
# ===========================================================================

class TestEvaluationHarnessProperties:
    """Tests for EvaluationHarness API surface not covered by existing tests."""

    def test_best_queries_returns_sorted_by_mrr(self) -> None:
        """
        best_queries() must return queries sorted by MRR descending (line 141).

        This is the mirror of worst_queries(). A system that surfaces the
        best-performing queries lets engineers verify where the engine excels,
        complementing the worst-query debugging use case.
        """
        from aac.presets import create_engine
        harness = _make_harness()
        engine = create_engine("stateless")
        result = harness.run(engine)

        best = result.best_queries(n=3)
        assert len(best) <= 3
        # MRR must be non-increasing
        mrrs = [q.mrr for q in best]
        assert mrrs == sorted(mrrs, reverse=True), (
            f"best_queries() must be sorted by MRR descending, got: {mrrs}"
        )

    def test_n_queries_property(self) -> None:
        """
        EvaluationHarness.n_queries must return len(log) (line 187).

        Used by WeightOptimiser to estimate evaluation time and display
        progress. An incorrect count would produce misleading time estimates
        in verbose mode.
        """
        log = make_synthetic_query_log(_SMALL_VOCAB, prefix_lengths=[2])
        harness = EvaluationHarness(log, k=5)
        assert harness.n_queries == len(log), (
            f"n_queries must equal len(log), got {harness.n_queries} vs {len(log)}"
        )

    def test_k_property(self) -> None:
        """
        EvaluationHarness.k must return the configured evaluation depth (line 192).

        The k value is a critical parameter - results at k=5 and k=10 are
        not comparable. Exposing it as a property lets callers verify they
        are using the harness they intended.
        """
        log = make_synthetic_query_log(_SMALL_VOCAB, prefix_lengths=[2])
        harness = EvaluationHarness(log, k=7)
        assert harness.k == 7

    def test_from_history_returns_harness_on_valid_input(self) -> None:
        """
        EvaluationHarness.from_history() must return a harness when history
        has sufficient entries (line 229: `return cls(log, k=k)`).

        The existing test only covers the ValueError path (empty history).
        This test verifies the success path - without it, from_history() is
        partially untested even though it is the primary user-facing API for
        building a harness from real usage data.
        """
        from aac.domain.history import History

        history = History()
        for _ in range(3):
            history.record("he", "hello")
        for _ in range(2):
            history.record("he", "help")
        history.record("wo", "world")

        harness = EvaluationHarness.from_history(history, min_count=2, k=5)
        assert isinstance(harness, EvaluationHarness)
        assert harness.n_queries >= 1
        assert harness.k == 5

    def test_constructor_rejects_empty_query_log(self) -> None:
        """
        EvaluationHarness must raise ValueError when constructed with an
        empty query log (harness.py line 171-175).

        An empty log would produce no query results, making all aggregated
        metrics undefined. Failing at construction is better than failing
        silently during run().
        """
        with pytest.raises(ValueError, match="non-empty"):
            EvaluationHarness([], k=5)  # type: ignore[arg-type]

    def test_run_with_no_matching_prefix_returns_zero_mrr(self) -> None:
        """
        run() with a prefix that matches nothing must succeed with MRR=0,
        not raise. The engine returning [] is a valid (poor) outcome -
        not an error condition.
        """
        from aac.presets import create_engine

        entry = QueryLogEntry(prefix="zzz", relevant={"zzzap"})
        harness = EvaluationHarness([entry], k=5)
        engine = create_engine("stateless")

        result = harness.run(engine)
        assert result.mean_mrr == pytest.approx(0.0), (
            "Engine returning no results for 'zzz' must produce MRR=0, not raise"
        )


# ===========================================================================
# datasets.py gaps
# ===========================================================================

class TestQueryLogEntryValidation:
    """Tests for QueryLogEntry __post_init__ validation (lines 50, 52)."""

    def test_empty_prefix_raises(self) -> None:
        """
        QueryLogEntry with an empty prefix must raise ValueError (line 50).

        A blank prefix is meaningless as an evaluation query: every word in
        the vocabulary is a valid completion for "", making the entry
        contribute nothing but noise to aggregate metrics.
        """
        with pytest.raises(ValueError, match="prefix"):
            QueryLogEntry(prefix="", relevant={"hello"})

    def test_empty_relevant_raises(self) -> None:
        """
        QueryLogEntry with an empty relevant set must raise ValueError (line 52).

        An entry with no relevant completions cannot contribute a positive
        signal to any metric. Allowing it would silently divide by |relevant|=0
        in recall_at_k and average_precision.
        """
        with pytest.raises(ValueError, match="relevant"):
            QueryLogEntry(prefix="he", relevant=set())


class TestMakeQueryLogFromHistoryEdgeCases:
    """Tests for make_query_log_from_history branches not covered by existing tests."""

    def test_all_words_below_min_count_produces_empty_log(self) -> None:
        """
        When every prefix has no word meeting min_count, the result must be
        an empty list - not a crash or a log with zero-relevant entries.

        This covers the `continue` on line 120: when `relevant_words` is
        empty for a prefix, the entry is skipped.
        """
        from aac.domain.history import History

        history = History()
        history.record("he", "hello")  # count=1, min_count=2 → skipped
        history.record("wo", "world")  # count=1, min_count=2 → skipped

        log = make_query_log_from_history(history, min_count=2)
        assert log == [], (
            "All prefixes below min_count must produce an empty log, not raise"
        )

    def test_max_entries_subsamples_deterministically(self) -> None:
        """
        make_query_log_from_history with max_entries must subsample to exactly
        that size (lines 138-139: `rng.sample(entries, max_entries)`).

        Subsampling must be deterministic: same history + same seed produces
        the same subsample. This matters for reproducible evaluation pipelines.
        """
        from aac.domain.history import History

        history = History()
        prefixes = [chr(ord("a") + i) * 2 for i in range(10)]  # aa, bb, cc...
        for prefix in prefixes:
            for _ in range(3):
                history.record(prefix, prefix + "z")

        log1 = make_query_log_from_history(history, min_count=1, max_entries=4, seed=42)
        log2 = make_query_log_from_history(history, min_count=1, max_entries=4, seed=42)

        assert len(log1) == 4, f"Expected exactly 4 entries, got {len(log1)}"
        assert [e.prefix for e in log1] == [e.prefix for e in log2], (
            "Subsampling must be deterministic for the same seed"
        )


class TestMakeSyntheticQueryLogDefaults:
    def test_prefix_lengths_none_uses_default(self) -> None:
        """
        make_synthetic_query_log(prefix_lengths=None) must use the default
        [2, 3, 4] and produce entries (line 175).

        If prefix_lengths=None were passed through as-is, iterating over it
        would raise TypeError. This test ensures the None→default substitution
        actually fires.
        """
        vocab = ["hello", "help", "hero", "world", "word"]
        log = make_synthetic_query_log(vocab, prefix_lengths=None)
        assert len(log) > 0, (
            "prefix_lengths=None must use default [2,3,4] and produce entries"
        )
        # Should have entries for prefixes of length 2, 3, and 4
        lengths = {len(e.prefix) for e in log}
        assert 2 in lengths or 3 in lengths, (
            f"Expected entries with 2- or 3-char prefixes, got lengths: {lengths}"
        )


class TestLoadQueryLogFromJsonl:
    def test_blank_and_comment_lines_are_skipped(self, tmp_path: Path) -> None:
        """
        load_query_log_from_jsonl must skip blank lines and comment lines
        (line 235: `continue` for empty/hash-prefixed lines).

        JSONL files with comments and blank lines are common in practice -
        developers add explanatory comments between entries. Silent skipping
        (not raising) is the correct behaviour.
        """
        jsonl = tmp_path / "log.jsonl"
        jsonl.write_text(
            '# This is a comment\n'
            '\n'
            '{"prefix": "he", "relevant": ["hello", "help"]}\n'
            '\n'
            '# Another comment\n'
            '{"prefix": "wo", "relevant": ["world"]}\n',
            encoding="utf-8",
        )
        log = load_jsonl(jsonl)
        assert len(log) == 2, (
            f"Blank lines and comments must be skipped, got {len(log)} entries"
        )
        assert log[0].prefix == "he"
        assert log[1].prefix == "wo"


# ===========================================================================
# optimiser.py gaps
# ===========================================================================

class TestOptimisationResultReportNoWeightsChanged:
    def test_report_no_weights_changed_branch(self) -> None:
        """
        OptimisationResult.report() with best_weights={} must use the
        '(no weights in search space changed from baseline)' branch (line 116).

        This path fires when the optimiser's search space contains no
        predictor names that are in best_weights - i.e. the search space
        was entirely outside the current engine's predictor set.
        """
        result = OptimisationResult(
            best_weights={},  # empty → no weights to display
            baseline_score=0.75,
            best_score=0.75,
            metric="mrr",
            strategy="grid_search",
            n_evaluations=1,
        )
        report = result.report()
        assert "no weights" in report.lower(), (
            f"Expected 'no weights' message in report, got:\n{report}"
        )


class TestWeightOptimiserVerbosePaths:
    """
    Tests that exercise verbose=True paths in WeightOptimiser.

    The verbose paths (lines 337-341, 359-363, 366-367, 418-423, 440-441,
    444-448, 456, 460-461, 179, 185) are inside `if self._verbose:` blocks
    and are never hit by tests that use verbose=False.

    We capture stdout rather than suppressing it - this verifies the output
    is actually written (not silently skipped) and that the format strings
    don't crash on unexpected types.
    """

    @staticmethod
    def _run_grid_verbose() -> str:
        harness = _make_harness()
        opt = WeightOptimiser(harness, metric="mrr", verbose=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            opt.grid_search(
                base_preset="stateless",
                weight_grid={"frequency": [0.5, 1.0]},
            )
        return buf.getvalue()

    @staticmethod
    def _run_coordinate_verbose() -> str:
        harness = _make_harness()
        opt = WeightOptimiser(harness, metric="mrr", verbose=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            opt.coordinate_descent(
                base_preset="stateless",
                weight_grid={"frequency": [0.5, 1.0, 2.0]},
                max_rounds=2,
            )
        return buf.getvalue()

    def test_grid_search_verbose_produces_output(self) -> None:
        """
        grid_search(verbose=True) must print progress lines covering:
        - 'Building ... preset indexes' (line 179) on first call
        - 'done.' (line 185) after index build
        - 'Grid search: N combinations' (lines 337-341)
        - 'Baseline mrr=...' (lines 341)
        - 'Done. Best:' (lines 365-367)
        """
        output = self._run_grid_verbose()
        assert "Grid search" in output, f"Expected 'Grid search' in output:\n{output}"
        assert "Baseline" in output, f"Expected 'Baseline' in output:\n{output}"
        assert "Done" in output, f"Expected 'Done' in output:\n{output}"

    def test_grid_search_verbose_index_build_message(self) -> None:
        """
        The first call to grid_search with a new preset must print the
        'Building ... preset indexes' message (lines 179, 185).

        On the second call, the cache is warm and the build branch is skipped.

        We assert "Building" specifically - not `"Building" or "stateless"`.
        The `or` fallback was vacuously true (stateless appears in every line)
        and would pass even if "Building" was never printed.
        """
        harness = _make_harness()
        opt = WeightOptimiser(harness, metric="mrr", verbose=True)

        buf = io.StringIO()
        with redirect_stdout(buf):
            opt.grid_search("stateless", {"frequency": [1.0]})
        first_output = buf.getvalue()
        assert "Building" in first_output, (
            f"Expected 'Building ... preset indexes' message on first call, got:\n{first_output}"
        )

        # Second call: cache is warm, 'Building' message must not appear
        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            opt.grid_search("stateless", {"frequency": [1.0]})
        second_output = buf2.getvalue()
        assert "Building" not in second_output, (
            "Second call must use cache and not re-build indexes"
        )

    def test_coordinate_descent_verbose_produces_output(self) -> None:
        """
        coordinate_descent(verbose=True) must print progress lines covering:
        - 'Coordinate descent:' header (lines 418-423)
        - 'Baseline mrr=...' (line 423)
        - 'Done. Best:' (lines 459-461)
        """
        output = self._run_coordinate_verbose()
        assert "Coordinate descent" in output, (
            f"Expected 'Coordinate descent' in output:\n{output}"
        )
        assert "Baseline" in output
        assert "Done" in output

    def test_coordinate_descent_verbose_convergence_message(self) -> None:
        """
        When coordinate_descent converges (no improvement in a round), it
        must print the convergence message (lines 455-456: 'converged').

        We force fast convergence by giving a single-value grid - there is
        only one weight to try, so any round after the first finds no improvement
        and triggers the convergence branch.
        """
        harness = _make_harness()
        opt = WeightOptimiser(harness, metric="mrr", verbose=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            # Single candidate weight: convergence guaranteed after round 1
            opt.coordinate_descent(
                base_preset="stateless",
                weight_grid={"frequency": [1.0]},
                max_rounds=3,
            )
        output = buf.getvalue()
        assert "converged" in output.lower(), (
            f"Expected convergence message when no improvement possible:\n{output}"
        )

    def test_coordinate_descent_verbose_round_header_printed(self) -> None:
        """
        coordinate_descent must print 'Round N' progress headers (lines 440-448)
        for each round that runs.

        Previous version asserted only `len(output) > 0`, which was a tautology
        guaranteed by the header test already passing. This test verifies the
        per-round output specifically: the 'Round N' line that appears at the
        start of each coordinate descent round and carries the weight being tried.
        """
        harness = _make_harness()
        opt = WeightOptimiser(harness, metric="mrr", verbose=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            opt.coordinate_descent(
                base_preset="stateless",
                weight_grid={"frequency": [0.1, 0.5, 1.0, 2.0, 5.0]},
                max_rounds=2,
            )
        output = buf.getvalue()
        assert "Round" in output, (
            f"Expected 'Round N' per-round header in verbose coordinate_descent output:\n{output}"
        )


class TestOptimiserLearningRankerRebuild:
    """
    Tests for the LearningRanker rebuild path in _rebuild_rankers_for_history
    (lines 227-248).

    When WeightOptimiser builds engines for a preset that uses LearningRanker
    (e.g. 'default'), it must rebuild the ranker with a fresh History per
    evaluation to prevent history bleed. Lines 236-242 are the LearningRanker
    branch that was previously uncovered because all tests used 'stateless'
    (ScoreRanker only).
    """

    def test_default_preset_grid_search_uses_fresh_history_per_eval(self) -> None:
        """
        Grid search over the 'default' preset (which uses LearningRanker)
        must produce a valid result with independent histories per engine.

        If histories bled, early evaluations would pollute later ones and
        produce non-monotone improvement curves. We verify the result is
        structurally valid; history independence is verified by the existing
        test_history_records_do_not_bleed_across_evaluations test (which
        we ensure also exercises LearningRanker by running against 'default').
        """
        harness = _make_harness()
        opt = WeightOptimiser(harness, metric="mrr", verbose=False)

        result = opt.grid_search(
            base_preset="default",
            weight_grid={"frequency": [0.5, 1.0]},
        )

        assert isinstance(result, OptimisationResult)
        assert result.strategy == "grid_search"
        assert 0.0 <= result.best_score <= 1.0
        assert 0.0 <= result.baseline_score <= 1.0
        # The LearningRanker cache must be populated
        assert "default" in opt._predictor_cache

    def test_default_preset_coordinate_descent_covers_learning_ranker_branch(self) -> None:
        """
        coordinate_descent over 'default' triggers the LearningRanker
        rebuild path (lines 236-242) on every engine build.

        Verifies that rebuilding with a fresh History doesn't corrupt ranker
        state between rounds.
        """
        harness = _make_harness()
        opt = WeightOptimiser(harness, metric="mrr", verbose=False)

        result = opt.coordinate_descent(
            base_preset="default",
            weight_grid={"frequency": [0.5, 1.0, 2.0]},
            max_rounds=1,
        )

        assert isinstance(result, OptimisationResult)
        assert result.n_evaluations >= 1


class TestPublicAPIExports:
    """Regression tests for public API surface completeness."""

    def test_average_precision_importable_from_top_level(self) -> None:
        """average_precision must be importable from aac.evaluation directly.

        Regression test: it was previously only accessible via
        aac.evaluation.metrics, which is an internal submodule.
        """
        from aac.evaluation import average_precision  # noqa: PLC0415
        assert callable(average_precision)

    def test_all_metric_functions_in_evaluation_all(self) -> None:
        """All public metric functions must appear in aac.evaluation.__all__."""
        import aac.evaluation as ev
        for name in ("precision_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k", "average_precision"):
            assert name in ev.__all__, f"{name!r} missing from aac.evaluation.__all__"
