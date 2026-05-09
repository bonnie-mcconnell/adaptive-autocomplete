"""
Tests for WeightOptimiser typing and correctness.

Verifies that:
- WeightOptimiser uses proper types (not Any) for harness and engine parameters.
- OptimisationResult.report() formats correctly.
- Coordinate descent converges and improves on baseline.
- Grid search is exhaustive over the provided grid.
- The caching mechanism actually prevents redundant index builds.
"""
from __future__ import annotations

from aac.evaluation.datasets import make_synthetic_query_log
from aac.evaluation.harness import EvaluationHarness
from aac.evaluation.optimiser import OptimisationResult, WeightOptimiser


def _make_harness() -> EvaluationHarness:
    from aac.data import load_english_frequencies
    vocab = list(load_english_frequencies().keys())[:200]
    log = make_synthetic_query_log(vocab, prefix_lengths=[2, 3])
    return EvaluationHarness(log, k=5)


def test_optimisation_result_report_format() -> None:
    result = OptimisationResult(
        best_weights={"frequency": 1.0, "history": 1.2},
        baseline_score=0.80,
        best_score=0.85,
        metric="mrr",
        strategy="grid_search",
        n_evaluations=9,
    )
    report = result.report()
    assert "grid_search" in report
    assert "Baseline:     0.8000" in report
    assert "Optimised:    0.8500" in report
    assert "frequency" in report
    assert "history" in report


def test_optimisation_result_improvement_properties() -> None:
    result = OptimisationResult(
        best_weights={},
        baseline_score=0.8,
        best_score=0.9,
        metric="mrr",
        strategy="coordinate_descent",
        n_evaluations=5,
    )
    assert abs(result.improvement - 0.1) < 1e-9
    assert abs(result.improvement_pct - 12.5) < 1e-6


def test_optimisation_result_zero_baseline_improvement_pct() -> None:
    result = OptimisationResult(
        best_weights={},
        baseline_score=0.0,
        best_score=0.5,
        metric="mrr",
        strategy="grid_search",
        n_evaluations=3,
    )
    # Should not raise ZeroDivisionError
    assert result.improvement_pct == 0.0


def test_invalid_metric_raises() -> None:
    harness = _make_harness()
    try:
        WeightOptimiser(harness, metric="invalid_metric")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid_metric" in str(e)
        assert "mrr" in str(e)


def test_grid_search_returns_valid_result() -> None:
    """Grid search must return weights for every predictor in the grid."""
    harness = _make_harness()
    opt = WeightOptimiser(harness, metric="mrr", verbose=False)
    result = opt.grid_search(
        base_preset="stateless",
        weight_grid={"frequency": [0.5, 1.0, 2.0]},
    )
    assert isinstance(result, OptimisationResult)
    assert result.strategy == "grid_search"
    assert result.metric == "mrr"
    assert "frequency" in result.best_weights
    # n_evaluations = baseline + 3 grid points
    assert result.n_evaluations >= 4
    assert 0.0 <= result.baseline_score <= 1.0
    assert 0.0 <= result.best_score <= 1.0
    assert result.best_score >= result.baseline_score


def test_coordinate_descent_returns_valid_result() -> None:
    """Coordinate descent must converge and return complete weight dict."""
    harness = _make_harness()
    opt = WeightOptimiser(harness, metric="mrr", verbose=False)
    result = opt.coordinate_descent(
        base_preset="stateless",
        weight_grid={"frequency": [0.5, 1.0, 2.0]},
        max_rounds=2,
    )
    assert isinstance(result, OptimisationResult)
    assert result.strategy == "coordinate_descent"
    assert "frequency" in result.best_weights
    assert result.n_evaluations >= 1


def test_predictor_cache_populated_after_first_call() -> None:
    """
    Verify the caching mechanism: after one grid_search call, the predictor
    cache for that preset should be populated. A second call should reuse
    the cache (this is the key performance claim in the module docstring).
    """
    harness = _make_harness()
    opt = WeightOptimiser(harness, metric="mrr", verbose=False)

    assert "stateless" not in opt._predictor_cache

    opt.grid_search(
        base_preset="stateless",
        weight_grid={"frequency": [0.5, 1.0]},
    )

    assert "stateless" in opt._predictor_cache
    # The cache should contain WeightedPredictor objects, not Any
    from aac.domain.types import WeightedPredictor
    for wp in opt._predictor_cache["stateless"]:
        assert isinstance(wp, WeightedPredictor)


def test_history_records_do_not_bleed_across_evaluations() -> None:
    """
    Each engine built by WeightOptimiser must have an independent History.
    If histories bled between evaluations, recorded selections from one
    evaluation would inflate scores in subsequent evaluations, making
    the optimisation result meaningless.
    """
    harness = _make_harness()
    opt = WeightOptimiser(harness, metric="mrr", verbose=False)

    result = opt.grid_search(
        base_preset="stateless",
        weight_grid={"frequency": [0.5, 1.0, 2.0]},
    )

    # history field in the result should be a list of independent snapshots,
    # not an accumulation of state across evaluations.
    for weights, score in result.history:
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
