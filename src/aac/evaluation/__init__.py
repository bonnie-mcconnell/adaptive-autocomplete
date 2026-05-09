"""
aac.evaluation - offline evaluation and weight optimisation.

Modules
-------
metrics
    Precision@k, MRR, NDCG - standard IR metrics computed over a query log.

harness
    EvaluationHarness: runs an engine against a labelled query log and
    returns a structured EvaluationResult with per-metric breakdowns.

datasets
    Synthetic and real query-log generators. Includes a factory for
    generating a labelled dataset from a History object (uses recorded
    selections as ground-truth relevance labels).

optimiser
    WeightOptimiser: grid-search and coordinate-descent over predictor
    weights to maximise a chosen metric (default: MRR@10).

Usage
-----
::

    from aac.evaluation import EvaluationHarness, WeightOptimiser
    from aac.presets import create_engine

    # 1. Evaluate current engine
    engine = create_engine("production")
    harness = EvaluationHarness.from_history(engine.history, min_count=2)
    result = harness.run(engine)
    print(result.summary())
    # n=142 queries @ k=10: P@k=0.214  MRR=0.847  NDCG=0.891

    # 2. Find better predictor weights via coordinate descent
    opt = WeightOptimiser(harness, metric="mrr")
    best = opt.coordinate_descent(
        base_preset="production",
        weight_grid={
            "frequency": [0.5, 1.0, 2.0],
            "history":   [0.8, 1.2, 1.6],
            "symspell":  [0.2, 0.35, 0.5],
            "trigram":   [0.2, 0.4, 0.6],
        },
    )
    print(best.best_weights)
    print(f"MRR improved {best.baseline_score:.3f} → {best.best_score:.3f}")
    print(best.report())
"""
from aac.evaluation.datasets import (
    QueryLog,
    QueryLogEntry,
    load_jsonl,
    make_query_log_from_history,
    make_synthetic_query_log,
    save_jsonl,
)
from aac.evaluation.harness import EvaluationHarness, EvaluationResult
from aac.evaluation.metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from aac.evaluation.optimiser import OptimisationResult, WeightOptimiser

__all__ = [
    "EvaluationHarness",
    "EvaluationResult",
    "OptimisationResult",
    "QueryLog",
    "QueryLogEntry",
    "WeightOptimiser",
    "load_jsonl",
    "make_query_log_from_history",
    "make_synthetic_query_log",
    "mrr_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "save_jsonl",
]
