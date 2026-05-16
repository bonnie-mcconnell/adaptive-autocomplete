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

    from aac.evaluation import EvaluationHarness, WeightOptimiser, make_synthetic_query_log
    from aac.data import load_english_frequencies
    from aac.presets import create_engine

    # 1. Build a query log and evaluate an engine
    vocab = list(load_english_frequencies().keys())
    log = make_synthetic_query_log(vocab, prefix_lengths=[2, 3, 4])

    engine = create_engine("production")
    harness = EvaluationHarness(log, k=10)
    result = harness.run(engine)
    print(result.summary())

    # 2. Find better predictor weights via coordinate descent
    opt = WeightOptimiser(harness, metric="mrr", verbose=False)
    best = opt.coordinate_descent(
        base_preset="production",
        weight_grid={
            "frequency": [0.5, 1.0, 2.0],
            "symspell":  [0.2, 0.35, 0.5],
        },
    )
    print(best.report())

    # 3. Build from real user history (requires ≥ min_count selections per prefix)
    #    engine = create_engine("production")
    #    ... (record user selections) ...
    #    harness = EvaluationHarness.from_history(engine.history, min_count=2)
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
from aac.evaluation.metrics import (
    average_precision,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from aac.evaluation.optimiser import OptimisationResult, WeightOptimiser

__all__ = [
    "EvaluationHarness",
    "average_precision",
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
