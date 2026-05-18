"""Offline evaluation and weight optimisation for autocomplete engines."""
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
