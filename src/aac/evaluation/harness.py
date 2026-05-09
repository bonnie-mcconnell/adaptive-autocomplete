"""
EvaluationHarness: run an engine against a labelled query log.

The harness evaluates a fully-configured engine on a QueryLog and
returns an EvaluationResult with per-metric, per-preset, and
per-query-length breakdowns.

Typical use
-----------
::

    from aac.evaluation import EvaluationHarness, make_query_log_from_history
    from aac.presets import create_engine

    engine = create_engine("production")
    log = make_query_log_from_history(engine.history, min_count=2)
    result = EvaluationHarness(log).run(engine)
    print(result.summary())
    print(result.to_markdown_table())

Design notes
------------
- The harness is stateless after construction: ``run()`` can be called
  multiple times with different engines without rebuilding the harness.
- Query log entries where the engine returns no results are counted as
  zero-score queries, not skipped. Skipping them would inflate metrics.
- The ``k`` parameter defaults to 10, matching standard IR evaluation.
  Use k=1 to evaluate "does the top result matter?", k=5 for typical
  autocomplete dropdowns.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aac.evaluation.datasets import QueryLog, QueryLogEntry
from aac.evaluation.metrics import (
    average_precision,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

if TYPE_CHECKING:
    from aac.engine.engine import AutocompleteEngine


@dataclass
class QueryResult:
    """Result for a single query."""
    entry: QueryLogEntry
    ranked: list[str]
    precision: float
    recall: float
    mrr: float
    ndcg: float
    ap: float
    hit: bool  # True if any relevant item appears in ranked list


@dataclass
class EvaluationResult:
    """
    Aggregated evaluation result for a full query log.

    Attributes:
        query_results: Per-query results (one per QueryLogEntry).
        k:             Evaluation depth.
        n_queries:     Number of queries evaluated.
        mean_precision: Mean P@k across all queries.
        mean_recall:    Mean R@k across all queries.
        mean_mrr:       Mean Reciprocal Rank @ k.
        mean_ndcg:      Mean NDCG@k.
        mean_ap:        Mean Average Precision@k.
        hit_rate:       Fraction of queries with at least one relevant result.
    """
    query_results: list[QueryResult]
    k: int
    n_queries: int
    mean_precision: float
    mean_recall: float
    mean_mrr: float
    mean_ndcg: float
    mean_ap: float
    hit_rate: float
    by_prefix_length: dict[int, dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line human-readable summary of key metrics."""
        return (
            f"n={self.n_queries} queries @ k={self.k}: "
            f"P@k={self.mean_precision:.3f}  "
            f"MRR={self.mean_mrr:.3f}  "
            f"NDCG={self.mean_ndcg:.3f}  "
            f"MAP={self.mean_ap:.3f}  "
            f"HitRate={self.hit_rate:.3f}"
        )

    def to_dict(self) -> dict[str, float | int]:
        """JSON-serialisable dict of aggregate metrics."""
        return {
            "n_queries": self.n_queries,
            "k": self.k,
            "mean_precision": round(self.mean_precision, 4),
            "mean_recall": round(self.mean_recall, 4),
            "mean_mrr": round(self.mean_mrr, 4),
            "mean_ndcg": round(self.mean_ndcg, 4),
            "mean_ap": round(self.mean_ap, 4),
            "hit_rate": round(self.hit_rate, 4),
        }

    def to_markdown_table(self) -> str:
        """Markdown table of aggregate metrics. Paste directly into a README."""
        rows = [
            ("Metric", "Value"),
            ("Queries evaluated", str(self.n_queries)),
            ("Evaluation depth k", str(self.k)),
            ("Precision@k", f"{self.mean_precision:.3f}"),
            ("Recall@k", f"{self.mean_recall:.3f}"),
            ("MRR@k", f"{self.mean_mrr:.3f}"),
            ("NDCG@k", f"{self.mean_ndcg:.3f}"),
            ("MAP@k", f"{self.mean_ap:.3f}"),
            ("Hit rate", f"{self.hit_rate:.1%}"),
        ]
        width_a = max(len(r[0]) for r in rows)
        width_b = max(len(r[1]) for r in rows)
        sep = f"| {'-' * width_a} | {'-' * width_b} |"
        lines = [f"| {'Metric':<{width_a}} | {'Value':<{width_b}} |", sep]
        for label, value in rows[1:]:
            lines.append(f"| {label:<{width_a}} | {value:<{width_b}} |")
        return "\n".join(lines)

    def worst_queries(self, n: int = 10) -> list[QueryResult]:
        """Return the n queries with the lowest MRR (hardest for the engine)."""
        return sorted(self.query_results, key=lambda q: q.mrr)[:n]

    def best_queries(self, n: int = 10) -> list[QueryResult]:
        """Return the n queries with the highest MRR."""
        return sorted(self.query_results, key=lambda q: q.mrr, reverse=True)[:n]


class EvaluationHarness:
    """
    Evaluate an engine against a labelled query log.

    The harness is constructed once and can evaluate multiple engines
    without rebuilding. This is intentional: you want to compare engines
    on the same log, not rebuild the log for each engine.

    Parameters:
        query_log: The labelled dataset. See ``aac.evaluation.datasets``.
        k:         Evaluation depth (default: 10). Only the first k
                   results from each suggest() call are scored.

    Example::

        from aac.evaluation import EvaluationHarness, make_synthetic_query_log
        from aac.data import load_english_frequencies
        from aac.presets import create_engine

        vocab = list(load_english_frequencies().keys())
        log = make_synthetic_query_log(vocab[:500], prefix_lengths=[2, 3])
        harness = EvaluationHarness(log)

        for preset in ["stateless", "default", "production"]:
            engine = create_engine(preset)
            result = harness.run(engine)
            print(f"{preset:12s}: {result.summary()}")
    """

    def __init__(self, query_log: QueryLog, *, k: int = 10) -> None:
        if not query_log:
            raise ValueError(
                "EvaluationHarness requires a non-empty QueryLog. "
                "Use make_query_log_from_history() or make_synthetic_query_log()."
            )
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._log = query_log
        self._k = k

    @property
    def n_queries(self) -> int:
        """Number of queries in this harness's query log."""
        return len(self._log)

    @property
    def k(self) -> int:
        """Evaluation depth."""
        return self._k

    @classmethod
    def from_history(
        cls,
        history: object,
        *,
        k: int = 10,
        min_count: int = 1,
        max_entries: int | None = None,
    ) -> EvaluationHarness:
        """
        Build a harness directly from a History instance.

        Convenience constructor that calls ``make_query_log_from_history()``
        and wraps the result in a harness.

        Raises:
            ValueError: If the history has no entries meeting min_count.
        """
        from aac.domain.history import History
        from aac.evaluation.datasets import make_query_log_from_history
        if not isinstance(history, History):
            raise TypeError(
                f"EvaluationHarness.from_history() requires a History instance, "
                f"got {type(history).__name__!r}. "
                f"Pass engine.history or a History() object."
            )
        log = make_query_log_from_history(
            history, min_count=min_count, max_entries=max_entries
        )
        if not log:
            raise ValueError(
                "History has no entries meeting min_count requirement. "
                f"min_count={min_count}. "
                "Record some selections first, or use make_synthetic_query_log()."
            )
        return cls(log, k=k)

    def run(self, engine: AutocompleteEngine) -> EvaluationResult:
        """
        Evaluate the engine against the full query log.

        For each QueryLogEntry, calls ``engine.suggest(prefix, limit=k)``
        and computes P@k, R@k, MRR@k, NDCG@k, AP@k.

        Entries where the engine returns no results count as zero-score
        (they are NOT skipped - skipping them would inflate metrics).

        Returns:
            EvaluationResult with aggregate metrics and per-query details.
        """
        query_results: list[QueryResult] = []

        for entry in self._log:
            ranked = engine.suggest(entry.prefix, limit=self._k)

            p = precision_at_k(ranked, entry.relevant, self._k)
            r = recall_at_k(ranked, entry.relevant, self._k)
            mrr = mrr_at_k(ranked, entry.relevant, self._k)
            ndcg = ndcg_at_k(
                ranked, entry.relevant, self._k,
                grades=entry.grades if entry.grades else None,
            )
            ap = average_precision(ranked, entry.relevant, self._k)
            hit = bool(set(ranked) & entry.relevant)

            query_results.append(QueryResult(
                entry=entry,
                ranked=ranked,
                precision=p,
                recall=r,
                mrr=mrr,
                ndcg=ndcg,
                ap=ap,
                hit=hit,
            ))

        n = len(query_results)
        if n == 0:
            raise RuntimeError("No query results produced - query log may be empty")

        def _mean(values: list[float]) -> float:
            return statistics.mean(values) if values else 0.0

        result = EvaluationResult(
            query_results=query_results,
            k=self._k,
            n_queries=n,
            mean_precision=_mean([q.precision for q in query_results]),
            mean_recall=_mean([q.recall for q in query_results]),
            mean_mrr=_mean([q.mrr for q in query_results]),
            mean_ndcg=_mean([q.ndcg for q in query_results]),
            mean_ap=_mean([q.ap for q in query_results]),
            hit_rate=sum(1 for q in query_results if q.hit) / n,
            by_prefix_length=_breakdown_by_length(query_results),
        )

        return result


def _breakdown_by_length(
    query_results: list[QueryResult],
) -> dict[int, dict[str, float]]:
    """
    Break down MRR by prefix length.

    Returns {prefix_length: {metric: value}} for prefix lengths 1–6+.
    Useful for understanding whether short prefixes (hard, many candidates)
    or long prefixes (easy, few candidates) are dragging down the aggregate.
    """
    from collections import defaultdict
    buckets: dict[int, list[QueryResult]] = defaultdict(list)

    for qr in query_results:
        length = min(len(qr.entry.prefix), 6)  # cap at 6 for readability
        buckets[length].append(qr)

    breakdown: dict[int, dict[str, float]] = {}
    for length, results in sorted(buckets.items()):
        breakdown[length] = {
            "n": len(results),
            "mrr": statistics.mean(r.mrr for r in results),
            "ndcg": statistics.mean(r.ndcg for r in results),
            "hit_rate": sum(1 for r in results if r.hit) / len(results),
        }
    return breakdown
