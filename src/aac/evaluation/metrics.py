"""
Standard information retrieval metrics.

All functions take:
    ranked_list:  List of suggestion values in ranked order (position 0 = rank 1).
    relevant:     Set of values that are considered relevant (ground truth).
    k:            Cut-off depth. Only the first k results are considered.

All return a float in [0.0, 1.0]. Return 0.0 if no relevant items exist.

References
----------
- Manning, Raghavan & Schutze: Introduction to Information Retrieval, Ch. 8
- https://en.wikipedia.org/wiki/Discounted_cumulative_gain
- https://en.wikipedia.org/wiki/Mean_reciprocal_rank
"""
from __future__ import annotations

import math


def precision_at_k(
    ranked_list: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """
    Fraction of the top-k results that are relevant.

    P@k = |{relevant} ∩ {ranked_list[:k]}| / k

    Example: ranked=['programming', 'program', 'progress'], relevant={'programming'},  k=3
        → 1/3 ≈ 0.333

    Returns 0.0 if k=0 or relevant is empty.
    """
    if k <= 0 or not relevant:
        return 0.0
    top_k = ranked_list[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(
    ranked_list: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """
    Fraction of relevant items found in the top-k results.

    R@k = |{relevant} ∩ {ranked_list[:k]}| / |relevant|

    Example: ranked=['programming', 'program', 'progress'], relevant={'programming', 'program'}, k=3
        → 2/2 = 1.0

    Returns 0.0 if relevant is empty.
    """
    if not relevant:
        return 0.0
    top_k = set(ranked_list[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def mrr_at_k(
    ranked_list: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """
    Reciprocal Rank of the first relevant item in the top-k results.

    MRR = 1/rank_of_first_relevant  (or 0 if no relevant item in top-k)

    This is MRR for a single query. The Mean Reciprocal Rank over a query log
    is the mean of mrr_at_k() across all queries.

    Example: ranked=['program', 'programming', 'progress'], relevant={'programming'}, k=5
        → 1/2 = 0.5  (programming is at rank 2)

    Returns 0.0 if no relevant item is in the top-k, or k=0.
    """
    if k <= 0 or not relevant:
        return 0.0
    for rank, item in enumerate(ranked_list[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_list: list[str],
    relevant: set[str],
    k: int,
    *,
    grades: dict[str, float] | None = None,
) -> float:
    """
    Normalised Discounted Cumulative Gain at depth k.

    NDCG@k = DCG@k / IDCG@k

    DCG@k = Σ gain(i) / log2(i+1)  for i in 1..k

    where gain(i) = grades[item_i] if item_i is relevant, else 0.
    If no grades are supplied, binary relevance is assumed (gain = 1 for relevant).

    IDCG@k is the DCG of the ideal (perfect) ranking.

    Example: ranked=['programming', 'program', 'progress'], relevant={'programming', 'program'}, k=3
        → DCG = 1/log2(2) + 1/log2(3) ≈ 1.000 + 0.631 = 1.631
        → IDCG = 1/log2(2) + 1/log2(3) ≈ 1.631  (ideal = same order)
        → NDCG = 1.0

    Returns 0.0 if relevant is empty or k=0.
    """
    if k <= 0 or not relevant:
        return 0.0

    def _gain(item: str) -> float:
        if item not in relevant:
            return 0.0
        if grades:
            return grades.get(item, 1.0)
        return 1.0

    def _dcg(items: list[str]) -> float:
        return sum(
            _gain(item) / math.log2(rank + 1)
            for rank, item in enumerate(items[:k], start=1)
        )

    dcg = _dcg(ranked_list)

    # Ideal DCG: sort relevant items by grade (or all gain=1 if no grades)
    if grades:
        ideal_items = sorted(relevant, key=lambda x: grades.get(x, 1.0), reverse=True)
    else:
        ideal_items = list(relevant)

    idcg = _dcg(ideal_items)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def average_precision(
    ranked_list: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """
    Average Precision (AP) at depth k.

    AP = (1/|relevant|) * Σ P@i * rel(i)  for i in 1..k

    where rel(i) = 1 if item at rank i is relevant, else 0.
    Used to compute Mean Average Precision (MAP) across a query log.

    Returns 0.0 if relevant is empty.
    """
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(ranked_list[:k], start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(relevant)
