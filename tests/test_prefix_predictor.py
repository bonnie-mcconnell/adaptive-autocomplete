"""
Golden tests for StaticPrefixPredictor.

These tests lock in exact output behaviour. The word "golden" means
the expected values are pinned intentionally - if they change, it is
a deliberate breaking change, not an accident.
"""
from __future__ import annotations

from aac.domain.types import CompletionContext
from aac.predictors.static_prefix import StaticPrefixPredictor


def test_prefix_predictor_returns_matching_words() -> None:
    predictor = StaticPrefixPredictor(
        vocabulary=["hello", "help", "helium", "hero"]
    )
    ctx = CompletionContext(text="he")
    results = predictor.predict(ctx)
    values = {r.suggestion.value for r in results}
    assert values == {"hello", "help", "helium", "hero"}


def test_prefix_predictor_preserves_insertion_order() -> None:
    """
    StaticPrefixPredictor returns results in vocabulary insertion order.
    This is intentional and documented: results are deterministic given
    the same vocabulary, regardless of the query.
    """
    predictor = StaticPrefixPredictor(
        vocabulary=["hello", "help", "helium", "hero"]
    )
    ctx = CompletionContext(text="he")
    results = predictor.predict(ctx)
    assert [r.suggestion.value for r in results] == ["hello", "help", "helium", "hero"]


def test_prefix_predictor_no_match() -> None:
    predictor = StaticPrefixPredictor(vocabulary=["hello", "help"])
    ctx = CompletionContext(text="z")
    assert predictor.predict(ctx) == []


def test_prefix_predictor_scores_are_deterministic() -> None:
    predictor = StaticPrefixPredictor(vocabulary=["hello", "help"])
    ctx = CompletionContext(text="he")
    results = predictor.predict(ctx)
    assert all(r.score == 1.0 for r in results)


def test_prefix_predictor_returns_new_objects() -> None:
    predictor = StaticPrefixPredictor(vocabulary=["hello"])
    ctx = CompletionContext(text="h")
    r1 = predictor.predict(ctx)
    r2 = predictor.predict(ctx)
    assert r1 is not r2
    assert r1[0] is not r2[0]