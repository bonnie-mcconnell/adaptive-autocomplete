from __future__ import annotations

import math

from aac.domain.types import CompletionContext, ScoredSuggestion, Suggestion
from aac.engine.engine import AutocompleteEngine
from aac.predictors.static_prefix import StaticPrefixPredictor
from aac.ranking.score import score_and_rank


def _scores(results: list[ScoredSuggestion]) -> list[float]:
    return [r.score for r in results]


def test_ranking_is_deterministic() -> None:
    engine = AutocompleteEngine(
        predictors=[
            StaticPrefixPredictor(vocabulary=["hello", "help", "helium"])
        ]
    )
    ctx = CompletionContext(text="he")
    first = engine.predict_scored(ctx)
    second = engine.predict_scored(ctx)
    assert first == second


def test_scores_are_finite_and_non_negative() -> None:
    results = score_and_rank(
        [
            ScoredSuggestion(suggestion=Suggestion("test"), score=1.0, explanation=None),
            ScoredSuggestion(suggestion=Suggestion("testing"), score=0.5, explanation=None),
        ]
    )
    for r in results:
        assert isinstance(r.score, float)
        assert math.isfinite(r.score)
        assert r.score >= 0.0


def test_results_are_sorted_by_score_descending() -> None:
    results = score_and_rank(
        [
            ScoredSuggestion(Suggestion("a"), 0.2, None),
            ScoredSuggestion(Suggestion("b"), 0.9, None),
            ScoredSuggestion(Suggestion("c"), 0.5, None),
        ]
    )
    scores = _scores(results)
    assert scores == sorted(scores, reverse=True)


def test_ranking_is_idempotent() -> None:
    initial = [
        ScoredSuggestion(Suggestion("a"), 0.8, None),
        ScoredSuggestion(Suggestion("b"), 0.4, None),
    ]
    once = score_and_rank(initial)
    twice = score_and_rank(once)
    assert once == twice


def test_predictors_do_not_mutate_each_other() -> None:
    engine = AutocompleteEngine(
        predictors=[
            StaticPrefixPredictor(vocabulary=["hello", "help"]),
            StaticPrefixPredictor(vocabulary=["helium"]),
        ]
    )
    ctx = CompletionContext(text="he")
    results = engine.predict_scored(ctx)
    values = [r.suggestion.value for r in results]
    assert "hello" in values
    assert "help" in values
    assert "helium" in values