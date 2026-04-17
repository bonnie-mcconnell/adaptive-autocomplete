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

# ------------------------------------------------------------------
# LearningRanker returns boosted scores
# ------------------------------------------------------------------

def test_learning_ranker_rank_returns_boosted_scores() -> None:
    """rank() must return ScoredSuggestion objects with the boosted score."""
    from aac.domain.history import History
    from aac.domain.types import ScoredSuggestion, Suggestion
    from aac.ranking.learning import LearningRanker

    history = History()
    history.record("he", "hello")
    history.record("he", "hello")

    ranker = LearningRanker(history, boost=1.0)
    suggestions = [
        ScoredSuggestion(Suggestion("hello"), score=10.0),
        ScoredSuggestion(Suggestion("help"), score=8.0),
    ]

    ranked = ranker.rank("he", suggestions)
    hello_result = next(s for s in ranked if s.suggestion.value == "hello")
    help_result = next(s for s in ranked if s.suggestion.value == "help")

    # hello: boost = min(2*1.0, 1.0*10.0) = 2.0 -> score = 12.0
    assert hello_result.score == 12.0
    assert help_result.score == 8.0


# ------------------------------------------------------------------
# LearningRanker.explain() uses pre-boost scores (not post-boost)
# ------------------------------------------------------------------

def test_learning_ranker_explain_uses_pre_boost_base_score() -> None:
    """
    explain() must compute boosts against pre-ranking base scores.

    If explain() uses the post-ranked score (which already includes the boost)
    as the base for recomputing the boost, the dominance cap fires at a
    different threshold than it did during rank(), producing an explanation
    that is arithmetically inconsistent with the actual ranking decision.
    """
    from aac.domain.history import History
    from aac.domain.types import ScoredSuggestion, Suggestion
    from aac.ranking.learning import LearningRanker

    history = History()
    history.record("he", "hello")
    history.record("he", "hello")

    # dominance_ratio=0.5: boost capped at 0.5 * base_score
    ranker = LearningRanker(history, boost=5.0, dominance_ratio=0.5)

    original = [
        ScoredSuggestion(Suggestion("hello"), score=10.0),
        ScoredSuggestion(Suggestion("help"), score=8.0),
    ]

    # rank() computes boost against pre-boost base_score=10.0:
    # raw_boost = 2 * 5.0 = 10.0; capped at 0.5 * 10.0 = 5.0
    # hello final_score = 15.0
    ranked = ranker.rank("he", original)
    hello_ranked = next(s for s in ranked if s.suggestion.value == "hello")
    assert hello_ranked.score == 15.0

    # explain() must also compute boost against base_score=10.0 (pre-boost),
    # NOT against 15.0 (post-boost). Using 15.0 would give:
    # cap = 0.5 * 15.0 = 7.5 -> boost = 7.5 -> final = 22.5 (WRONG)
    exps = ranker.explain("he", original)
    hello_exp = next(e for e in exps if e.value == "hello")

    assert hello_exp.base_score == 10.0, (
        f"explain() must use pre-boost base_score=10.0, got {hello_exp.base_score}"
    )
    assert hello_exp.history_boost == 5.0, (
        f"boost must be capped at 0.5*10.0=5.0, got {hello_exp.history_boost}"
    )
    assert hello_exp.final_score == 15.0, (
        f"final must be 15.0, got {hello_exp.final_score}"
    )
