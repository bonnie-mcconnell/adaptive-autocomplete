"""
Tests for AutocompleteEngine core behaviour.

Covers: aggregation across predictors, weighted scoring,
learning via record_selection, explain immutability,
and mutation safety.
"""
from __future__ import annotations

from collections.abc import Sequence

import pytest

from aac.domain.history import History
from aac.domain.types import (
    CompletionContext,
    ScoredSuggestion,
    Suggestion,
    WeightedPredictor,
)
from aac.engine.engine import AutocompleteEngine
from aac.predictors.frequency import FrequencyPredictor
from aac.predictors.static_prefix import StaticPrefixPredictor
from aac.predictors.trie import TriePrefixPredictor
from aac.ranking.learning import LearningRanker


class _FakePredictor:
    def __init__(self, name: str, suggestions: list[ScoredSuggestion]) -> None:
        self.name = name
        self._suggestions = suggestions

    def predict(self, ctx: CompletionContext) -> list[ScoredSuggestion]:
        return self._suggestions


def test_engine_aggregates_and_sorts() -> None:
    p1 = _FakePredictor("p1", [
        ScoredSuggestion(Suggestion("foo"), 0.2),
        ScoredSuggestion(Suggestion("bar"), 0.9),
    ])
    p2 = _FakePredictor("p2", [ScoredSuggestion(Suggestion("baz"), 0.5)])
    engine = AutocompleteEngine([p1, p2])
    assert [s.value for s in engine.suggest("x")] == ["bar", "baz", "foo"]


def test_engine_combines_frequency_and_trie() -> None:
    engine = AutocompleteEngine(predictors=[
        TriePrefixPredictor(["hello", "help"]),
        FrequencyPredictor({"hello": 10, "help": 1}),
    ])
    results = engine.suggest("he")
    assert results[0].value == "hello"


def test_engine_weighted_predictors_sum_scores() -> None:
    class _FixedPredictor:
        name = "fixed"

        def predict(self, ctx: CompletionContext) -> list[ScoredSuggestion]:
            return [ScoredSuggestion(Suggestion("hello"), score=1.0)]

    engine = AutocompleteEngine(predictors=[
        WeightedPredictor(_FixedPredictor(), weight=1.0),
        WeightedPredictor(_FixedPredictor(), weight=3.0),
    ])
    results = engine.predict_scored(CompletionContext("h"))
    assert len(results) == 1
    assert results[0].score == 4.0


def test_engine_multiple_predictors_all_results_present() -> None:
    engine = AutocompleteEngine(predictors=[
        StaticPrefixPredictor(["hello", "help"]),
        StaticPrefixPredictor(["helium"]),
    ])
    values = [r.suggestion.value for r in engine.predict_scored(CompletionContext("he"))]
    assert "hello" in values
    assert "help" in values
    assert "helium" in values


def test_engine_adapts_after_selection() -> None:
    history = History()
    engine = AutocompleteEngine(
        predictors=[StaticPrefixPredictor(["hello", "help"])],
        ranker=LearningRanker(history),
    )
    assert [s.value for s in engine.suggest("he")] == ["hello", "help"]
    engine.record_selection("he", "help")
    assert engine.suggest("he")[0].value == "help"


def test_engine_explain_does_not_mutate_history() -> None:
    history = History()
    engine = AutocompleteEngine(
        predictors=[StaticPrefixPredictor(["hello"])],
        ranker=LearningRanker(history),
    )
    engine.explain("he")
    assert history.entries() == ()


def test_engine_explain_as_dicts_has_required_keys() -> None:
    engine = AutocompleteEngine(predictors=[StaticPrefixPredictor(["hello", "help"])])
    for row in engine.explain_as_dicts("he"):
        assert {"value", "base_score", "history_boost", "final_score"} <= row.keys()


def test_engine_does_not_mutate_predictor_state() -> None:
    """Multiple predict() calls must not accumulate state."""
    engine = AutocompleteEngine(predictors=[
        StaticPrefixPredictor(["hello", "help"]),
    ])
    first = engine.predict_scored(CompletionContext("he"))
    second = engine.predict_scored(CompletionContext("he"))
    assert first == second

# ------------------------------------------------------------------
# Ranker-adds-suggestions invariant (RuntimeError path)
# ------------------------------------------------------------------

def test_engine_raises_if_ranker_adds_suggestions() -> None:
    """
    The engine must raise RuntimeError if a ranker adds suggestions.

    This is the most critical invariant in the engine: rankers may only
    reorder and rescore, never add or remove. The check uses RuntimeError
    (not assert, which is disabled under -O) so it fires in production.
    """
    from aac.domain.types import ScoredSuggestion, Suggestion
    from aac.ranking.base import Ranker
    from aac.ranking.explanation import RankingExplanation

    class _AddingRanker(Ranker):
        """Deliberately adds a suggestion - must trigger invariant check."""
        def rank(self, prefix: str, suggestions: Sequence[ScoredSuggestion]) -> list[ScoredSuggestion]:
            return list(suggestions) + [ScoredSuggestion(Suggestion("injected"), score=999.0)]

        def explain(self, prefix: str, suggestions: Sequence[ScoredSuggestion]) -> list[RankingExplanation]:
            return []

    engine = AutocompleteEngine(
        predictors=[StaticPrefixPredictor(["hello", "help"])],
        ranker=_AddingRanker(),
    )

    with pytest.raises(RuntimeError, match="modified the suggestion set"):
        engine.suggest("he")


def test_engine_raises_if_ranker_removes_suggestions() -> None:
    """The engine must raise RuntimeError if a ranker removes suggestions."""
    from aac.ranking.base import Ranker
    from aac.ranking.explanation import RankingExplanation

    class _RemovingRanker(Ranker):
        """Deliberately removes a suggestion - must trigger invariant check."""
        def rank(self, prefix: str, suggestions: Sequence[ScoredSuggestion]) -> list[ScoredSuggestion]:
            return list(suggestions)[:-1]  # drop the last one

        def explain(self, prefix: str, suggestions: Sequence[ScoredSuggestion]) -> list[RankingExplanation]:
            return []

    engine = AutocompleteEngine(
        predictors=[StaticPrefixPredictor(["hello", "help", "helium"])],
        ranker=_RemovingRanker(),
    )

    with pytest.raises(RuntimeError, match="modified the suggestion set"):
        engine.suggest("he")


class TestEngineHistoryDivergence:
    """Engine __init__ must reject divergent History instances at construction time."""

    def test_explicit_history_conflicts_with_learning_ranker(self) -> None:
        """ValueError if explicit history != a learning ranker's history."""
        from aac.domain.history import History
        from aac.domain.types import WeightedPredictor
        from aac.engine.engine import AutocompleteEngine
        from aac.predictors.frequency import FrequencyPredictor
        from aac.ranking.learning import LearningRanker

        ranker_history = History()
        engine_history = History()  # different object
        assert ranker_history is not engine_history

        with pytest.raises(ValueError, match="different History instance"):
            AutocompleteEngine(
                predictors=[WeightedPredictor(FrequencyPredictor({"a": 1}), weight=1.0)],
                ranker=LearningRanker(ranker_history),
                history=engine_history,
            )

    def test_two_learning_rankers_with_divergent_histories(self) -> None:
        """ValueError if two learning rankers each own a different History."""
        from aac.domain.history import History
        from aac.domain.types import WeightedPredictor
        from aac.engine.engine import AutocompleteEngine
        from aac.predictors.frequency import FrequencyPredictor
        from aac.ranking.learning import LearningRanker

        history_a = History()
        history_b = History()

        with pytest.raises(ValueError, match="different History instance"):
            AutocompleteEngine(
                predictors=[WeightedPredictor(FrequencyPredictor({"a": 1}), weight=1.0)],
                ranker=[LearningRanker(history_a), LearningRanker(history_b)],
            )

    def test_non_finite_score_raises(self) -> None:
        """ValueError if a ranker produces a non-finite score."""
        import math

        from aac.domain.types import (
            ScoredSuggestion,
            WeightedPredictor,
        )
        from aac.engine.engine import AutocompleteEngine
        from aac.predictors.frequency import FrequencyPredictor
        from aac.ranking.base import Ranker
        from aac.ranking.explanation import RankingExplanation

        class InfiniteRanker(Ranker):
            def rank(self, prefix: str, suggestions: Sequence[ScoredSuggestion]) -> list[ScoredSuggestion]:
                return [
                    ScoredSuggestion(suggestion=s.suggestion, score=math.inf)
                    for s in suggestions
                ]
            def explain(self, prefix: str, suggestions: Sequence[ScoredSuggestion]) -> list[RankingExplanation]:
                return []

        engine = AutocompleteEngine(
            predictors=[WeightedPredictor(FrequencyPredictor({"hello": 1}), weight=1.0)],
            ranker=InfiniteRanker(),
        )
        with pytest.raises(ValueError, match="Non-finite score"):
            engine.suggest("he")

    def test_predict_scored_unranked_returns_unranked(self) -> None:
        """_predict_scored_unranked must return scores without applying rankers."""
        from aac.domain.types import CompletionContext, WeightedPredictor
        from aac.engine.engine import AutocompleteEngine
        from aac.predictors.frequency import FrequencyPredictor

        engine = AutocompleteEngine(
            predictors=[WeightedPredictor(FrequencyPredictor({"hello": 100, "help": 50}), weight=1.0)],
        )
        ctx = CompletionContext("he")
        unranked = engine._predict_scored_unranked(ctx)
        # Should return scored suggestions without ranking order enforced
        assert len(unranked) > 0
        assert all(hasattr(s, "score") for s in unranked)
