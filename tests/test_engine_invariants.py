"""
Tests for engine-enforced invariants.

The README advertises these guarantees explicitly. They must be tested.

Invariant 1: Rankers cannot add or remove suggestions.
             Enforced with RuntimeError (not assert, which -O disables).

Invariant 2: Non-finite scores are rejected with ValueError.

Invariant 3: engine.debug() returns the correct shape.

Invariant 4: engine.describe() returns complete metadata.
"""
from __future__ import annotations

import pytest

from aac.domain.types import (
    CompletionContext,
    ScoredSuggestion,
    Suggestion,
)
from aac.engine.engine import AutocompleteEngine, DescribeState
from aac.predictors.static_prefix import StaticPrefixPredictor
from aac.ranking.base import Ranker
from aac.ranking.explanation import RankingExplanation


class _AddingRanker(Ranker):
    """Ranker that illegally inserts a suggestion."""

    def rank(
        self, prefix: str, suggestions: list[ScoredSuggestion]
    ) -> list[ScoredSuggestion]:
        extra = ScoredSuggestion(suggestion=Suggestion("injected"), score=999.0)
        return list(suggestions) + [extra]

    def explain(
        self, prefix: str, suggestions: list[ScoredSuggestion]
    ) -> list[RankingExplanation]:
        return []


class _RemovingRanker(Ranker):
    """Ranker that illegally drops a suggestion."""

    def rank(
        self, prefix: str, suggestions: list[ScoredSuggestion]
    ) -> list[ScoredSuggestion]:
        return list(suggestions)[:-1]

    def explain(
        self, prefix: str, suggestions: list[ScoredSuggestion]
    ) -> list[RankingExplanation]:
        return []


class _NanRanker(Ranker):
    """Ranker that produces a NaN score."""

    def rank(
        self, prefix: str, suggestions: list[ScoredSuggestion]
    ) -> list[ScoredSuggestion]:
        return [
            ScoredSuggestion(
                suggestion=s.suggestion,
                score=float("nan"),
            )
            for s in suggestions
        ]

    def explain(
        self, prefix: str, suggestions: list[ScoredSuggestion]
    ) -> list[RankingExplanation]:
        return []


# ------------------------------------------------------------------
# Invariant 1: rankers cannot add or remove suggestions
# ------------------------------------------------------------------

def test_adding_ranker_raises_runtime_error() -> None:
    engine = AutocompleteEngine(
        predictors=[StaticPrefixPredictor(["hello", "help"])],
        ranker=_AddingRanker(),
    )
    with pytest.raises(RuntimeError, match="modified the suggestion set"):
        engine.suggest("he")


def test_removing_ranker_raises_runtime_error() -> None:
    engine = AutocompleteEngine(
        predictors=[StaticPrefixPredictor(["hello", "help"])],
        ranker=_RemovingRanker(),
    )
    with pytest.raises(RuntimeError, match="modified the suggestion set"):
        engine.suggest("he")


# ------------------------------------------------------------------
# Invariant 2: non-finite scores are rejected
# ------------------------------------------------------------------

def test_nan_score_raises_value_error() -> None:
    engine = AutocompleteEngine(
        predictors=[StaticPrefixPredictor(["hello", "help"])],
        ranker=_NanRanker(),
    )
    with pytest.raises(ValueError, match="Non-finite score"):
        engine.suggest("he")


# ------------------------------------------------------------------
# engine.debug() shape
# ------------------------------------------------------------------

def test_debug_returns_correct_shape() -> None:
    engine = AutocompleteEngine(
        predictors=[StaticPrefixPredictor(["hello", "help", "hero"])]
    )
    state = engine.debug("he")

    assert "input" in state
    assert "scored" in state
    assert "ranked" in state
    assert "suggestions" in state
    assert state["input"] == "he"
    assert isinstance(state["suggestions"], list)
    assert all(isinstance(v, str) for v in state["suggestions"])


# ------------------------------------------------------------------
# engine.describe() content
# ------------------------------------------------------------------

def test_describe_returns_predictor_names() -> None:
    engine = AutocompleteEngine(
        predictors=[StaticPrefixPredictor(["hello"])]
    )
    info: DescribeState = engine.describe()
    assert info["predictors"][0]["name"] == "static_prefix"


def test_describe_returns_ranker_names() -> None:
    engine = AutocompleteEngine(
        predictors=[StaticPrefixPredictor(["hello"])]
    )
    info: DescribeState = engine.describe()
    assert "ScoreRanker" in info["rankers"]


def test_describe_history_entries_increments_after_record() -> None:
    engine = AutocompleteEngine(
        predictors=[StaticPrefixPredictor(["hello", "help"])]
    )
    info_before: DescribeState = engine.describe()
    before: int = info_before["history_entries"]
    engine.record_selection("he", "hello")
    info_after: DescribeState = engine.describe()
    after: int = info_after["history_entries"]
    assert after == before + 1


# ------------------------------------------------------------------
# _predict_scored_unranked (internal diagnostic)
# ------------------------------------------------------------------

def test_predict_scored_unranked_bypasses_rankers() -> None:
    """Internal diagnostic should return scores without ordering guarantee."""
    engine = AutocompleteEngine(
        predictors=[StaticPrefixPredictor(["hello", "help"])]
    )
    ctx = CompletionContext("he")
    results = engine._predict_scored_unranked(ctx)
    assert len(results) == 2
    values = {r.suggestion.value for r in results}
    assert values == {"hello", "help"}