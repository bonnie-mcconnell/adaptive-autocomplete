"""
Coverage for remaining gaps in engine.py.

Tests target the following uncovered lines:
  125, 142  - construction-time History divergence errors
  240-242   - ranker invariant violation (modified suggestion set)
  271       - non-finite score guard
  419       - contribution_pct empty path in explain()
  582       - non-dominant confidence branch in describe()
  658       - non-dominant confidence branch in suggest_with_confidence()
  788       - batch_suggest()
  815-819   - batch_explain()
  849-851   - batch_suggest_async()
  943       - suggest_full() method
"""
from __future__ import annotations

from collections.abc import Sequence

import pytest

from aac.domain.history import History
from aac.domain.types import (
    ScoredSuggestion,
    Suggestion,
    WeightedPredictor,
)
from aac.engine.engine import AutocompleteEngine
from aac.predictors.frequency import FrequencyPredictor
from aac.ranking.base import Ranker
from aac.ranking.explanation import RankingExplanation
from aac.ranking.learning import LearningRanker
from aac.ranking.score import ScoreRanker

_VOCAB = {"hello": 100, "help": 80, "hero": 60, "world": 200, "word": 150}


def _basic_engine() -> AutocompleteEngine:
    """Stateless engine with a small vocabulary. Fast to construct."""
    return AutocompleteEngine(
        predictors=[WeightedPredictor(FrequencyPredictor(_VOCAB), weight=1.0)],
        ranker=None,
        history=None,
    )


def _learning_engine(history: History | None = None) -> tuple[AutocompleteEngine, History]:
    """Engine with LearningRanker. Returns (engine, shared_history)."""
    h = history or History()
    engine = AutocompleteEngine(
        predictors=[WeightedPredictor(FrequencyPredictor(_VOCAB), weight=1.0)],
        ranker=[ScoreRanker(), LearningRanker(history=h, boost=2.0)],
        history=h,
    )
    return engine, h


def _make_no_op_ranker() -> Ranker:
    """
    Minimal concrete Ranker that passes suggestions through unchanged.

    Ranker is an ABC requiring both rank() and explain(). Any test ranker
    must implement both to be instantiable. We use this as a base for
    test doubles rather than repeating the boilerplate.
    """
    class _PassThrough(Ranker):
        def rank(
            self,
            prefix: str,
            suggestions: Sequence[ScoredSuggestion],
        ) -> list[ScoredSuggestion]:
            return list(suggestions)

        def explain(
            self,
            prefix: str,
            suggestions: Sequence[ScoredSuggestion],
        ) -> list[RankingExplanation]:
            return [
                RankingExplanation.from_predictor(
                    value=s.suggestion.value,
                    score=s.score,
                    source="passthrough",
                )
                for s in suggestions
            ]

    return _PassThrough()


# ===========================================================================
# Construction-time History divergence checks (lines 125, 142)
# ===========================================================================

class TestEngineHistoryDivergenceErrors:
    """
    AutocompleteEngine enforces at construction time that all learning
    rankers share the same History instance as the engine.
    """

    def test_explicit_history_different_from_ranker_raises(self) -> None:
        """
        Passing history=A to the engine but constructing LearningRanker
        with history=B must raise ValueError (line 125).
        """
        h_engine = History()
        h_ranker = History()  # different object

        with pytest.raises(ValueError, match="different History"):
            AutocompleteEngine(
                predictors=[WeightedPredictor(FrequencyPredictor(_VOCAB), weight=1.0)],
                ranker=[ScoreRanker(), LearningRanker(history=h_ranker)],
                history=h_engine,  # diverges from h_ranker
            )

    def test_two_rankers_with_different_histories_raises(self) -> None:
        """
        When history=None is passed but two learning rankers own different
        History instances, the engine must raise ValueError (line 142).
        """
        h1 = History()
        h2 = History()  # different from h1

        class _AnotherLearner(LearningRanker):
            pass

        with pytest.raises(ValueError, match="different History"):
            AutocompleteEngine(
                predictors=[WeightedPredictor(FrequencyPredictor(_VOCAB), weight=1.0)],
                ranker=[
                    LearningRanker(history=h1),
                    _AnotherLearner(history=h2),
                ],
                history=None,
            )


# ===========================================================================
# Ranking invariant violation (lines 240-242)
# ===========================================================================

class TestRankerInvariantViolation:
    """
    AutocompleteEngine._check_ranker_invariant() must raise RuntimeError
    when a ranker modifies the suggestion set (adds or removes suggestions).
    """

    def test_ranker_that_adds_suggestion_raises(self) -> None:
        """A ranker that adds a new suggestion must trigger RuntimeError (lines 240-242)."""

        class _AddingSuggestion(Ranker):
            def rank(
                self,
                prefix: str,
                suggestions: Sequence[ScoredSuggestion],
            ) -> list[ScoredSuggestion]:
                extra = ScoredSuggestion(
                    suggestion=Suggestion(value="injected"),
                    score=999.0,
                )
                return [extra, *list(suggestions)]

            def explain(
                self,
                prefix: str,
                suggestions: Sequence[ScoredSuggestion],
            ) -> list[RankingExplanation]:
                return []

        engine = AutocompleteEngine(
            predictors=[WeightedPredictor(FrequencyPredictor(_VOCAB), weight=1.0)],
            ranker=[_AddingSuggestion()],
            history=History(),
        )
        with pytest.raises(RuntimeError, match="modified the suggestion set"):
            engine.suggest("he")

    def test_ranker_that_removes_suggestion_raises(self) -> None:
        """A ranker that removes a suggestion must raise RuntimeError."""

        class _FilteringRanker(Ranker):
            def rank(
                self,
                prefix: str,
                suggestions: Sequence[ScoredSuggestion],
            ) -> list[ScoredSuggestion]:
                return list(suggestions)[:-1]  # drop the last one

            def explain(
                self,
                prefix: str,
                suggestions: Sequence[ScoredSuggestion],
            ) -> list[RankingExplanation]:
                return []

        engine = AutocompleteEngine(
            predictors=[WeightedPredictor(FrequencyPredictor(_VOCAB), weight=1.0)],
            ranker=[_FilteringRanker()],
            history=History(),
        )
        with pytest.raises(RuntimeError, match="modified the suggestion set"):
            engine.suggest("he")


# ===========================================================================
# Non-finite score guard (line 271)
# ===========================================================================

class TestNonFiniteScoreGuard:
    """
    The engine must raise ValueError if any ranker produces a non-finite
    (NaN or ±Inf) score (line 271).
    """

    def _engine_with_ranker(self, ranker: Ranker) -> AutocompleteEngine:
        return AutocompleteEngine(
            predictors=[WeightedPredictor(FrequencyPredictor(_VOCAB), weight=1.0)],
            ranker=[ranker],
            history=History(),
        )

    def test_ranker_producing_nan_raises(self) -> None:
        class _NaNRanker(Ranker):
            def rank(
                self,
                prefix: str,
                suggestions: Sequence[ScoredSuggestion],
            ) -> list[ScoredSuggestion]:
                return [
                    ScoredSuggestion(suggestion=s.suggestion, score=float("nan"))
                    for s in suggestions
                ]

            def explain(
                self,
                prefix: str,
                suggestions: Sequence[ScoredSuggestion],
            ) -> list[RankingExplanation]:
                return []

        with pytest.raises(ValueError, match="Non-finite"):
            self._engine_with_ranker(_NaNRanker()).suggest("he")

    def test_ranker_producing_inf_raises(self) -> None:
        class _InfRanker(Ranker):
            def rank(
                self,
                prefix: str,
                suggestions: Sequence[ScoredSuggestion],
            ) -> list[ScoredSuggestion]:
                return [
                    ScoredSuggestion(suggestion=s.suggestion, score=float("inf"))
                    for s in suggestions
                ]

            def explain(
                self,
                prefix: str,
                suggestions: Sequence[ScoredSuggestion],
            ) -> list[RankingExplanation]:
                return []

        with pytest.raises(ValueError, match="Non-finite"):
            self._engine_with_ranker(_InfRanker()).suggest("he")


# ===========================================================================
# explain() contribution_pct empty path (line 419)
# ===========================================================================

class TestExplainContributionPctEmpty:
    """
    explain() sets contribution_pct={} when final_score is near zero.

    We use a custom ranker that zeroes out all scores. The near-zero
    threshold is 1e-12 - scores below this produce contribution_pct={}.
    """

    def test_explain_near_zero_score_gives_empty_contribution(self) -> None:
        class _ZeroScoreRanker(Ranker):
            def rank(
                self,
                prefix: str,
                suggestions: Sequence[ScoredSuggestion],
            ) -> list[ScoredSuggestion]:
                return [
                    ScoredSuggestion(
                        suggestion=s.suggestion,
                        score=1e-15,  # below 1e-12 threshold
                        explanation=s.explanation,
                    )
                    for s in suggestions
                ]

            def explain(
                self,
                prefix: str,
                suggestions: Sequence[ScoredSuggestion],
            ) -> list[RankingExplanation]:
                return [
                    RankingExplanation(
                        value=s.suggestion.value,
                        base_score=1e-15,
                        history_boost=0.0,
                        final_score=1e-15,
                        source="zero",
                    )
                    for s in suggestions
                ]

        engine = AutocompleteEngine(
            predictors=[WeightedPredictor(FrequencyPredictor(_VOCAB), weight=1.0)],
            ranker=[_ZeroScoreRanker()],
            history=History(),
        )

        explanations = engine.explain("he")
        assert len(explanations) > 0

        near_zero = [e for e in explanations if abs(e.final_score) < 1e-12]
        if near_zero:
            assert near_zero[0].contribution_pct == {}, (
                "Near-zero final_score must produce empty contribution_pct"
            )


# ===========================================================================
# describe() (line 582 path via suggest_with_confidence, and line 658)
# ===========================================================================

class TestDescribeMethod:
    """
    describe() returns a typed dict of engine configuration.
    It takes no arguments (not even a prefix).
    """

    def test_describe_returns_expected_keys(self) -> None:
        """describe() must return a dict with 'predictors', 'rankers', 'history_entries'."""
        engine = _basic_engine()
        desc = engine.describe()
        assert "predictors" in desc
        assert "rankers" in desc
        assert "history_entries" in desc

    def test_describe_predictor_list_shape(self) -> None:
        engine = _basic_engine()
        desc = engine.describe()
        assert isinstance(desc["predictors"], list)
        assert len(desc["predictors"]) == 1
        p = desc["predictors"][0]
        assert p["name"] == "frequency"
        assert p["weight"] == pytest.approx(1.0)

    def test_describe_reflects_history_entries(self) -> None:
        engine, h = _learning_engine()
        assert engine.describe()["history_entries"] == 0
        h.record("he", "hello")
        assert engine.describe()["history_entries"] == 1


class TestSuggestWithConfidenceNonDominant:
    """
    suggest_with_confidence() non-dominant branch (line 658).
    When score spread is flat, confidence normalises by max score.
    """

    def test_returns_unit_interval_confidences(self) -> None:
        engine = _basic_engine()
        results = engine.suggest_with_confidence("he")
        assert len(results) > 0
        for word, conf in results:
            assert isinstance(word, str)
            assert 0.0 <= conf <= 1.0, f"confidence for '{word}' is {conf}"

    def test_two_equal_predictors_produce_valid_confidences(self) -> None:
        """
        Two equal-weight predictors with identical scores produce a flat
        distribution - the non-dominant path normalises by max score.
        """
        engine = AutocompleteEngine(
            predictors=[
                WeightedPredictor(FrequencyPredictor(_VOCAB), weight=1.0),
                WeightedPredictor(FrequencyPredictor(_VOCAB), weight=1.0),
            ],
            ranker=None,
            history=None,
        )
        results = engine.suggest_with_confidence("he")
        assert len(results) > 0
        for _, conf in results:
            assert 0.0 <= conf <= 1.0


# ===========================================================================
# batch_suggest, batch_explain, batch_suggest_async (788, 815-819, 849-851)
# ===========================================================================

class TestBatchAPIs:
    """Tests for the three batch APIs."""

    def test_batch_suggest_returns_dict_keyed_by_input(self) -> None:
        """batch_suggest() must return a dict mapping each prefix to its suggestions."""
        engine = _basic_engine()
        result = engine.batch_suggest(["he", "wo", "xyz"])

        assert set(result.keys()) == {"he", "wo", "xyz"}
        assert isinstance(result["he"], list)
        assert isinstance(result["xyz"], list)
        assert all(isinstance(s, str) for s in result["he"])

    def test_batch_suggest_respects_limit(self) -> None:
        engine = _basic_engine()
        result = engine.batch_suggest(["he", "wo"], limit=1)
        assert all(len(v) <= 1 for v in result.values())

    def test_batch_suggest_empty_input(self) -> None:
        engine = _basic_engine()
        assert engine.batch_suggest([]) == {}

    def test_batch_explain_returns_ranking_explanations(self) -> None:
        """batch_explain() must return a dict of RankingExplanation lists."""
        engine = _basic_engine()
        result = engine.batch_explain(["he", "wo"])

        assert set(result.keys()) == {"he", "wo"}
        for exps in result.values():
            assert isinstance(exps, list)
            for exp in exps:
                assert isinstance(exp, RankingExplanation)

    def test_batch_explain_respects_limit(self) -> None:
        engine = _basic_engine()
        result = engine.batch_explain(["he", "wo"], limit=2)
        assert all(len(v) <= 2 for v in result.values())

    def test_batch_explain_empty_input(self) -> None:
        engine = _basic_engine()
        assert engine.batch_explain([]) == {}

    @pytest.mark.asyncio
    async def test_batch_suggest_async_matches_sync(self) -> None:
        """
        batch_suggest_async() must return the same results as batch_suggest().
        """
        engine = _basic_engine()
        prefixes = ["he", "wo", "xyz"]
        sync_result = engine.batch_suggest(prefixes)
        async_result = await engine.batch_suggest_async(prefixes)
        assert sync_result == async_result

    @pytest.mark.asyncio
    async def test_batch_suggest_async_respects_limit(self) -> None:
        engine = _basic_engine()
        result = await engine.batch_suggest_async(["he", "wo"], limit=2)
        assert all(len(v) <= 2 for v in result.values())

    @pytest.mark.asyncio
    async def test_batch_suggest_async_empty_input(self) -> None:
        engine = _basic_engine()
        assert await engine.batch_suggest_async([]) == {}


# ===========================================================================
# suggest_full (line 943)
# ===========================================================================

class TestSuggestFull:
    """
    suggest_full() returns dicts with 'word', 'count', 'confidence' keys.
    It runs the pipeline once to produce all three signals simultaneously.
    """

    def test_suggest_full_returns_expected_keys(self) -> None:
        engine = _basic_engine()
        results = engine.suggest_full("he")

        assert len(results) > 0
        for item in results:
            assert "word" in item, f"Missing 'word' key in {item}"
            assert "count" in item, f"Missing 'count' key in {item}"
            assert "confidence" in item, f"Missing 'confidence' key in {item}"

    def test_suggest_full_values_are_correct_types(self) -> None:
        engine = _basic_engine()
        for item in engine.suggest_full("he"):
            assert isinstance(item["word"], str)
            assert isinstance(item["count"], int)
            assert isinstance(item["confidence"], float)
            assert 0.0 <= item["confidence"] <= 1.0

    def test_suggest_full_order_matches_suggest(self) -> None:
        """suggest_full() results in order must match suggest() in order."""
        engine = _basic_engine()
        full = engine.suggest_full("he")
        plain = engine.suggest("he")
        assert [item["word"] for item in full] == plain

    def test_suggest_full_respects_limit(self) -> None:
        engine = _basic_engine()
        results = engine.suggest_full("he", limit=2)
        assert len(results) <= 2

    def test_suggest_full_empty_prefix_returns_empty(self) -> None:
        engine = _basic_engine()
        results = engine.suggest_full("")
        assert results == []

