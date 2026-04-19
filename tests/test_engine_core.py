"""
Tests for AutocompleteEngine core behaviour.

Covers: aggregation across predictors, weighted scoring,
learning via record_selection, explain immutability,
and mutation safety.
"""
from __future__ import annotations

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

# ---------------------------------------------------------------------------
# record_selection() key correctness
# ---------------------------------------------------------------------------

class TestRecordSelectionKeyCorrectness:
    """
    record_selection() must record under ctx.prefix(), not ctx.text.

    If the key is wrong, counts_for_prefix() never returns the recorded
    selection and the learning system silently does nothing.
    """

    def test_learning_visible_after_record_selection(self) -> None:
        from aac.domain.history import History
        from aac.engine.engine import AutocompleteEngine
        from aac.predictors.frequency import FrequencyPredictor
        from aac.domain.types import WeightedPredictor

        vocab = {"hello": 100, "help": 80, "hero": 1}
        history = History()
        engine = AutocompleteEngine(
            predictors=[WeightedPredictor(FrequencyPredictor(vocab), weight=1.0)],
            history=history,
        )

        # Without any selection, hero should be last (lowest frequency)
        before = [s.value for s in engine.suggest("he")]
        assert before.index("hero") > before.index("hello")

        # Record hero many times
        for _ in range(20):
            engine.record_selection("he", "hero")

        # hero should now be in history under the prefix "he"
        counts = history.counts_for_prefix("he")
        assert counts.get("hero", 0) == 20, (
            f"Expected 20 recordings under prefix 'he', got: {counts}"
        )

    def test_record_selection_normalises_case(self) -> None:
        """record_selection('He', value) must record under 'he' not 'He'."""
        from aac.domain.history import History
        from aac.engine.engine import AutocompleteEngine
        from aac.predictors.frequency import FrequencyPredictor
        from aac.domain.types import WeightedPredictor

        vocab = {"hello": 100, "hero": 1}
        history = History()
        engine = AutocompleteEngine(
            predictors=[WeightedPredictor(FrequencyPredictor(vocab), weight=1.0)],
            history=history,
        )

        engine.record_selection("He", "hero")

        # Should be stored under "he" (normalised prefix), not "He"
        assert history.counts_for_prefix("he").get("hero", 0) == 1
        assert history.counts_for_prefix("He").get("hero", 0) == 0
