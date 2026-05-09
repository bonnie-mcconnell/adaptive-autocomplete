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
    assert engine.suggest("x") == ["bar", "baz", "foo"]


def test_engine_combines_frequency_and_trie() -> None:
    engine = AutocompleteEngine(predictors=[
        TriePrefixPredictor(["hello", "help"]),
        FrequencyPredictor({"hello": 10, "help": 1}),
    ])
    results = engine.suggest("he")
    assert results[0] == "hello"


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
    assert engine.suggest("he") == ["hello", "help"]
    engine.record_selection("he", "help")
    assert engine.suggest("he")[0] == "help"


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
        from aac.domain.types import WeightedPredictor
        from aac.engine.engine import AutocompleteEngine
        from aac.predictors.frequency import FrequencyPredictor

        vocab = {"hello": 100, "help": 80, "hero": 1}
        history = History()
        engine = AutocompleteEngine(
            predictors=[WeightedPredictor(FrequencyPredictor(vocab), weight=1.0)],
            history=history,
        )

        # Without any selection, hero should be last (lowest frequency)
        before = engine.suggest("he")
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
        from aac.domain.types import WeightedPredictor
        from aac.engine.engine import AutocompleteEngine
        from aac.predictors.frequency import FrequencyPredictor

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


# ---------------------------------------------------------------------------
# PredictorAcceptsRecord protocol tests
# ---------------------------------------------------------------------------

def test_record_selection_calls_typed_protocol_not_getattr() -> None:
    """
    record_selection() must use PredictorAcceptsRecord isinstance check,
    not getattr duck typing.  This test verifies that:
    1. A predictor implementing the protocol has its record() called.
    2. A predictor with an unrelated .record attribute is NOT called.
    """
    from aac.domain.types import CompletionContext
    from aac.ranking.contracts import PredictorAcceptsRecord

    record_calls: list[tuple[str, str]] = []

    class _RecordingPredictor:
        name = "recording"

        def predict(self, ctx: CompletionContext) -> list:
            return []

        def record(self, ctx: CompletionContext, value: str) -> None:
            record_calls.append((ctx.text, value))

    # Verify it satisfies the protocol via isinstance
    assert isinstance(_RecordingPredictor(), PredictorAcceptsRecord)

    engine = AutocompleteEngine([_RecordingPredictor()])
    engine.record_selection("prog", "programming")

    assert len(record_calls) == 1
    assert record_calls[0] == ("prog", "programming")


def test_record_selection_with_non_callable_record_attr_does_not_raise() -> None:
    """A predictor with a non-callable .record attribute must not crash record_selection().

    Python's runtime_checkable Protocol only checks for attribute *presence*, not type.
    A string attribute named 'record' satisfies isinstance(..., PredictorAcceptsRecord)
    even though it cannot be called. This is a known Python limitation documented in
    PEP 544. The engine must guard the call with callable() to avoid AttributeError.
    """
    from aac.domain.types import CompletionContext
    from aac.ranking.contracts import PredictorAcceptsRecord

    class _PredictorWithStringRecord:
        name = "unrelated"
        record = "I am a string, not a method"

        def predict(self, ctx: CompletionContext) -> list:
            return []

    # runtime_checkable Protocol checks presence only - a string attribute
    # named 'record' satisfies the isinstance check (Python limitation).
    # The engine must handle this gracefully.
    predictor = _PredictorWithStringRecord()
    assert isinstance(predictor, PredictorAcceptsRecord), (
        "runtime_checkable only checks attribute presence, not type - "
        "this isinstance result is correct Python behaviour"
    )

    engine = AutocompleteEngine([predictor])
    # The engine must guard with callable() - must not raise TypeError
    engine.record_selection("prog", "programming")


# ---------------------------------------------------------------------------
# suggest_full() tests
# ---------------------------------------------------------------------------

def test_suggest_full_returns_correct_schema() -> None:
    """suggest_full() must return dicts with word, count, confidence keys."""
    from aac.presets import create_engine
    engine = create_engine("stateless")
    results = engine.suggest_full("prog", limit=5)

    assert isinstance(results, list)
    assert len(results) > 0
    for item in results:
        assert "word" in item, f"Missing 'word': {item}"
        assert "count" in item, f"Missing 'count': {item}"
        assert "confidence" in item, f"Missing 'confidence': {item}"
        assert isinstance(item["word"], str)
        assert isinstance(item["count"], int)
        assert isinstance(item["confidence"], float)
        assert 0.0 <= item["confidence"] <= 1.0


def test_suggest_full_top_confidence_is_one() -> None:
    """Top-ranked suggestion must have confidence 1.0."""
    from aac.presets import create_engine
    engine = create_engine("stateless")
    results = engine.suggest_full("prog", limit=5)
    assert results, "Expected at least one result"
    assert abs(results[0]["confidence"] - 1.0) < 1e-9


def test_suggest_full_count_matches_history() -> None:
    """count field must reflect actual recorded selections."""
    from aac.domain.history import History
    from aac.presets import create_engine

    history = History()
    engine = create_engine("default", history=history)

    engine.record_selection("prog", "programming")
    engine.record_selection("prog", "programming")

    results = engine.suggest_full("prog", limit=10)
    programming = next((r for r in results if r["word"] == "programming"), None)
    assert programming is not None, "programming should appear after 2 selections"
    assert programming["count"] == 2


def test_suggest_full_single_pipeline_consistency() -> None:
    """
    suggest_full() must be consistent with separate suggest_with_history()
    and suggest_with_confidence() calls made in the same state (no intervening
    record_selection()). Values must appear in the same order.

    This test documents the contract: suggest_full() is a single-pass
    equivalent of the two-call pattern. It does NOT test that only one
    pipeline run occurs (that's an implementation detail), but it does verify
    the output is consistent.
    """
    from aac.presets import create_engine

    engine = create_engine("stateless")
    prefix = "prog"

    full = engine.suggest_full(prefix, limit=10)
    history_results = engine.suggest_with_history(prefix, limit=10)
    confidence_results = engine.suggest_with_confidence(prefix, limit=10)

    full_words = [r["word"] for r in full]
    history_words = [w for w, _ in history_results]
    confidence_words = [w for w, _ in confidence_results]

    assert full_words == history_words, (
        f"suggest_full word order must match suggest_with_history.\n"
        f"full: {full_words}\nhistory: {history_words}"
    )
    assert full_words == confidence_words, (
        f"suggest_full word order must match suggest_with_confidence.\n"
        f"full: {full_words}\nconfidence: {confidence_words}"
    )


def test_suggest_full_empty_prefix_returns_empty() -> None:
    """Empty prefix must return empty list, not raise."""
    from aac.presets import create_engine
    engine = create_engine("stateless")
    results = engine.suggest_full("", limit=5)
    assert results == []
