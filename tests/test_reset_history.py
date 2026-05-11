"""
Tests for AutocompleteEngine.reset_history() and the PredictorLearnsFromHistory
protocol that governs how reset propagates to predictors.

These two concerns are tested together because the protocol is the mechanism
by which reset propagates - they cannot be tested in isolation without
duplicating the engine construction boilerplate.
"""
from __future__ import annotations

from aac.domain.history import History
from aac.domain.types import WeightedPredictor
from aac.engine.engine import AutocompleteEngine
from aac.predictors.frequency import FrequencyPredictor
from aac.predictors.history import HistoryPredictor
from aac.presets import create_engine
from aac.ranking.contracts import PredictorLearnsFromHistory
from aac.ranking.score import ScoreRanker

_VOCAB = {
    "hello": 100, "help": 80, "hero": 50, "her": 200,
    "here": 120, "heap": 40, "world": 300, "word": 150,
    "programming": 500, "program": 400, "progress": 300,
}


# ---------------------------------------------------------------------------
# reset_history() core behaviour
# ---------------------------------------------------------------------------

class TestResetHistory:
    def test_clears_in_memory_history(self) -> None:
        engine = create_engine("default", vocabulary=_VOCAB)
        engine.record_selection("he", "heap")
        assert len(engine.history) > 0

        engine.reset_history()
        assert len(engine.history) == 0

    def test_affects_subsequent_suggestions(self) -> None:
        """After reset, suggestions must reflect zero history."""
        engine = create_engine("default", vocabulary=_VOCAB)
        for _ in range(10):
            engine.record_selection("he", "heap")
        assert engine.suggest("he")[0] == "heap", "heap should lead before reset"

        engine.reset_history()
        after = engine.suggest("he")
        assert after[0] != "heap", (
            f"heap should no longer lead after reset, got: {after[:3]}"
        )

    def test_history_object_is_replaced(self) -> None:
        """reset_history() replaces the internal History instance, not just clears it."""
        engine = create_engine("default", vocabulary=_VOCAB)
        engine.record_selection("he", "hello")
        old_history = engine.history

        engine.reset_history()
        assert engine.history is not old_history
        assert len(engine.history) == 0

    def test_learning_rankers_see_new_history(self) -> None:
        """Learning rankers must read from the new History after reset."""
        engine = create_engine("production", vocabulary=_VOCAB)
        for _ in range(8):
            engine.record_selection("he", "heap")
        assert engine.suggest("he")[0] == "heap"

        engine.reset_history()
        after = engine.suggest("he")
        assert after[0] != "heap", (
            f"heap should not lead after reset, got: {after[:3]}"
        )

    def test_does_not_affect_persisted_store(self) -> None:
        """reset_history() clears in-memory state only; a JsonHistoryStore is unaffected."""
        import tempfile
        from pathlib import Path

        from aac.storage.json_store import JsonHistoryStore

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.json"
            store = JsonHistoryStore(path)
            history = store.load()

            engine = create_engine("default", vocabulary=_VOCAB, history=history)
            engine.record_selection("he", "hero")
            store.save(engine.history)

            engine.reset_history()
            assert len(engine.history) == 0

            # Store file must still have the old entry
            reloaded = store.load()
            assert len(reloaded) == 1

    def test_multiple_resets_are_idempotent(self) -> None:
        engine = create_engine("default", vocabulary=_VOCAB)
        engine.record_selection("he", "hello")
        engine.reset_history()
        engine.reset_history()  # must not raise
        assert len(engine.history) == 0

    def test_can_record_after_reset(self) -> None:
        """Engine must be fully functional after reset."""
        engine = create_engine("default", vocabulary=_VOCAB)
        engine.reset_history()
        engine.record_selection("he", "help")
        assert engine.history.counts_for_prefix("he").get("help", 0) == 1


# ---------------------------------------------------------------------------
# PredictorLearnsFromHistory protocol
# ---------------------------------------------------------------------------

class TestPredictorLearnsFromHistoryProtocol:
    def test_history_predictor_satisfies_protocol(self) -> None:
        history = History()
        predictor = HistoryPredictor(history)
        assert isinstance(predictor, PredictorLearnsFromHistory), (
            "HistoryPredictor must satisfy PredictorLearnsFromHistory"
        )

    def test_frequency_predictor_does_not_satisfy_protocol(self) -> None:
        predictor = FrequencyPredictor(_VOCAB)
        assert not isinstance(predictor, PredictorLearnsFromHistory), (
            "FrequencyPredictor must not satisfy PredictorLearnsFromHistory"
        )

    def test_stateless_predictor_is_not_updated_by_reset(self) -> None:
        """A predictor with no 'history' attribute must not be touched by reset_history().

        Verifies the most important protection: predictors that don't opt in
        to the protocol are left completely untouched.
        """
        class _StatelessPredictor:
            name = "stateless_fake"
            _marker = "original"

            def predict(self, ctx):  # type: ignore[override]
                return []

        history = History()
        fake = _StatelessPredictor()

        engine = AutocompleteEngine(
            predictors=[
                WeightedPredictor(FrequencyPredictor(_VOCAB), weight=1.0),
                WeightedPredictor(fake, weight=1.0),  # type: ignore[arg-type]
            ],
            ranker=ScoreRanker(),
            history=history,
        )

        engine.reset_history()

        assert not hasattr(fake, "history"), (
            "reset_history() must not inject 'history' into a predictor "
            "that does not implement PredictorLearnsFromHistory"
        )
        assert fake._marker == "original", "predictor state must be unchanged"

    def test_reset_propagates_to_history_predictor_via_protocol(self) -> None:
        """reset_history() must update HistoryPredictor through the protocol."""
        history = History()
        history.record("he", "hello")
        hist_predictor = HistoryPredictor(history)

        engine = AutocompleteEngine(
            predictors=[
                WeightedPredictor(FrequencyPredictor(_VOCAB), weight=1.0),
                WeightedPredictor(hist_predictor, weight=1.5),
            ],
            ranker=ScoreRanker(),
            history=history,
        )

        engine.reset_history()

        assert hist_predictor.history is engine.history, (
            "HistoryPredictor.history must point to the new History after reset"
        )
        assert len(hist_predictor.history) == 0, (
            "HistoryPredictor must see the new empty history after reset"
        )
