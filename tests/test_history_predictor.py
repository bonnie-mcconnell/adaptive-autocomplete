from __future__ import annotations

from aac.domain.history import History
from aac.predictors.history import HistoryPredictor


def test_history_predictor_learns() -> None:
    """HistoryPredictor reads from the shared History object.
    Record via History directly - HistoryPredictor has no record hook
    by design, since the engine writes to the shared History and
    HistoryPredictor reads from it. A separate hook would double-count.
    """
    history = History()
    predictor = HistoryPredictor(history)

    history.record("he", "hello")
    history.record("he", "hello")
    history.record("he", "help")

    results = predictor.predict("he")

    scores = {r.suggestion.value: r.score for r in results}

    assert scores["hello"] == 2.0
    assert scores["help"] == 1.0


def test_history_predictor_returns_empty_with_no_history() -> None:
    history = History()
    predictor = HistoryPredictor(history)
    assert predictor.predict("he") == []


def test_history_predictor_ignores_unrelated_prefix() -> None:
    history = History()
    predictor = HistoryPredictor(history)

    history.record("wo", "world")

    results = predictor.predict("he")
    assert results == []
