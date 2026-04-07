from __future__ import annotations

from aac.domain.history import History
from aac.predictors.history import HistoryPredictor


def test_history_predictor_learns() -> None:
    history = History()
    history.record("he", "hello")
    history.record("he", "hello")
    history.record("he", "help")

    predictor = HistoryPredictor(history)
    results = predictor.predict("he")

    scores = {r.suggestion.value: r.score for r in results}

    assert scores["hello"] == 2.0
    assert scores["help"] == 1.0


def test_history_predictor_returns_nothing_with_no_history() -> None:
    history = History()
    predictor = HistoryPredictor(history)
    assert predictor.predict("he") == []


def test_history_predictor_only_returns_matching_prefix() -> None:
    history = History()
    history.record("he", "hello")
    history.record("wo", "world")

    predictor = HistoryPredictor(history)
    results = predictor.predict("he")
    values = {r.suggestion.value for r in results}

    assert values == {"hello"}
    assert "world" not in values