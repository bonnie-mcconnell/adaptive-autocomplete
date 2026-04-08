"""
Tests for HistoryPredictor.

HistoryPredictor reads from the shared History object - it has no
record() hook by design. The engine writes to History directly;
a separate hook on the predictor would double-count selections.
"""
from __future__ import annotations

from aac.domain.history import History
from aac.predictors.history import HistoryPredictor


def test_scores_reflect_selection_frequency() -> None:
    history = History()
    history.record("he", "hello")
    history.record("he", "hello")
    history.record("he", "help")
    predictor = HistoryPredictor(history)
    scores = {r.suggestion.value: r.score for r in predictor.predict("he")}
    assert scores["hello"] == 2.0
    assert scores["help"] == 1.0


def test_returns_empty_with_no_history() -> None:
    assert HistoryPredictor(History()).predict("he") == []


def test_ignores_unrelated_prefix() -> None:
    history = History()
    history.record("he", "hello")
    history.record("wo", "world")
    predictor = HistoryPredictor(history)
    values = {r.suggestion.value for r in predictor.predict("he")}
    assert values == {"hello"}
    assert "world" not in values


def test_empty_prefix_returns_empty() -> None:
    history = History()
    history.record("he", "hello")
    assert HistoryPredictor(history).predict("") == []


def test_confidence_relative_to_max_count() -> None:
    history = History()
    history.record("he", "hello")
    history.record("he", "hello")
    history.record("he", "help")
    predictor = HistoryPredictor(history)
    results = {r.suggestion.value: r for r in predictor.predict("he")}
    assert results["hello"].explanation is not None
    assert results["hello"].explanation.confidence == 1.0
    assert results["help"].explanation is not None
    assert results["help"].explanation.confidence == 0.5