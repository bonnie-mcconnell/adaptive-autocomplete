"""
Tests for HistoryPredictor.

HistoryPredictor reads from the shared History object - it has no
record() hook by design. The engine writes to History directly;
a separate hook on the predictor would double-count selections.
"""
from __future__ import annotations

import math

import pytest

from aac.domain.history import History
from aac.predictors.history import HistoryPredictor


def test_scores_reflect_selection_frequency() -> None:
    """Scores are log-normalised: more selections → higher score, in (0, 1]."""
    history = History()
    history.record("he", "hello")
    history.record("he", "hello")
    history.record("he", "help")
    predictor = HistoryPredictor(history)
    scores = {r.suggestion.value: r.score for r in predictor.predict("he")}

    # hello (count=2) must score above help (count=1)
    assert scores["hello"] > scores["help"]

    # All scores in (0, 1]
    for s in scores.values():
        assert 0.0 < s <= 1.0

    # Score formula: log(1+count) / log(1+max_count), max_count=2
    log_max = math.log1p(2)
    assert scores["hello"] == pytest.approx(math.log1p(2) / log_max)   # == 1.0
    assert scores["help"] == pytest.approx(math.log1p(1) / log_max)


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
    """Confidence equals the log-normalised score for HistoryPredictor."""
    history = History()
    history.record("he", "hello")
    history.record("he", "hello")
    history.record("he", "help")
    predictor = HistoryPredictor(history)
    results = {r.suggestion.value: r for r in predictor.predict("he")}

    assert results["hello"].explanation is not None
    assert results["help"].explanation is not None

    # Most-selected word scores 1.0
    assert results["hello"].explanation.confidence == pytest.approx(1.0)

    # help has count=1, max_count=2: log(2)/log(3)
    expected = math.log1p(1) / math.log1p(2)
    assert results["help"].explanation.confidence == pytest.approx(expected)
