"""
Tests for predictor edge cases - empty prefix, validation errors,
and constructor guards that are exercised on unusual inputs.
"""
from __future__ import annotations

import pytest

from aac.domain.history import History
from aac.domain.types import CompletionContext
from aac.predictors.edit_distance import EditDistancePredictor, levenshtein
from aac.predictors.frequency import FrequencyPredictor
from aac.predictors.history import HistoryPredictor
from aac.predictors.static_prefix import StaticPrefixPredictor
from aac.predictors.trie import TriePrefixPredictor

# ------------------------------------------------------------------
# Empty prefix - all predictors must return [] not raise
# ------------------------------------------------------------------

def test_frequency_predictor_empty_prefix() -> None:
    p = FrequencyPredictor({"hello": 100})
    assert p.predict(CompletionContext("")) == []


def test_history_predictor_empty_prefix() -> None:
    h = History()
    h.record("he", "hello")
    p = HistoryPredictor(h)
    assert p.predict(CompletionContext("")) == []


def test_static_prefix_predictor_empty_prefix() -> None:
    p = StaticPrefixPredictor(["hello", "help"])
    assert p.predict(CompletionContext("")) == []


def test_trie_predictor_empty_prefix() -> None:
    p = TriePrefixPredictor(["hello", "help"])
    assert p.predict(CompletionContext("")) == []


def test_edit_distance_predictor_empty_prefix() -> None:
    p = EditDistancePredictor(["hello", "help"], max_distance=1)
    assert p.predict(CompletionContext("")) == []


# ------------------------------------------------------------------
# Constructor validation
# ------------------------------------------------------------------

def test_frequency_predictor_rejects_empty_vocabulary() -> None:
    with pytest.raises(ValueError):
        FrequencyPredictor({})


def test_frequency_predictor_rejects_zero_max_frequency() -> None:
    with pytest.raises(ValueError):
        FrequencyPredictor({"hello": 0})


def test_learning_ranker_rejects_negative_boost() -> None:
    from aac.ranking.learning import LearningRanker
    h = History()
    with pytest.raises(ValueError, match="boost"):
        LearningRanker(h, boost=-1.0)


def test_learning_ranker_rejects_negative_dominance_ratio() -> None:
    from aac.ranking.learning import LearningRanker
    h = History()
    with pytest.raises(ValueError, match="dominance_ratio"):
        LearningRanker(h, dominance_ratio=-0.1)


# ------------------------------------------------------------------
# levenshtein_distance public API
# ------------------------------------------------------------------

def test_levenshtein_distance_public_function() -> None:
    assert levenshtein("hello", "hello") == 0
    assert levenshtein("hello", "helo") == 1
    assert levenshtein("", "abc") == 3


# ------------------------------------------------------------------
# WeightedPredictor.name property
# ------------------------------------------------------------------

def test_weighted_predictor_name_delegates_to_predictor() -> None:
    from aac.domain.types import WeightedPredictor
    p = FrequencyPredictor({"hello": 100})
    wp = WeightedPredictor(predictor=p, weight=1.0)
    assert wp.name == "frequency"


# ------------------------------------------------------------------
# DecayFunction: future timestamp and naive timestamp validation
# ------------------------------------------------------------------

def test_decay_function_future_timestamp_returns_one() -> None:
    from datetime import datetime, timedelta, timezone

    from aac.ranking.decay import DecayFunction
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    future = now + timedelta(hours=1)
    decay = DecayFunction(half_life_seconds=3600)
    assert decay.weight(now=now, event_time=future) == 1.0


def test_decay_function_rejects_naive_timestamp() -> None:
    from datetime import datetime, timezone

    from aac.ranking.decay import DecayFunction
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    naive = datetime(2024, 1, 1)  # no tzinfo
    decay = DecayFunction(half_life_seconds=3600)
    with pytest.raises(ValueError):
        decay.weight(now=now, event_time=naive)


# ------------------------------------------------------------------
# engine: explain_as_dicts adapter, history property, record hook
# ------------------------------------------------------------------

def test_engine_explain_as_dicts_shape() -> None:
    from aac.engine.engine import AutocompleteEngine
    engine = AutocompleteEngine(predictors=[StaticPrefixPredictor(["hello", "help"])])
    result = engine.explain_as_dicts("he")
    assert isinstance(result, list)
    for row in result:
        assert "value" in row
        assert "base_score" in row
        assert "history_boost" in row
        assert "final_score" in row


def test_engine_history_property() -> None:
    from aac.domain.history import History
    from aac.engine.engine import AutocompleteEngine
    h = History()
    engine = AutocompleteEngine(
        predictors=[StaticPrefixPredictor(["hello"])],
        history=h,
    )
    assert engine.history is h


def test_engine_record_calls_predictor_hook_if_present() -> None:
    """A predictor with a record() hook should have it called on selection."""
    from aac.domain.types import ScoredSuggestion, Suggestion
    from aac.engine.engine import AutocompleteEngine

    recorded = []

    class HookPredictor:
        name = "hook"

        def predict(self, ctx: CompletionContext) -> list[ScoredSuggestion]:
            return [ScoredSuggestion(suggestion=Suggestion("hello"), score=1.0)]

        def record(self, ctx: CompletionContext, value: str) -> None:
            recorded.append((ctx.text, value))

    engine = AutocompleteEngine(predictors=[HookPredictor()])
    engine.record_selection("he", "hello")
    assert recorded == [("he", "hello")]


# ------------------------------------------------------------------
# presets.describe_presets
# ------------------------------------------------------------------

def test_describe_presets_contains_all_preset_names() -> None:
    from aac.presets import available_presets, describe_presets
    output = describe_presets()
    for name in available_presets():
        assert name in output


# ------------------------------------------------------------------
# Trie: limit enforcement inside _collect
# ------------------------------------------------------------------

def test_trie_respects_max_results_limit() -> None:
    """_collect must stop at max_results even when more matches exist."""
    p = TriePrefixPredictor(["help", "hello", "helium", "herald", "heron"], max_results=2)
    results = p.predict(CompletionContext("he"))
    assert len(results) <= 2


# ------------------------------------------------------------------
# DecayRanker: empty suggestions list
# ------------------------------------------------------------------

def test_decay_ranker_empty_suggestions_returns_empty() -> None:
    from datetime import datetime, timezone

    from aac.ranking.decay import DecayFunction, DecayRanker
    h = History()
    ranker = DecayRanker(
        history=h,
        decay=DecayFunction(half_life_seconds=3600),
        now=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    assert ranker.rank("he", []) == []


# ------------------------------------------------------------------
# LearningRanker: empty suggestions list
# ------------------------------------------------------------------

def test_learning_ranker_empty_suggestions_returns_empty() -> None:
    from aac.ranking.learning import LearningRanker
    h = History()
    ranker = LearningRanker(h)
    assert ranker.rank("he", []) == []


# ------------------------------------------------------------------
# JsonHistoryStore: non-list entries key in v2
# ------------------------------------------------------------------

def test_json_store_non_list_entries_returns_empty() -> None:
    import json
    import os
    import tempfile
    from pathlib import Path

    from aac.storage.json_store import JsonHistoryStore

    fd, name = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    path = Path(name)
    try:
        path.write_text(
            json.dumps({"version": 2, "entries": "not-a-list"}),
            encoding="utf-8",
        )
        loaded = JsonHistoryStore(path).load()
        assert list(loaded.entries()) == []
    finally:
        path.unlink()
