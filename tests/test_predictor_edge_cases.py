"""
Edge-case and validation tests for all predictors and supporting types.

Covers: empty prefix contracts, constructor guards, the public levenshtein
API, WeightedPredictor.name, DecayFunction boundary conditions, engine
adapters, and preset output.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aac.domain.history import History
from aac.domain.types import (
    CompletionContext,
    ScoredSuggestion,
    Suggestion,
    WeightedPredictor,
)
from aac.engine.engine import AutocompleteEngine
from aac.predictors.edit_distance import EditDistancePredictor, levenshtein
from aac.predictors.frequency import FrequencyPredictor
from aac.predictors.history import HistoryPredictor
from aac.predictors.static_prefix import StaticPrefixPredictor
from aac.predictors.trie import TriePrefixPredictor
from aac.ranking.decay import DecayFunction, DecayRanker
from aac.ranking.learning import LearningRanker
from aac.storage.json_store import JsonHistoryStore

# ------------------------------------------------------------------
# Empty prefix - all predictors must return [] not raise
# ------------------------------------------------------------------

def test_frequency_predictor_empty_prefix() -> None:
    assert FrequencyPredictor({"hello": 100}).predict(CompletionContext("")) == []


def test_history_predictor_empty_prefix() -> None:
    h = History()
    h.record("he", "hello")
    assert HistoryPredictor(h).predict(CompletionContext("")) == []


def test_static_prefix_predictor_empty_prefix() -> None:
    assert StaticPrefixPredictor(["hello", "help"]).predict(CompletionContext("")) == []


def test_trie_predictor_empty_prefix() -> None:
    assert TriePrefixPredictor(["hello", "help"]).predict(CompletionContext("")) == []


def test_edit_distance_predictor_empty_prefix() -> None:
    assert EditDistancePredictor(["hello", "help"], max_distance=1).predict(
        CompletionContext("")
    ) == []


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
    with pytest.raises(ValueError, match="boost"):
        LearningRanker(History(), boost=-1.0)


def test_learning_ranker_rejects_negative_dominance_ratio() -> None:
    with pytest.raises(ValueError, match="dominance_ratio"):
        LearningRanker(History(), dominance_ratio=-0.1)


# ------------------------------------------------------------------
# Public levenshtein function
# ------------------------------------------------------------------

def test_levenshtein_exact_match() -> None:
    assert levenshtein("hello", "hello") == 0


def test_levenshtein_one_edit() -> None:
    assert levenshtein("hello", "helo") == 1


def test_levenshtein_empty_string() -> None:
    assert levenshtein("", "abc") == 3


# ------------------------------------------------------------------
# WeightedPredictor.name delegates to inner predictor
# ------------------------------------------------------------------

def test_weighted_predictor_name_delegates() -> None:
    p = FrequencyPredictor({"hello": 100})
    wp = WeightedPredictor(predictor=p, weight=1.0)
    assert wp.name == "frequency"


# ------------------------------------------------------------------
# DecayFunction boundary conditions
# ------------------------------------------------------------------

def test_decay_function_future_event_returns_one() -> None:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    future = now + timedelta(hours=1)
    assert DecayFunction(half_life_seconds=3600).weight(now=now, event_time=future) == 1.0


def test_decay_function_rejects_naive_event_time() -> None:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    naive = datetime(2024, 1, 1)
    with pytest.raises(ValueError):
        DecayFunction(half_life_seconds=3600).weight(now=now, event_time=naive)


def test_decay_ranker_skips_entries_for_other_prefixes() -> None:
    """_decayed_counts must not leak boosts across prefixes."""
    h = History()
    h.record("wo", "world")  # different prefix - must not affect "he" completions
    ranker = DecayRanker(
        history=h,
        decay=DecayFunction(half_life_seconds=3600),
        now=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    suggestions = [ScoredSuggestion(suggestion=Suggestion("world"), score=1.0)]
    result = ranker.rank("he", suggestions)
    assert result[0].score == 1.0


def test_decay_ranker_empty_suggestions_returns_empty() -> None:
    h = History()
    ranker = DecayRanker(
        history=h,
        decay=DecayFunction(half_life_seconds=3600),
        now=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    assert ranker.rank("he", []) == []


# ------------------------------------------------------------------
# Engine adapters
# ------------------------------------------------------------------

def test_engine_explain_as_dicts_schema() -> None:
    engine = AutocompleteEngine(predictors=[StaticPrefixPredictor(["hello", "help"])])
    for row in engine.explain_as_dicts("he"):
        assert {"value", "base_score", "history_boost", "final_score"} <= row.keys()


def test_engine_history_property_is_the_injected_instance() -> None:
    h = History()
    engine = AutocompleteEngine(
        predictors=[StaticPrefixPredictor(["hello"])],
        history=h,
    )
    assert engine.history is h


def test_engine_record_calls_predictor_hook_if_present() -> None:
    """A predictor with a record() hook must have it called on selection."""
    recorded: list[tuple[str, str]] = []

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
# Trie: limit enforcement inside _collect
# ------------------------------------------------------------------

def test_trie_respects_max_results_limit() -> None:
    p = TriePrefixPredictor(
        ["help", "hello", "helium", "herald", "heron", "hex"], max_results=2
    )
    assert len(p.predict(CompletionContext("he"))) == 2


# ------------------------------------------------------------------
# presets.describe_presets
# ------------------------------------------------------------------

def test_describe_presets_contains_all_preset_names() -> None:
    from aac.presets import available_presets, describe_presets
    output = describe_presets()
    for name in available_presets():
        assert name in output


# ------------------------------------------------------------------
# JsonHistoryStore defensive branches
# ------------------------------------------------------------------

def test_json_store_non_list_entries_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "bad_entries.json"
    path.write_text(
        json.dumps({"version": 2, "entries": "not-a-list"}), encoding="utf-8"
    )
    assert list(JsonHistoryStore(path).load().entries()) == []


def test_v2_non_string_timestamp_skipped(tmp_path: Path) -> None:
    path = tmp_path / "bad_ts.json"
    payload = {
        "version": 2,
        "entries": [
            {"prefix": "he", "value": "hello", "timestamp": 12345},
            {"prefix": "he", "value": "help", "timestamp": "2024-01-01T12:00:00+00:00"},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    counts = JsonHistoryStore(path).load().counts_for_prefix("he")
    assert "help" in counts
    assert "hello" not in counts


def test_v2_naive_timestamp_gets_utc(tmp_path: Path) -> None:
    path = tmp_path / "naive_ts.json"
    payload = {
        "version": 2,
        "entries": [{"prefix": "he", "value": "hello", "timestamp": "2024-01-01T12:00:00"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    entries = list(JsonHistoryStore(path).load().entries())
    assert len(entries) == 1
    assert entries[0].timestamp.tzinfo is not None