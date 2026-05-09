"""
Tests for EngineConfig serialisation, registry, and round-trip reconstruction.

Covers:
  - Preset engine JSON round-trip produces identical suggestions
  - Custom engine round-trip via PredictorRegistry
  - EngineConfig.diff() identifies changes
  - Schema version mismatch raises ValueError
  - Unknown predictor name raises KeyError with helpful message
  - Metadata round-trip preserved
  - Third-party predictor registration and reconstruction
"""
from __future__ import annotations

import json

import pytest

from aac.domain.history import History
from aac.domain.types import WeightedPredictor
from aac.engine.config import EngineConfig, PredictorConfig, PredictorRegistry, RankerConfig
from aac.engine.engine import AutocompleteEngine
from aac.predictors.frequency import FrequencyPredictor
from aac.presets import available_presets, create_engine

_TEST_VOCAB = {
    "hello": 100,
    "help": 90,
    "hero": 50,
    "world": 200,
    "word": 180,
    "work": 150,
    "worth": 80,
}


class TestPresetRoundTrip:
    @pytest.mark.parametrize("preset", available_presets())
    def test_preset_round_trip_matches_suggestions(self, preset: str) -> None:
        """Preset engine -> to_config -> build() -> suggest() must agree."""
        engine = create_engine(preset)
        config = engine.to_config(preset=preset)
        engine2 = config.build()

        for prefix in ("he", "prog", "programing", "wor"):
            assert engine.suggest(prefix) == engine2.suggest(prefix), (
                f"Preset {preset!r} round-trip mismatch for prefix {prefix!r}"
            )

    @pytest.mark.parametrize("preset", available_presets())
    def test_preset_json_round_trip(self, preset: str) -> None:
        """Config serialises to valid JSON and deserialises correctly."""
        engine = create_engine(preset)
        config = engine.to_config(preset=preset)

        json_str = config.to_json()
        data = json.loads(json_str)

        assert data["preset"] == preset
        assert data["version"] == 1
        assert isinstance(data["predictors"], list)
        assert isinstance(data["rankers"], list)

        config2 = EngineConfig.from_json(json_str)
        assert config2.preset == config.preset
        assert [p.name for p in config2.predictors] == [p.name for p in config.predictors]


class TestCustomEngineRoundTrip:
    def test_custom_frequency_engine_round_trip(self) -> None:
        """A single-predictor custom engine reconstructs correctly."""
        from aac.ranking.score import ScoreRanker

        engine = AutocompleteEngine(
            predictors=[WeightedPredictor(FrequencyPredictor(_TEST_VOCAB), weight=1.0)],
            ranker=ScoreRanker(),
        )

        config = engine.to_config()
        assert config.preset is None

        engine2 = config.build(vocabulary=_TEST_VOCAB)
        for prefix in ("he", "wo"):
            assert engine.suggest(prefix) == engine2.suggest(prefix), (
                f"Custom engine round-trip mismatch for prefix {prefix!r}"
            )

    def test_custom_engine_with_history_round_trip(self) -> None:
        """Custom engine with history-predictor round-trips with history preserved."""
        history = History()
        history.record("he", "hello")
        history.record("he", "hello")

        engine = create_engine("default", history=history)
        config = engine.to_config()

        engine2 = config.build(history=history)
        assert engine.suggest("he") == engine2.suggest("he")


class TestPredictorRegistry:
    def test_all_builtins_registered(self) -> None:
        """All built-in predictor names must be registered at import time."""
        expected = {"frequency", "history", "symspell", "trigram", "bktree", "trie", "static_prefix"}
        registered = set(PredictorRegistry.registered_names())
        missing = expected - registered
        assert not missing, f"Built-in predictors not registered: {missing}"

    def test_register_custom_predictor(self) -> None:
        """A third-party predictor can be registered and reconstructed via build()."""
        class PassthroughPredictor:
            """Returns the query itself as the only suggestion."""
            name = "passthrough_test"
            def predict(self, ctx):  # type: ignore[override]
                from aac.domain.types import ScoredSuggestion, Suggestion
                return [ScoredSuggestion(Suggestion(ctx.text), score=1.0)]

        PredictorRegistry.register(
            "passthrough_test",
            lambda vocab, params: PassthroughPredictor(),
        )

        config = EngineConfig(
            preset=None,
            predictors=[PredictorConfig("passthrough_test", weight=1.0)],
            rankers=[RankerConfig("score")],
        )

        engine = config.build()
        result = engine.suggest("hello")
        assert result == ["hello"], f"Expected ['hello'], got {result}"

        # Clean up to not pollute other tests
        del PredictorRegistry._registry["passthrough_test"]

    def test_unknown_predictor_raises_key_error_with_helpful_message(self) -> None:
        """build() with an unregistered predictor name gives a clear error."""
        config = EngineConfig(
            preset=None,
            predictors=[PredictorConfig("nonexistent_predictor", weight=1.0)],
            rankers=[],
        )

        with pytest.raises(KeyError, match="nonexistent_predictor"):
            config.build()


class TestEngineConfigDiff:
    def test_identical_configs_produce_empty_diff(self) -> None:
        a = create_engine("stateless").to_config(preset="stateless")
        b = create_engine("stateless").to_config(preset="stateless")
        assert a.diff(b) == []

    def test_different_preset_shows_in_diff(self) -> None:
        a = create_engine("stateless").to_config(preset="stateless")
        b = create_engine("default").to_config(preset="default")
        diffs = a.diff(b)
        assert any("preset" in d for d in diffs), f"Expected preset diff, got: {diffs}"

    def test_weight_change_shows_in_diff(self) -> None:
        a = EngineConfig(
            preset=None,
            predictors=[PredictorConfig("frequency", weight=1.0)],
            rankers=[],
        )
        b = EngineConfig(
            preset=None,
            predictors=[PredictorConfig("frequency", weight=2.0)],
            rankers=[],
        )
        diffs = a.diff(b)
        assert any("frequency" in d and "weight" in d for d in diffs), (
            f"Expected weight diff, got: {diffs}"
        )

    def test_added_predictor_shows_in_diff(self) -> None:
        a = EngineConfig(
            preset=None,
            predictors=[PredictorConfig("frequency", weight=1.0)],
            rankers=[],
        )
        b = EngineConfig(
            preset=None,
            predictors=[
                PredictorConfig("frequency", weight=1.0),
                PredictorConfig("symspell", weight=0.5),
            ],
            rankers=[],
        )
        diffs = a.diff(b)
        assert any("added" in d and "symspell" in d for d in diffs), (
            f"Expected added-predictor diff, got: {diffs}"
        )


class TestEngineConfigSchemaVersioning:
    def test_wrong_schema_version_raises_value_error(self) -> None:
        data = {
            "version": 999,
            "preset": "stateless",
            "predictors": [],
            "rankers": [],
        }
        with pytest.raises(ValueError, match="999"):
            EngineConfig.from_dict(data)

    def test_missing_version_defaults_to_1(self) -> None:
        data = {
            "preset": "stateless",
            "predictors": [],
            "rankers": [],
        }
        config = EngineConfig.from_dict(data)
        assert config.version == 1

    def test_metadata_preserved_through_json_round_trip(self) -> None:
        config = EngineConfig(
            preset="stateless",
            predictors=[],
            rankers=[],
            metadata={"vocabulary_path": "/data/vocab.json", "env": "prod"},
        )
        config2 = EngineConfig.from_json(config.to_json())
        assert config2.metadata == {"vocabulary_path": "/data/vocab.json", "env": "prod"}
