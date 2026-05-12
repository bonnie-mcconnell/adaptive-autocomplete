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



class TestEngineConfigBuildCustomRankers:
    """
    Tests for build() with custom (non-preset) engines using decay and
    learning rankers.

    These paths are the most important uncovered section: lines 444-465 in
    config.py are the ranker reconstruction branches in build(). The production
    preset uses DecayRanker; the default preset uses LearningRanker-equivalent
    via HistoryPredictor. Ensuring round-trip correctness for both ranker types
    guarantees that config.to_json() → EngineConfig.from_json().build() works
    for any engine a user might construct.
    """

    def test_custom_engine_with_decay_ranker_round_trips(self) -> None:
        """
        A custom engine using DecayRanker must reconstruct correctly.

        Verifies the 'decay' branch in build()'s ranker loop (lines 444-453).
        """
        from aac.domain.history import History
        from aac.domain.types import WeightedPredictor
        from aac.engine.engine import AutocompleteEngine
        from aac.predictors.frequency import FrequencyPredictor
        from aac.ranking.decay import DecayFunction, DecayRanker
        from aac.ranking.score import ScoreRanker

        history = History()
        engine = AutocompleteEngine(
            predictors=[WeightedPredictor(FrequencyPredictor(_TEST_VOCAB), weight=1.0)],
            ranker=[
                ScoreRanker(),
                DecayRanker(
                    history=history,
                    decay=DecayFunction(half_life_seconds=7200.0),
                    weight=2.0,
                ),
            ],
            history=history,
        )

        config = engine.to_config()
        assert config.preset is None

        # The decay ranker config must be captured
        ranker_names = [r.name for r in config.rankers]
        assert "decay" in ranker_names

        decay_cfg = next(r for r in config.rankers if r.name == "decay")
        assert decay_cfg.params["half_life_seconds"] == pytest.approx(7200.0)
        assert decay_cfg.params["weight"] == pytest.approx(2.0)

        # Round-trip: rebuild and verify suggestions match
        engine2 = config.build(vocabulary=_TEST_VOCAB)
        assert engine.suggest("he") == engine2.suggest("he")

    def test_custom_engine_with_learning_ranker_round_trips(self) -> None:
        """
        A custom engine using LearningRanker must reconstruct correctly.

        Verifies the 'learning' branch in build()'s ranker loop (lines 454-461).
        """
        from aac.domain.history import History
        from aac.domain.types import WeightedPredictor
        from aac.engine.engine import AutocompleteEngine
        from aac.predictors.frequency import FrequencyPredictor
        from aac.ranking.learning import LearningRanker
        from aac.ranking.score import ScoreRanker

        history = History()
        engine = AutocompleteEngine(
            predictors=[WeightedPredictor(FrequencyPredictor(_TEST_VOCAB), weight=1.0)],
            ranker=[
                ScoreRanker(),
                LearningRanker(history=history, boost=2.0, dominance_ratio=1.5),
            ],
            history=history,
        )

        config = engine.to_config()
        ranker_names = [r.name for r in config.rankers]
        assert "learning" in ranker_names

        learning_cfg = next(r for r in config.rankers if r.name == "learning")
        assert learning_cfg.params["boost"] == pytest.approx(2.0)
        assert learning_cfg.params["dominance_ratio"] == pytest.approx(1.5)

        engine2 = config.build(vocabulary=_TEST_VOCAB)
        assert engine.suggest("he") == engine2.suggest("he")

    def test_unknown_ranker_name_raises_value_error(self) -> None:
        """
        build() with an unrecognised ranker name must raise ValueError.

        The error message must name the offending ranker to help the user.
        """
        config = EngineConfig(
            preset=None,
            predictors=[PredictorConfig("frequency", weight=1.0)],
            rankers=[RankerConfig("not_a_real_ranker")],
        )
        with pytest.raises(ValueError, match="not_a_real_ranker"):
            config.build(vocabulary=_TEST_VOCAB)

    def test_build_emits_warning_when_vocabulary_path_in_metadata_but_no_vocab(self) -> None:
        """
        When config.metadata contains 'vocabulary_path' but build() is called
        without a vocabulary argument, a UserWarning must be emitted.

        This prevents silent correctness bugs: the user stored the vocab path
        in config so they could reload it, but forgot to pass it. The warning
        tells them exactly what to do.
        """
        import warnings as _warnings
        config = EngineConfig(
            preset=None,
            predictors=[PredictorConfig("frequency", weight=1.0)],
            rankers=[RankerConfig("score")],
            metadata={"vocabulary_path": "/path/to/vocab.json"},
        )
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            config.build()  # no vocabulary= argument

        assert any(
            issubclass(w.category, UserWarning) and "vocabulary_path" in str(w.message)
            for w in caught
        ), f"Expected UserWarning mentioning vocabulary_path, got: {caught}"


class TestEngineConfigDiffExtended:
    """
    Additional diff() tests covering branches not exercised by the existing suite.

    The existing tests cover: identical configs, preset change, weight change,
    added predictor. Missing: removed predictor, ranker params change.
    """

    def test_removed_predictor_shows_in_diff(self) -> None:
        """
        When config b has fewer predictors than a, diff must report the removed
        predictor by name and weight.
        """
        a = EngineConfig(
            preset=None,
            predictors=[
                PredictorConfig("frequency", weight=1.0),
                PredictorConfig("history", weight=1.5),
            ],
            rankers=[],
        )
        b = EngineConfig(
            preset=None,
            predictors=[PredictorConfig("frequency", weight=1.0)],
            rankers=[],
        )
        diffs = a.diff(b)
        assert any("removed" in d and "history" in d for d in diffs), (
            f"Expected removed-predictor diff mentioning 'history', got: {diffs}"
        )

    def test_ranker_set_change_shows_in_diff(self) -> None:
        """
        When two configs use completely different rankers, diff() must report
        the full ranker set change.

        This triggers line 521 in config.py: the ``rankers: [...] → [...]``
        diff entry, which only fires when the ranker NAMES differ between
        configs (not just params).
        """
        a = EngineConfig(
            preset=None,
            predictors=[PredictorConfig("frequency", weight=1.0)],
            rankers=[RankerConfig("score")],
        )
        b = EngineConfig(
            preset=None,
            predictors=[PredictorConfig("frequency", weight=1.0)],
            rankers=[RankerConfig("score"), RankerConfig("decay")],
        )
        diffs = a.diff(b)
        assert any("rankers" in d for d in diffs), (
            f"Expected ranker-set change in diff, got: {diffs}"
        )

    def test_ranker_params_change_shows_in_diff(self) -> None:
        """
        When two configs have the same ranker set but different params,
        diff must report the param change.

        This tests the `a_r.params != b_r.params` branch in diff().
        """
        a = EngineConfig(
            preset=None,
            predictors=[PredictorConfig("frequency", weight=1.0)],
            rankers=[RankerConfig("decay", params={"half_life_seconds": 3600.0, "weight": 1.0})],
        )
        b = EngineConfig(
            preset=None,
            predictors=[PredictorConfig("frequency", weight=1.0)],
            rankers=[RankerConfig("decay", params={"half_life_seconds": 7200.0, "weight": 2.0})],
        )
        diffs = a.diff(b)
        assert any("decay" in d and "params" in d for d in diffs), (
            f"Expected ranker params diff, got: {diffs}"
        )


class TestEngineConfigRepr:
    def test_repr_contains_preset_and_predictor_info(self) -> None:
        """
        __repr__ must include the preset name and predictor weights so it's
        informative in debuggers and error messages.
        """
        config = EngineConfig(
            preset="stateless",
            predictors=[PredictorConfig("frequency", weight=1.0)],
            rankers=[RankerConfig("score")],
        )
        r = repr(config)
        assert "stateless" in r
        assert "frequency" in r
        assert "score" in r

    def test_repr_none_preset(self) -> None:
        config = EngineConfig(
            preset=None,
            predictors=[PredictorConfig("frequency", weight=2.5)],
            rankers=[],
        )
        r = repr(config)
        assert "None" in r
        assert "2.5" in r


class TestPredictorRegistryBuiltins:
    """
    Verify that built-in registry factories actually produce working predictors.

    The registry factories for symspell, trigram, bktree, trie, static_prefix
    use a small vocabulary to avoid the 5s SymSpell build time. Lines 188-228
    in config.py (the factory bodies) are only covered when build_predictor()
    is called with those names, which the existing round-trip tests don't do
    for non-preset configs.
    """

    _SMALL_VOCAB = {"hello": 100, "help": 80, "hero": 50, "world": 200}

    def test_history_predictor_factory(self) -> None:
        pred = PredictorRegistry.build_predictor("history", self._SMALL_VOCAB, {})
        assert pred.name == "history"

    def test_history_predictor_factory_with_history_param(self) -> None:
        from aac.domain.history import History
        h = History()
        h.record("he", "hello")
        pred = PredictorRegistry.build_predictor("history", self._SMALL_VOCAB, {}, history=h)
        # Predictor should surface previously recorded selections
        results = {s.suggestion.value for s in pred.predict("he")}
        assert "hello" in results

    def test_trie_predictor_factory(self) -> None:
        pred = PredictorRegistry.build_predictor("trie", self._SMALL_VOCAB, {})
        results = {s.suggestion.value for s in pred.predict("he")}
        assert "hello" in results or "help" in results

    def test_static_prefix_factory(self) -> None:
        pred = PredictorRegistry.build_predictor("static_prefix", self._SMALL_VOCAB, {})
        results = {s.suggestion.value for s in pred.predict("he")}
        assert "hello" in results or "help" in results  # at least one he-prefix word

    def test_bktree_factory(self) -> None:
        pred = PredictorRegistry.build_predictor("bktree", self._SMALL_VOCAB, {})
        # EditDistancePredictor uses name="edit_distance" - the registry key "bktree"
        # is the alias used in config serialisation. Verify the predictor works.
        results = {s.suggestion.value for s in pred.predict("helo")}  # typo
        assert "hello" in results or len(results) > 0  # finds typo corrections

    def test_symspell_factory(self) -> None:
        """
        SymSpellPredictor factory (registry key 'symspell') must build and
        return a working predictor with the supplied small vocabulary.

        This calls the _symspell closure defined in _register_builtins() -
        lines 188-190 in config.py. Using a 4-word vocabulary keeps the
        delete-neighbourhood index build time under 1ms.
        """
        pred = PredictorRegistry.build_predictor("symspell", self._SMALL_VOCAB, {})
        assert pred.name == "symspell"
        # Typo recovery: "helo" → "hello"
        results = {s.suggestion.value for s in pred.predict("helo")}
        assert "hello" in results, f"symspell must recover 'helo' → 'hello', got: {results}"

    def test_trigram_factory(self) -> None:
        """
        TrigramPredictor factory (registry key 'trigram') must build and
        return a working predictor.

        Covers lines 197-199 in config.py (_trigram closure body).
        """
        pred = PredictorRegistry.build_predictor("trigram", self._SMALL_VOCAB, {})
        assert pred.name == "trigram"

    def test_adaptive_symspell_factory(self) -> None:
        """
        AdaptiveSymSpellPredictor factory (registry key 'adaptive_symspell')
        must build and return a working predictor.

        Covers lines 225-228 in config.py (_adaptive_symspell closure body).
        AdaptiveSymSpellPredictor.name is "symspell" (inherited from parent).
        """
        from aac.predictors.adaptive_symspell import AdaptiveSymSpellPredictor
        pred = PredictorRegistry.build_predictor("adaptive_symspell", self._SMALL_VOCAB, {})
        assert isinstance(pred, AdaptiveSymSpellPredictor)

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
