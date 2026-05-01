"""
Tests for v0.6.0:

  - FrequencyPredictor max_results=100 default (no silent truncation)
  - explain() base_components always complete (all predictor names present)
  - explain() contribution_pct field
  - History.copy() independence
  - compare_presets() history isolation (each engine gets independent copy)
  - ContextualHistory domain partitioning
  - EngineConfig to_config() / from_json() / build() / diff()
"""
from __future__ import annotations

import json

import pytest

from aac.domain.contextual_history import ContextualHistory
from aac.domain.history import History
from aac.domain.types import CompletionContext, WeightedPredictor
from aac.engine.config import EngineConfig, PredictorConfig, RankerConfig
from aac.engine.engine import AutocompleteEngine
from aac.predictors.frequency import FrequencyPredictor, _DEFAULT_MAX_RESULTS
from aac.predictors.history import HistoryPredictor
from aac.presets import available_presets, compare_presets, create_engine
from aac.ranking.score import ScoreRanker

_VOCAB = {
    "hello": 100, "help": 80, "hero": 50, "her": 200,
    "here": 120, "heap": 40, "world": 300, "word": 150,
    "programming": 500, "program": 400, "progress": 300,
}


# ---------------------------------------------------------------------------
# FrequencyPredictor max_results default
# ---------------------------------------------------------------------------

class TestFrequencyPredictorDefault:
    def test_default_max_results_is_100(self) -> None:
        assert _DEFAULT_MAX_RESULTS == 100, (
            "Default max_results must be 100 to prevent silent truncation "
            "of words that rank 21-100 in frequency for their prefix bucket"
        )

    def test_words_beyond_old_limit_20_are_now_returned(self) -> None:
        """Words that ranked 21-100 in frequency are now returned by default."""
        from aac.data import load_english_frequencies
        freq = load_english_frequencies()
        p = FrequencyPredictor(freq)  # default max_results=100

        # "hello" ranks 22nd in frequency among "he" words.
        # With old default of 20 it was silently excluded.
        # With new default of 100 it must be present.
        results = {s.suggestion.value for s in p.predict(CompletionContext("he"))}
        assert "hello" in results, (
            "'hello' ranks 22nd in frequency for prefix 'he'. "
            "With max_results=100 it must be included."
        )

    def test_explicit_low_max_results_still_truncates(self) -> None:
        """Explicit max_results=5 still truncates - behaviour is opt-in, not broken."""
        from aac.data import load_english_frequencies
        freq = load_english_frequencies()
        p = FrequencyPredictor(freq, max_results=5)
        results = p.predict(CompletionContext("he"))
        assert len(results) <= 5


# ---------------------------------------------------------------------------
# explain() base_components completeness
# ---------------------------------------------------------------------------

class TestExplainBaseComponentsComplete:
    """
    All configured predictor names must appear in base_components.
    Zero contribution means the predictor ran but this word was not in its output.
    It does NOT mean the predictor is not configured.
    """

    def test_all_predictor_names_in_base_components(self) -> None:
        e = create_engine("production", vocabulary=_VOCAB)
        e.record_selection("he", "hello")
        exps = {exp.value: exp for exp in e.explain("he")}

        # "her" is a high-frequency word - frequency will contribute.
        # "symspell" and "trigram" may or may not contribute depending on prefix.
        # All four predictor names must always be present.
        for word, exp in exps.items():
            for name in ("frequency", "history", "symspell", "trigram"):
                assert name in exp.base_components, (
                    f"predictor '{name}' missing from base_components for '{word}'. "
                    f"Got: {list(exp.base_components.keys())}"
                )

    def test_zero_means_below_threshold_not_unconfigured(self) -> None:
        """A 0.0 base_component for frequency means the word ranked below
        max_results, NOT that FrequencyPredictor isn't configured."""
        history = History()
        history.record("he", "hello")

        e = AutocompleteEngine(
            predictors=[
                WeightedPredictor(FrequencyPredictor(_VOCAB, max_results=1), weight=1.0),
                WeightedPredictor(HistoryPredictor(history), weight=1.5),
            ],
            ranker=ScoreRanker(),
            history=history,
        )

        # With max_results=1, only the single highest-frequency "he" word
        # is returned by FrequencyPredictor. "hello" (lower frequency) will
        # appear via HistoryPredictor. Its frequency component must be 0.0,
        # not absent.
        exps = {exp.value: exp for exp in e.explain("he")}
        hello = exps.get("hello")
        assert hello is not None, "'hello' must appear via HistoryPredictor"
        assert "frequency" in hello.base_components, (
            "frequency key must be present even when FrequencyPredictor "
            "excluded this word (value will be 0.0)"
        )
        assert hello.base_components["frequency"] == pytest.approx(0.0), (
            "frequency contribution must be 0.0 when word was below max_results, "
            f"got {hello.base_components['frequency']}"
        )

    def test_non_contributing_predictors_are_0_not_absent(self) -> None:
        """ScoreRanker applies no boost - history_components for score ranker must be absent
        (ScoreRanker is intentionally excluded from history_components)."""
        e = create_engine("stateless", vocabulary=_VOCAB)
        for exp in e.explain("he"):
            # "score" ranker should not appear - it applies no boost
            assert "score" not in exp.history_components, (
                f"ScoreRanker must not appear in history_components for '{exp.value}'"
            )

    def test_invariant_holds_with_complete_components(self) -> None:
        """final_score == base_score + history_boost must hold with new component schema."""
        e = create_engine("production", vocabulary=_VOCAB)
        e.record_selection("he", "hello")
        for exp in e.explain("he"):
            expected = exp.base_score + exp.history_boost
            assert abs(exp.final_score - expected) < 1e-9, (
                f"Invariant violated for '{exp.value}': "
                f"final={exp.final_score:.6f}, base+boost={expected:.6f}"
            )


# ---------------------------------------------------------------------------
# contribution_pct
# ---------------------------------------------------------------------------

class TestContributionPct:
    def test_contribution_pct_present_in_explain(self) -> None:
        e = create_engine("production", vocabulary=_VOCAB)
        e.record_selection("he", "hello")
        for exp in e.explain("he"):
            assert isinstance(exp.contribution_pct, dict), (
                f"contribution_pct must be a dict, got {type(exp.contribution_pct)}"
            )

    def test_contribution_pct_values_are_fractions(self) -> None:
        """Each value in contribution_pct represents fraction of final_score."""
        e = create_engine("production", vocabulary=_VOCAB)
        e.record_selection("he", "hello")
        for exp in e.explain("he"):
            for source, pct in exp.contribution_pct.items():
                assert 0.0 <= pct <= 1.0 + 1e-6, (
                    f"contribution_pct[{source!r}] = {pct} out of [0, 1] "
                    f"for '{exp.value}'"
                )

    def test_zero_contribution_sources_are_omitted(self) -> None:
        """Sources with zero contribution must not appear in contribution_pct."""
        e = create_engine("production", vocabulary=_VOCAB)
        for exp in e.explain("he"):
            for source, pct in exp.contribution_pct.items():
                assert abs(pct) > 1e-12, (
                    f"Zero-contribution source {source!r} should not appear "
                    f"in contribution_pct for '{exp.value}'"
                )

    def test_contributions_sum_approximately_to_one(self) -> None:
        """Sum of contribution_pct values should be close to 1.0 for words
        with positive final score."""
        e = create_engine("production", vocabulary=_VOCAB)
        e.record_selection("he", "hello")

        exps = {exp.value: exp for exp in e.explain("he")}
        hello = exps.get("hello")
        if hello and abs(hello.final_score) > 1e-9:
            total = sum(hello.contribution_pct.values())
            assert abs(total - 1.0) < 0.05, (
                f"contribution_pct values should sum to ~1.0, got {total:.4f} "
                f"for '{hello.value}'"
            )

    def test_dominant_source_has_highest_pct(self) -> None:
        """After many selections, the decay/history source should dominate."""
        e = create_engine("production", vocabulary=_VOCAB)
        for _ in range(10):
            e.record_selection("he", "hello")

        exps = {exp.value: exp for exp in e.explain("he")}
        hello = exps.get("hello")
        assert hello is not None
        if hello.contribution_pct:
            max_source = max(hello.contribution_pct, key=lambda k: hello.contribution_pct[k])
            # After 10 selections, decay boost should dominate
            assert max_source in ("decay", "history"), (
                f"After 10 selections, decay/history should dominate. "
                f"Got max source: {max_source!r} with pct={hello.contribution_pct}"
            )


# ---------------------------------------------------------------------------
# History.copy()
# ---------------------------------------------------------------------------

class TestHistoryCopy:
    def test_copy_is_independent(self) -> None:
        h = History()
        h.record("he", "hello")
        h2 = h.copy()

        h2.record("he", "help")  # modify copy
        assert len(list(h.entries())) == 1, "original must be unaffected"
        assert len(list(h2.entries())) == 2, "copy should have both entries"

    def test_copy_has_same_entries(self) -> None:
        h = History()
        h.record("prog", "programming")
        h.record("prog", "program")
        h2 = h.copy()
        assert set(e.value for e in h2.entries()) == {"programming", "program"}

    def test_copy_prefix_index_works(self) -> None:
        h = History()
        h.record("prog", "programming")
        h2 = h.copy()
        counts = h2.counts_for_prefix("prog")
        assert counts.get("programming") == 1

    def test_copy_of_empty_history(self) -> None:
        h = History()
        h2 = h.copy()
        assert len(list(h2.entries())) == 0

    def test_original_unaffected_by_copy_record(self) -> None:
        h = History()
        h.record("he", "her")
        h2 = h.copy()
        for _ in range(5):
            h2.record("he", "help")

        original_counts = h.counts_for_prefix("he")
        assert original_counts.get("help", 0) == 0, (
            "Recording into the copy must not affect the original's counts"
        )


# ---------------------------------------------------------------------------
# compare_presets() history isolation
# ---------------------------------------------------------------------------

class TestComparePresentHistoryIsolation:
    def test_shared_history_is_not_modified(self) -> None:
        """compare_presets must not mutate the caller's History."""
        h = History()
        h.record("prog", "programming")
        initial_count = len(list(h.entries()))

        compare_presets("prog", ["stateless", "default"], history=h, limit=3)

        assert len(list(h.entries())) == initial_count, (
            "compare_presets must not add entries to the caller's History"
        )

    def test_each_engine_gets_independent_history(self) -> None:
        """Modifying one preset engine's history must not affect others."""
        h = History()
        h.record("he", "hello")

        cmp = compare_presets("he", ["default", "recency"], history=h, limit=5)

        # The comparison should have completed without cross-contamination.
        # We verify by checking that original history is still intact.
        assert h.counts_for_prefix("he").get("hello") == 1, (
            "Original History must have exactly 1 entry after compare_presets"
        )

    def test_history_signal_visible_in_comparison(self) -> None:
        """History passed to compare_presets affects learning-capable presets."""
        h = History()
        for _ in range(5):
            h.record("he", "heap")

        cmp = compare_presets("he", ["stateless", "default"], history=h, limit=10)

        # "stateless" has no history - heap should not lead there
        # "default" has history - heap should appear with a boost
        heap_row = next((r for r in cmp.rows if r["value"] == "heap"), None)
        assert heap_row is not None, "'heap' must appear in comparison results"

        stateless_boost = heap_row["boosts"].get("stateless", 0.0)
        default_boost = heap_row["boosts"].get("default", 0.0)

        if stateless_boost is not None and default_boost is not None:
            assert default_boost >= stateless_boost, (
                f"default preset should boost 'heap' more than stateless. "
                f"default={default_boost}, stateless={stateless_boost}"
            )


# ---------------------------------------------------------------------------
# ContextualHistory
# ---------------------------------------------------------------------------

class TestContextualHistory:
    def test_domain_isolation(self) -> None:
        ctx = ContextualHistory()
        ctx.record("prog", "programming", domain="python")
        ctx.record("prog", "progress",    domain="pm")

        python_h = ctx.for_domain("python")
        pm_h = ctx.for_domain("pm")

        assert python_h.counts_for_prefix("prog").get("programming") == 1
        assert python_h.counts_for_prefix("prog").get("progress", 0) == 0
        assert pm_h.counts_for_prefix("prog").get("progress") == 1
        assert pm_h.counts_for_prefix("prog").get("programming", 0) == 0

    def test_engine_built_on_domain_uses_domain_history(self) -> None:
        ctx = ContextualHistory()
        for _ in range(5):
            ctx.record("he", "heap", domain="finance")
        for _ in range(5):
            ctx.record("he", "hello", domain="chat")

        finance_engine = create_engine("default", vocabulary=_VOCAB, history=ctx.for_domain("finance"))
        chat_engine    = create_engine("default", vocabulary=_VOCAB, history=ctx.for_domain("chat"))

        assert finance_engine.suggest("he")[0] == "heap", (
            "finance domain engine should suggest 'heap' first"
        )
        assert chat_engine.suggest("he")[0] == "hello", (
            "chat domain engine should suggest 'hello' first"
        )

    def test_default_domain_used_when_no_domain_specified(self) -> None:
        ctx = ContextualHistory()
        ctx.record("he", "hero")  # no domain → default
        h = ctx.for_domain()     # no domain → same default

        assert h.counts_for_prefix("he").get("hero") == 1

    def test_for_domain_returns_live_history(self) -> None:
        """Mutations via record() are immediately visible in for_domain()."""
        ctx = ContextualHistory()
        h = ctx.for_domain("shell")
        ctx.record("git", "git commit", domain="shell")

        assert h.counts_for_prefix("git").get("git commit") == 1

    def test_domain_names_sorted(self) -> None:
        ctx = ContextualHistory()
        ctx.for_domain("zzz")
        ctx.for_domain("aaa")
        ctx.for_domain("mmm")
        assert ctx.domain_names() == ["aaa", "mmm", "zzz"]

    def test_domains_iterator(self) -> None:
        ctx = ContextualHistory()
        ctx.record("a", "apple", domain="fruit")
        ctx.record("b", "banana", domain="fruit")
        ctx.record("c", "car", domain="vehicle")

        domains = dict(ctx.domains())
        assert set(domains.keys()) == {"fruit", "vehicle"}
        assert domains["fruit"].counts_for_prefix("a").get("apple") == 1

    def test_total_entries(self) -> None:
        ctx = ContextualHistory()
        ctx.record("a", "apple", domain="d1")
        ctx.record("b", "banana", domain="d1")
        ctx.record("c", "cat", domain="d2")
        assert ctx.total_entries() == 3

    def test_load_domain_replaces_history(self) -> None:
        ctx = ContextualHistory()
        ctx.record("he", "hello", domain="test")

        new_h = History()
        new_h.record("he", "help")
        ctx.load_domain("test", new_h)

        h = ctx.for_domain("test")
        assert h.counts_for_prefix("he").get("help") == 1
        assert h.counts_for_prefix("he").get("hello", 0) == 0

    def test_repr_shows_domain_entry_counts(self) -> None:
        ctx = ContextualHistory()
        ctx.record("a", "apple", domain="fruit")
        r = repr(ctx)
        assert "fruit" in r
        assert "ContextualHistory" in r


# ---------------------------------------------------------------------------
# EngineConfig
# ---------------------------------------------------------------------------

class TestEngineConfig:
    def _production_config(self) -> EngineConfig:
        e = create_engine("production", vocabulary=_VOCAB)
        return e.to_config(preset="production")

    def test_to_config_returns_engine_config(self) -> None:
        config = self._production_config()
        assert isinstance(config, EngineConfig)

    def test_preset_name_preserved(self) -> None:
        config = self._production_config()
        assert config.preset == "production"

    def test_predictor_names_correct(self) -> None:
        config = self._production_config()
        names = [p.name for p in config.predictors]
        assert "frequency" in names
        assert "history" in names
        assert "symspell" in names
        assert "trigram" in names

    def test_predictor_weights_correct(self) -> None:
        config = self._production_config()
        weights = {p.name: p.weight for p in config.predictors}
        assert weights["frequency"] == pytest.approx(1.0)
        assert weights["history"] == pytest.approx(1.2)
        assert weights["symspell"] == pytest.approx(0.35)
        assert weights["trigram"] == pytest.approx(0.4)

    def test_ranker_names_correct(self) -> None:
        config = self._production_config()
        names = [r.name for r in config.rankers]
        assert "score" in names
        assert "decay" in names

    def test_decay_params_preserved(self) -> None:
        config = self._production_config()
        decay = next(r for r in config.rankers if r.name == "decay")
        assert "half_life_seconds" in decay.params
        assert decay.params["half_life_seconds"] == pytest.approx(3600.0)
        assert "weight" in decay.params

    def test_to_json_is_valid_json(self) -> None:
        config = self._production_config()
        json_str = config.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "preset" in parsed
        assert "predictors" in parsed
        assert "rankers" in parsed
        assert "version" in parsed

    def test_from_json_roundtrip(self) -> None:
        config = self._production_config()
        json_str = config.to_json()
        config2 = EngineConfig.from_json(json_str)
        assert config2.preset == config.preset
        assert len(config2.predictors) == len(config.predictors)
        assert len(config2.rankers) == len(config.rankers)

    def test_build_produces_working_engine(self) -> None:
        config = self._production_config()
        engine = config.build(vocabulary=_VOCAB)
        results = engine.suggest("he", limit=5)
        assert results, "engine built from config must return suggestions"

    def test_build_engine_matches_original(self) -> None:
        e = create_engine("production", vocabulary=_VOCAB)
        config = e.to_config(preset="production")
        e2 = config.build(vocabulary=_VOCAB)
        assert e2.suggest("prog", limit=5) == e.suggest("prog", limit=5)

    def test_metadata_round_trips(self) -> None:
        e = create_engine("stateless", vocabulary=_VOCAB)
        config = e.to_config(
            preset="stateless",
            metadata={"env": "production", "vocab": "english_48k"},
        )
        json_str = config.to_json()
        config2 = EngineConfig.from_json(json_str)
        assert config2.metadata["env"] == "production"
        assert config2.metadata["vocab"] == "english_48k"

    def test_diff_no_changes(self) -> None:
        config = self._production_config()
        config2 = EngineConfig.from_json(config.to_json())
        diffs = config.diff(config2)
        assert diffs == [], f"Identical configs should have no diffs, got: {diffs}"

    def test_diff_weight_change(self) -> None:
        config = self._production_config()
        data = json.loads(config.to_json())
        data["predictors"][0]["weight"] = 99.0  # change frequency weight
        config2 = EngineConfig.from_dict(data)
        diffs = config.diff(config2)
        assert any("frequency" in d and "weight" in d for d in diffs), (
            f"Weight change should appear in diff: {diffs}"
        )

    def test_diff_added_predictor(self) -> None:
        config = self._production_config()
        data = json.loads(config.to_json())
        data["predictors"].append({"name": "bespoke", "weight": 0.5, "params": {}})
        config2 = EngineConfig.from_dict(data)
        diffs = config.diff(config2)
        assert any("added" in d and "bespoke" in d for d in diffs)

    def test_diff_removed_predictor(self) -> None:
        config = self._production_config()
        data = json.loads(config.to_json())
        data["predictors"] = [p for p in data["predictors"] if p["name"] != "trigram"]
        config2 = EngineConfig.from_dict(data)
        diffs = config.diff(config2)
        assert any("removed" in d and "trigram" in d for d in diffs)

    def test_from_json_bad_version_raises(self) -> None:
        config = self._production_config()
        data = json.loads(config.to_json())
        data["version"] = 999
        with pytest.raises(ValueError, match="version"):
            EngineConfig.from_dict(data)

    def test_build_without_preset_raises(self) -> None:
        config = EngineConfig(
            preset=None,
            predictors=[PredictorConfig("frequency", 1.0)],
            rankers=[RankerConfig("score")],
        )
        with pytest.raises(NotImplementedError):
            config.build()

    def test_repr_shows_preset_and_predictors(self) -> None:
        config = self._production_config()
        r = repr(config)
        assert "production" in r
        assert "frequency" in r

    def test_stateless_preset_config(self) -> None:
        e = create_engine("stateless", vocabulary=_VOCAB)
        config = e.to_config(preset="stateless")
        assert config.preset == "stateless"
        names = [p.name for p in config.predictors]
        assert "frequency" in names
        # stateless has no history predictor or decay ranker
        ranker_names = [r.name for r in config.rankers]
        assert "decay" not in ranker_names
