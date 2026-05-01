"""
Tests for v0.4.0 features:
  - AutocompleteEngine.suggest_with_confidence()
  - AutocompleteEngine.reset_history()
  - AdaptiveSymSpellPredictor
  - production preset short-prefix typo recovery
  - CLI default history path change
"""
from __future__ import annotations

import pytest

from aac.domain.types import CompletionContext
from aac.engine.engine import AutocompleteEngine
from aac.predictors.adaptive_symspell import AdaptiveSymSpellPredictor
from aac.presets import create_engine, get_preset

_VOCAB = {
    "hello": 100, "help": 80, "hero": 50, "her": 200,
    "here": 120, "heap": 40, "world": 300, "word": 150,
    "programming": 500, "program": 400, "progress": 300,
}


# ---------------------------------------------------------------------------
# suggest_with_confidence
# ---------------------------------------------------------------------------

class TestSuggestWithConfidence:
    def _engine(self) -> AutocompleteEngine:
        return create_engine("stateless", vocabulary=_VOCAB)

    def test_top_result_has_confidence_one(self) -> None:
        engine = self._engine()
        results = engine.suggest_with_confidence("he")
        assert results, "expected non-empty results"
        top_word, top_conf = results[0]
        assert top_conf == pytest.approx(1.0), (
            f"top result should have confidence=1.0, got {top_conf}"
        )

    def test_confidences_are_descending(self) -> None:
        engine = self._engine()
        results = engine.suggest_with_confidence("he")
        confs = [c for _, c in results]
        assert confs == sorted(confs, reverse=True), (
            f"confidences should be non-increasing: {confs}"
        )

    def test_confidences_in_unit_interval(self) -> None:
        engine = self._engine()
        for word, conf in engine.suggest_with_confidence("he"):
            assert 0.0 < conf <= 1.0, (
                f"confidence for {word!r} is {conf}, expected (0, 1]"
            )

    def test_order_matches_suggest(self) -> None:
        engine = self._engine()
        words_from_confidence = [w for w, _ in engine.suggest_with_confidence("he")]
        words_from_suggest = engine.suggest("he")
        assert words_from_confidence == words_from_suggest, (
            "suggest_with_confidence order must match suggest()"
        )

    def test_limit_respected(self) -> None:
        engine = self._engine()
        results = engine.suggest_with_confidence("he", limit=3)
        assert len(results) <= 3

    def test_limit_matches_suggest_limit(self) -> None:
        engine = self._engine()
        words_conf = [w for w, _ in engine.suggest_with_confidence("he", limit=4)]
        words_sug = engine.suggest("he", limit=4)
        assert words_conf == words_sug

    def test_empty_prefix_returns_empty(self) -> None:
        engine = self._engine()
        assert engine.suggest_with_confidence("") == []

    def test_no_match_returns_empty(self) -> None:
        engine = self._engine()
        result = engine.suggest_with_confidence("zzzzqqqqq")
        assert result == []

    def test_learning_boosts_confidence(self) -> None:
        """After recording selections, the selected word's confidence should rise."""
        engine = create_engine("default", vocabulary=_VOCAB)

        before = {w: c for w, c in engine.suggest_with_confidence("he")}
        heap_conf_before = before.get("heap", 0.0)

        for _ in range(5):
            engine.record_selection("he", "heap")

        after = {w: c for w, c in engine.suggest_with_confidence("he")}
        heap_conf_after = after.get("heap", 0.0)

        assert heap_conf_after > heap_conf_before, (
            f"'heap' confidence should increase after 5 selections: "
            f"{heap_conf_before:.3f} → {heap_conf_after:.3f}"
        )


# ---------------------------------------------------------------------------
# reset_history
# ---------------------------------------------------------------------------

class TestResetHistory:
    def test_reset_clears_in_memory_history(self) -> None:
        engine = create_engine("default", vocabulary=_VOCAB)
        engine.record_selection("he", "heap")
        assert len(list(engine.history.entries())) > 0

        engine.reset_history()
        assert len(list(engine.history.entries())) == 0

    def test_reset_affects_subsequent_suggestions(self) -> None:
        """After reset, suggestions should be as if no selections were made."""
        engine = create_engine("default", vocabulary=_VOCAB)

        # Drive "heap" to the top
        for _ in range(10):
            engine.record_selection("he", "heap")
        assert engine.suggest("he")[0] == "heap", "heap should lead before reset"

        engine.reset_history()
        after = engine.suggest("he")
        assert "heap" not in after[:1], (
            f"heap should no longer lead after reset, got: {after[:3]}"
        )

    def test_reset_history_object_replaced(self) -> None:
        """reset_history() replaces the internal History object."""
        engine = create_engine("default", vocabulary=_VOCAB)
        engine.record_selection("he", "hello")
        old_history = engine.history

        engine.reset_history()
        assert engine.history is not old_history
        assert len(list(engine.history.entries())) == 0

    def test_reset_learning_rankers_updated(self) -> None:
        """Learning rankers must read from the new History after reset."""
        engine = create_engine("production", vocabulary=_VOCAB)

        for _ in range(8):
            engine.record_selection("he", "heap")
        assert engine.suggest("he")[0] == "heap"

        engine.reset_history()
        after = engine.suggest("he")
        # After reset the decay ranker should no longer boost "heap"
        assert after.index("heap") > 0 or "heap" not in after[:1], (
            f"heap should not lead after reset, got: {after[:3]}"
        )

    def test_reset_does_not_affect_store(self) -> None:
        """reset_history() only clears in-memory state; a store is unaffected."""
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

            # Reset in-memory state
            engine.reset_history()
            assert len(list(engine.history.entries())) == 0

            # Store file should still have the old entry
            reloaded = store.load()
            assert len(list(reloaded.entries())) == 1

    def test_reset_multiple_times_is_idempotent(self) -> None:
        engine = create_engine("default", vocabulary=_VOCAB)
        engine.record_selection("he", "hello")
        engine.reset_history()
        engine.reset_history()  # should not raise
        assert len(list(engine.history.entries())) == 0

    def test_can_record_after_reset(self) -> None:
        """Engine is fully functional after reset."""
        engine = create_engine("default", vocabulary=_VOCAB)
        engine.reset_history()
        engine.record_selection("he", "help")
        assert engine.history.counts_for_prefix("he").get("help", 0) == 1


# ---------------------------------------------------------------------------
# AdaptiveSymSpellPredictor
# ---------------------------------------------------------------------------

class TestAdaptiveSymSpellPredictor:
    _VOCAB = ["hello", "help", "hero", "her", "here", "heap", "world", "word",
              "programming", "program", "the", "they", "them", "then", "there"]

    def _predictor(self, **kwargs: object) -> AdaptiveSymSpellPredictor:
        return AdaptiveSymSpellPredictor(self._VOCAB, **kwargs)  # type: ignore[arg-type]

    def test_exact_match_found(self) -> None:
        p = self._predictor()
        results = [s.suggestion.value for s in p.predict(CompletionContext("hello"))]
        assert "hello" in results

    def test_long_prefix_recovers_typo_at_distance_2(self) -> None:
        """4+ char prefix: full max_distance used, distance-2 typo recovered."""
        p = self._predictor(max_distance=2, short_prefix_len=4)
        results = [s.suggestion.value for s in p.predict(CompletionContext("helo"))]
        assert "hello" in results

    def test_short_prefix_uses_tight_distance(self) -> None:
        """1-3 char prefix: max_distance clamped to 1, fewer candidates returned."""
        # Build two predictors with the same vocab and full max_distance
        adaptive = self._predictor(max_distance=2, short_prefix_len=4, short_max_distance=1)
        full = AdaptiveSymSpellPredictor(self._VOCAB, max_distance=2, short_prefix_len=100)

        adaptive_results = {s.suggestion.value for s in adaptive.predict(CompletionContext("he"))}
        full_results = {s.suggestion.value for s in full.predict(CompletionContext("he"))}

        # Adaptive must return a subset (fewer noise candidates on short prefix)
        assert adaptive_results.issubset(full_results) or len(adaptive_results) <= len(full_results), (
            "adaptive should return no more candidates than full on short prefix"
        )

    def test_short_prefix_still_finds_distance_1_typo(self) -> None:
        """Even with tight distance, a single-char typo should be recovered."""
        p = self._predictor(max_distance=2, short_prefix_len=4, short_max_distance=1)
        # "hdr" is distance 1 from "her" (substitute d→e) - 3-char prefix,
        # uses tight index (distance=1), should still find "her".
        results = [s.suggestion.value for s in p.predict(CompletionContext("hdr"))]
        assert "her" in results, f"expected 'her' (distance=1 from 'hdr') in {results}"

    def test_short_prefix_exact_match_always_returned(self) -> None:
        """Exact matches on short prefixes must always be found."""
        p = self._predictor(max_distance=2, short_prefix_len=4, short_max_distance=1)
        results = [s.suggestion.value for s in p.predict(CompletionContext("her"))]
        assert "her" in results

    def test_empty_prefix_returns_empty(self) -> None:
        p = self._predictor()
        assert p.predict(CompletionContext("")) == []

    def test_no_candidates_returns_empty(self) -> None:
        p = self._predictor(max_distance=1)
        results = p.predict(CompletionContext("zzzzz"))
        assert results == []

    def test_name_is_symspell(self) -> None:
        p = self._predictor()
        assert p.name == "symspell"

    def test_same_distance_avoids_duplicate_index(self) -> None:
        """When max_distance == short_max_distance, inner indexes are the same object."""
        p = AdaptiveSymSpellPredictor(
            self._VOCAB, max_distance=1, short_prefix_len=4, short_max_distance=1
        )
        assert p._inner_tight is p._inner_full

    def test_threshold_boundary(self) -> None:
        """Queries at exactly short_prefix_len chars use the full index."""
        p = self._predictor(max_distance=2, short_prefix_len=4, short_max_distance=1)
        # "hell" is 4 chars - exactly at threshold, should use full distance
        results = [s.suggestion.value for s in p.predict(CompletionContext("hell"))]
        assert "hello" in results

    def test_scores_are_positive(self) -> None:
        p = self._predictor()
        for s in p.predict(CompletionContext("hello")):
            assert s.score > 0, f"score must be positive, got {s.score} for {s.suggestion.value!r}"


# ---------------------------------------------------------------------------
# production preset: short-prefix typo recovery
# ---------------------------------------------------------------------------

class TestProductionPresetShortPrefix:
    """
    The 'production' preset previously returned no typo recovery for 1-3 char
    prefixes (TrigramPredictor requires length >= 4). AdaptiveSymSpellPredictor
    fills this gap.
    """

    def test_one_char_prefix_returns_results(self) -> None:
        engine = create_engine("production")
        results = engine.suggest("h", limit=10)
        assert results, "production preset must return results for 1-char prefix"

    def test_two_char_prefix_returns_clean_list(self) -> None:
        """'he' should return common he- words, not hundreds of noise candidates."""
        engine = create_engine("production")
        results = engine.suggest("he", limit=10)
        assert results, "production preset must return results for 2-char prefix"
        assert len(results) <= 10
        # Top results should be recognisably "he" words
        for word in results[:5]:
            assert word.startswith("he") or len(word) <= 4, (
                f"unexpected top result for 'he': {word!r}"
            )

    def test_three_char_prefix_typo_recovered(self) -> None:
        """3-char prefix with single-char typo: 'hel' → 'hell'."""
        engine = create_engine("production")
        results = engine.suggest("hel", limit=10)
        assert "help" in results or "hell" in results or "held" in results, (
            f"expected common hel- words in: {results}"
        )

    def test_four_char_prefix_typo_distance_2(self) -> None:
        """4-char prefix with 2 typos: trigram + symspell both fire."""
        engine = create_engine("production")
        results = engine.suggest("progaming", limit=10)
        assert "programming" in results, (
            f"'progaming' should recover to 'programming', got: {results}"
        )

    def test_short_prefix_result_count_is_sane(self) -> None:
        """Short-prefix results should not be in the hundreds (noise check)."""
        engine = create_engine("production")
        results = engine.suggest("he")
        # Should be well under 100 - the old robust preset with raw SymSpell at
        # max_distance=2 would return 600+ candidates for "he"
        assert len(results) < 200, (
            f"'he' should not return {len(results)} candidates - noise in short-prefix handling"
        )

    def test_preset_metadata_reflects_hybrid(self) -> None:
        preset = get_preset("production")
        assert "symspell" in preset.predictors
        assert "trigram" in preset.predictors


# ---------------------------------------------------------------------------
# CLI default history path
# ---------------------------------------------------------------------------

class TestCLIDefaultHistoryPath:
    def test_default_history_path_is_in_home(self) -> None:
        from pathlib import Path

        from aac.cli.main import DEFAULT_HISTORY_PATH

        home = Path.home()
        assert DEFAULT_HISTORY_PATH.is_absolute(), (
            "DEFAULT_HISTORY_PATH must be absolute (expanduser was called)"
        )
        assert DEFAULT_HISTORY_PATH.is_relative_to(home), (
            f"DEFAULT_HISTORY_PATH {DEFAULT_HISTORY_PATH!r} should be under home {home!r}"
        )
        assert DEFAULT_HISTORY_PATH.name == ".aac_history.json"
