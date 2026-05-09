"""
Tests for the rich suggestion API surface:
  - suggest_with_confidence()
  - suggest_with_history()
  - suggest_full()

These are the three methods that return richer data than plain suggest().
They share pipeline structure (score → rank → annotate) so their correctness
properties are tested together.
"""
from __future__ import annotations

import math

import pytest

from aac.engine.engine import AutocompleteEngine
from aac.presets import create_engine

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

    def test_top_result_confidence_is_one(self) -> None:
        results = self._engine().suggest_with_confidence("he")
        assert results, "expected non-empty results for 'he'"
        _, top_conf = results[0]
        assert top_conf == pytest.approx(1.0), (
            f"top confidence must be exactly 1.0, got {top_conf}"
        )

    def test_confidences_descending(self) -> None:
        confs = [c for _, c in self._engine().suggest_with_confidence("he")]
        assert confs == sorted(confs, reverse=True), (
            f"confidences must be non-increasing: {confs}"
        )

    def test_confidences_in_unit_interval(self) -> None:
        for word, conf in self._engine().suggest_with_confidence("he"):
            assert 0.0 < conf <= 1.0, (
                f"confidence for {word!r} out of (0, 1]: {conf}"
            )

    def test_order_matches_suggest(self) -> None:
        e = self._engine()
        words_conf = [w for w, _ in e.suggest_with_confidence("he")]
        assert words_conf == e.suggest("he"), (
            "suggest_with_confidence order must match suggest()"
        )

    def test_limit_respected(self) -> None:
        assert len(self._engine().suggest_with_confidence("he", limit=3)) <= 3

    def test_limit_matches_suggest(self) -> None:
        e = self._engine()
        words_conf = [w for w, _ in e.suggest_with_confidence("he", limit=4)]
        assert words_conf == e.suggest("he", limit=4)

    def test_empty_prefix_returns_empty(self) -> None:
        assert self._engine().suggest_with_confidence("") == []

    def test_no_match_returns_empty(self) -> None:
        assert self._engine().suggest_with_confidence("zzzzqqqq") == []

    def test_near_zero_top_score_stays_finite(self) -> None:
        """A tiny-frequency vocabulary must not produce infinite confidences.

        The guard is ``max(abs(top_score), 1e-9)`` - not ``top_score or 1.0``
        which only catches exactly zero.  A score of 1e-12 would explode all
        confidences toward infinity if the guard didn't use abs().
        """
        e = create_engine("stateless", vocabulary={"zephyr": 1})
        for word, conf in e.suggest_with_confidence("ze"):
            assert math.isfinite(conf), f"confidence for {word!r} is not finite: {conf}"
            assert 0.0 < conf <= 1.0, f"confidence for {word!r} out of (0, 1]: {conf}"

    def test_single_candidate_confidence_is_one(self) -> None:
        """When only one suggestion exists its confidence must be exactly 1.0."""
        e = create_engine("stateless", vocabulary={"zephyr": 100})
        results = e.suggest_with_confidence("ze")
        assert results, "expected 'zephyr' to match prefix 'ze'"
        assert len(results) == 1, f"expected exactly one result, got {len(results)}"
        assert results[0][1] == pytest.approx(1.0)

    def test_learning_increases_confidence(self) -> None:
        """After recording selections the boosted word's confidence must rise."""
        e = create_engine("default", vocabulary=_VOCAB)
        before = {w: c for w, c in e.suggest_with_confidence("he")}

        for _ in range(5):
            e.record_selection("he", "heap")

        after = {w: c for w, c in e.suggest_with_confidence("he")}
        assert after.get("heap", 0.0) > before.get("heap", 0.0), (
            f"'heap' confidence should increase after 5 selections: "
            f"{before.get('heap', 0.0):.4f} → {after.get('heap', 0.0):.4f}"
        )


# ---------------------------------------------------------------------------
# suggest_with_history
# ---------------------------------------------------------------------------

class TestSuggestWithHistory:
    def _engine(self) -> AutocompleteEngine:
        return create_engine("default", vocabulary=_VOCAB)

    def test_no_history_all_counts_zero(self) -> None:
        results = self._engine().suggest_with_history("he", limit=10)
        assert results, "expected results for 'he'"
        for word, count in results:
            assert count == 0, (
                f"'{word}' should have count=0 with no history, got {count}"
            )

    def test_recorded_selection_shows_count(self) -> None:
        e = self._engine()
        e.record_selection("he", "hello")
        e.record_selection("he", "hello")
        assert dict(e.suggest_with_history("he")).get("hello") == 2

    def test_unselected_words_have_count_zero(self) -> None:
        e = self._engine()
        e.record_selection("he", "hello")
        for word, count in e.suggest_with_history("he"):
            if word != "hello":
                assert count == 0, f"'{word}' should have count=0, got {count}"

    def test_order_matches_suggest(self) -> None:
        e = self._engine()
        e.record_selection("he", "help")
        words_hist = [w for w, _ in e.suggest_with_history("he")]
        assert words_hist == e.suggest("he"), (
            "suggest_with_history order must match suggest()"
        )

    def test_limit_respected(self) -> None:
        assert len(self._engine().suggest_with_history("he", limit=3)) <= 3

    def test_limit_matches_suggest(self) -> None:
        e = self._engine()
        words_hist = [w for w, _ in e.suggest_with_history("he", limit=4)]
        assert words_hist == e.suggest("he", limit=4)

    def test_empty_prefix_returns_empty(self) -> None:
        assert self._engine().suggest_with_history("") == []

    def test_no_match_returns_empty(self) -> None:
        assert self._engine().suggest_with_history("zzzzqqqq") == []

    def test_multiple_prefixes_are_independent(self) -> None:
        """Selections for 'he' must not affect counts for 'hel'."""
        e = self._engine()
        e.record_selection("he", "hello")
        e.record_selection("he", "hello")
        assert dict(e.suggest_with_history("he")).get("hello", 0) == 2
        # 'hel' has a different normalised prefix - no bleed-over
        assert dict(e.suggest_with_history("hel")).get("hello", 0) == 0, (
            "counts for 'he' prefix must not bleed into 'hel' prefix"
        )

    def test_multiple_different_selections_counted_separately(self) -> None:
        e = self._engine()
        e.record_selection("he", "hello")
        e.record_selection("he", "help")
        e.record_selection("he", "hello")
        results = dict(e.suggest_with_history("he"))
        assert results.get("hello") == 2
        assert results.get("help") == 1

    def test_reset_clears_counts(self) -> None:
        e = self._engine()
        e.record_selection("he", "hello")
        e.reset_history()
        assert all(c == 0 for _, c in e.suggest_with_history("he")), (
            "all counts must be 0 after reset_history()"
        )


# ---------------------------------------------------------------------------
# suggest_full (single-pass combination)
# ---------------------------------------------------------------------------

class TestSuggestFull:
    def _engine(self) -> AutocompleteEngine:
        return create_engine("default", vocabulary=_VOCAB)

    def test_returns_correct_keys(self) -> None:
        for item in self._engine().suggest_full("he", limit=5):
            assert "word" in item and "count" in item and "confidence" in item

    def test_order_matches_suggest(self) -> None:
        e = self._engine()
        full_words = [item["word"] for item in e.suggest_full("he")]
        assert full_words == e.suggest("he"), (
            "suggest_full order must match suggest()"
        )

    def test_counts_match_suggest_with_history(self) -> None:
        e = self._engine()
        e.record_selection("he", "hello")
        e.record_selection("he", "hello")
        full = {item["word"]: item["count"] for item in e.suggest_full("he")}
        hist = dict(e.suggest_with_history("he"))
        for word in full:
            assert full[word] == hist.get(word, 0), (
                f"suggest_full count for '{word}' ({full[word]}) must match "
                f"suggest_with_history ({hist.get(word, 0)})"
            )

    def test_confidences_match_suggest_with_confidence(self) -> None:
        e = self._engine()
        full = {item["word"]: item["confidence"] for item in e.suggest_full("he")}
        conf = dict(e.suggest_with_confidence("he"))
        for word in full:
            assert full[word] == pytest.approx(conf.get(word, 0.0), abs=1e-9), (
                f"suggest_full confidence for '{word}' must match suggest_with_confidence"
            )

    def test_empty_prefix_returns_empty(self) -> None:
        assert self._engine().suggest_full("") == []

    def test_limit_respected(self) -> None:
        assert len(self._engine().suggest_full("he", limit=3)) <= 3
