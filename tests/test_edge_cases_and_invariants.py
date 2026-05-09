"""
Edge case tests for suggest_with_confidence() and the _deletes() helper.

These cover paths that the main test suite did not exercise:
  - suggest_with_confidence() with near-zero top score
  - suggest_with_confidence() with empty result set
  - _deletes() correctness including the necessary role of empty string
  - _deletes() correctness for normal inputs
"""
from __future__ import annotations

import pytest

from aac.domain.types import CompletionContext, WeightedPredictor
from aac.engine.engine import AutocompleteEngine
from aac.predictors.frequency import FrequencyPredictor
from aac.predictors.symspell import SymSpellPredictor, _deletes
from aac.presets import create_engine

# ---------------------------------------------------------------------------
# suggest_with_confidence edge cases
# ---------------------------------------------------------------------------

class TestSuggestWithConfidence:
    def test_empty_prefix_returns_empty(self) -> None:
        engine = create_engine("stateless")
        result = engine.suggest_with_confidence("")
        assert result == []

    def test_normal_case_top_confidence_is_one(self) -> None:
        engine = create_engine("stateless")
        result = engine.suggest_with_confidence("prog", limit=5)
        assert result, "Expected at least one suggestion"
        top_word, top_conf = result[0]
        assert top_conf == pytest.approx(1.0), (
            f"Top confidence should be 1.0, got {top_conf}"
        )

    def test_all_confidences_in_range(self) -> None:
        engine = create_engine("production")
        result = engine.suggest_with_confidence("he", limit=10)
        for word, conf in result:
            assert 0.0 < conf <= 1.0, (
                f"Confidence for {word!r} out of range: {conf}"
            )

    def test_heavy_history_does_not_collapse_alternatives(self) -> None:
        """
        After many selections, the second suggestion must still have meaningful
        confidence (> 20%), not collapse toward 0.

        This was the core bug: raw score division after 5 selections gave the
        second result ~6% confidence. The hybrid approach gives it ~71%.
        """
        engine = create_engine("production")
        for _ in range(5):
            engine.record_selection("prog", "programming")

        result = engine.suggest_with_confidence("prog", limit=5)
        assert result, "Expected suggestions"
        assert result[0][0] == "programming", "Top should be programming after 5 selections"

        # Second suggestion must not collapse to near-zero
        _, second_conf = result[1]
        assert second_conf > 0.2, (
            f"Second result confidence collapsed to {second_conf:.3f}. "
            f"Expected > 0.2. Raw score division after history boost breaks this."
        )

    def test_near_zero_top_score_no_division_error(self) -> None:
        """
        When all candidates have near-zero scores relative to the max-frequency
        word, the normaliser must not produce infinite or NaN confidences.
        """
        tiny_vocab = {"ab": 1, "abc": 1, "abd": 1, "abe": 1}
        giant_vocab = {"z" * 10: 10_000_000_000, **tiny_vocab}

        engine = AutocompleteEngine(
            predictors=[WeightedPredictor(FrequencyPredictor(giant_vocab), weight=1.0)],
        )

        result = engine.suggest_with_confidence("ab", limit=5)
        assert result, "Expected suggestions"
        for word, conf in result:
            assert 0.0 < conf <= 1.0, f"Confidence out of range for {word!r}: {conf}"
            assert conf == conf, f"NaN confidence for {word!r}"
            assert conf != float("inf"), f"Infinite confidence for {word!r}"

    def test_confidences_are_non_increasing(self) -> None:
        """Confidence scores must be non-increasing (highest-ranked first)."""
        engine = create_engine("stateless")
        result = engine.suggest_with_confidence("pro", limit=10)
        confs = [c for _, c in result]
        assert confs == sorted(confs, reverse=True), (
            f"Confidences are not non-increasing: {confs}"
        )

    def test_with_history_top_is_selected_word(self) -> None:
        """After enough selections, the selected word should be top confidence."""
        engine = create_engine("production")
        for _ in range(5):
            engine.record_selection("prog", "programming")

        result = engine.suggest_with_confidence("prog", limit=5)
        assert result, "Expected suggestions"
        top_word, _ = result[0]
        assert top_word == "programming", (
            f"Expected 'programming' at top after 5 selections, got {top_word!r}"
        )


# ---------------------------------------------------------------------------
# _deletes() correctness tests
# ---------------------------------------------------------------------------

class TestDeletes:
    def test_single_char_word_produces_empty_string(self) -> None:
        """
        _deletes('a', 1) must produce '' in its output.
        The empty string is a legitimate delete-neighbour key: both 'a' and
        'b' map to '', so SymSpell correctly identifies them as edit-distance-1
        candidates (substitution) via their shared '' key. Filtering '' here
        would introduce false negatives for all single-char vocab word lookups.
        """
        result = _deletes("a", max_distance=1)
        assert "" in result, (
            "_deletes('a', 1) must produce ''. "
            "It is needed for 1-char words to find each other via the shared '' key."
        )

    def test_two_char_word_max_distance_2_produces_empty(self) -> None:
        """'he' -> 'h' -> '' and 'he' -> 'e' -> '': '' is reachable in 2 steps."""
        result = _deletes("he", max_distance=2)
        assert "" in result

    def test_original_word_not_in_deletes(self) -> None:
        """The original word is never included in its own delete set."""
        for word in ["hello", "a", "ab", "test"]:
            result = _deletes(word, max_distance=2)
            assert word not in result, f"{word!r} should not appear in its own _deletes"

    def test_known_distance_1_deletes(self) -> None:
        """Verify the specific delete-neighbours of 'abc' at distance 1."""
        result = _deletes("abc", max_distance=1)
        assert "bc" in result
        assert "ac" in result
        assert "ab" in result

    def test_distance_2_includes_distance_1(self) -> None:
        """Distance-2 set must be a superset of distance-1 set."""
        word = "hello"
        d1 = _deletes(word, max_distance=1)
        d2 = _deletes(word, max_distance=2)
        assert d1.issubset(d2), "Distance-2 deletes must include all distance-1 deletes"

    def test_empty_word_returns_empty(self) -> None:
        """_deletes('', ...) should return an empty frozenset (nothing to delete)."""
        result = _deletes("", max_distance=1)
        assert result == frozenset(), f"Expected empty frozenset, got {result}"

    def test_symspell_single_char_vocab_finds_substitutions(self) -> None:
        """
        SymSpellPredictor must find other 1-char words as edit-distance-1
        candidates. 'b', 'c', 'd' are all edit-distance 1 from 'a'
        (one substitution), and SymSpell finds this via the shared '' key.
        This verifies the '' key is preserved in the index, not filtered out.
        """
        vocab = ["a", "b", "c", "d"]
        predictor = SymSpellPredictor(vocab, max_distance=1)
        ctx = CompletionContext("a")
        results = {r.suggestion.value for r in predictor.predict(ctx)}
        # 'a' itself (exact match) plus substitutions 'b', 'c', 'd'
        assert "b" in results, f"'b' should be found from 'a' at distance 1. Got: {results}"
        assert "c" in results
        assert "d" in results

    def test_symspell_two_char_query_finds_other_two_char_words(self) -> None:
        """
        2-char query should find 2-char words within edit distance 2.
        This exercises the path that goes through '' as an intermediate key.
        """
        vocab = ["he", "an", "or", "so", "to", "hello", "world"]
        predictor = SymSpellPredictor(vocab, max_distance=2)
        ctx = CompletionContext("he")
        results = {r.suggestion.value for r in predictor.predict(ctx)}
        # 'an' is 2 edits from 'he' (h->a, e->n): must be found
        assert "an" in results, f"'an' should be within distance 2 of 'he'. Got: {results}"
