from __future__ import annotations

import math

import pytest

from aac.domain.types import CompletionContext
from aac.predictors.frequency import _DEFAULT_MAX_RESULTS, FrequencyPredictor


@pytest.fixture()
def predictor() -> FrequencyPredictor:
    return FrequencyPredictor({"hello": 10, "help": 5, "helium": 1, "world": 20})


def test_returns_words_matching_prefix(predictor: FrequencyPredictor) -> None:
    results = predictor.predict(CompletionContext("he"))
    values = {r.suggestion.value for r in results}
    assert values == {"hello", "help", "helium"}


def test_scores_reflect_frequency(predictor: FrequencyPredictor) -> None:
    """Scores are log-normalised: higher frequency → higher score, in (0, 1]."""
    results = predictor.predict(CompletionContext("he"))
    by_value = {r.suggestion.value: r.score for r in results}

    # Ordering must be preserved: hello > help > helium
    assert by_value["hello"] > by_value["help"] > by_value["helium"]

    # All scores must be in (0, 1]
    for score in by_value.values():
        assert 0.0 < score <= 1.0

    # Score formula: log(1+freq) / log(1+max_freq), max_freq=20 (world)
    log_max = math.log1p(20)
    assert by_value["hello"] == pytest.approx(math.log1p(10) / log_max)
    assert by_value["help"] == pytest.approx(math.log1p(5) / log_max)
    assert by_value["helium"] == pytest.approx(math.log1p(1) / log_max)


def test_does_not_return_non_matching_words(predictor: FrequencyPredictor) -> None:
    results = predictor.predict(CompletionContext("he"))
    values = {r.suggestion.value for r in results}
    assert "world" not in values


def test_exact_match_excluded(predictor: FrequencyPredictor) -> None:
    # Typing the complete word should not return the word itself -
    # completing 'hello' to 'hello' adds no information.
    results = predictor.predict(CompletionContext("hello"))
    values = {r.suggestion.value for r in results}
    assert "hello" not in values


def test_empty_prefix_returns_nothing(predictor: FrequencyPredictor) -> None:
    assert predictor.predict(CompletionContext("")) == []


def test_no_results_for_unknown_prefix(predictor: FrequencyPredictor) -> None:
    assert predictor.predict(CompletionContext("xyz")) == []


def test_confidence_is_log_normalised(predictor: FrequencyPredictor) -> None:
    """Confidence equals the log-normalised score (same value, different name)."""
    results = predictor.predict(CompletionContext("he"))
    by_value = {r.suggestion.value: r for r in results}

    assert by_value["hello"].explanation is not None
    assert by_value["helium"].explanation is not None

    # Confidence == score for FrequencyPredictor (log-normalised)
    for v in ("hello", "help", "helium"):
        assert by_value[v].explanation is not None
        assert by_value[v].explanation.confidence == pytest.approx(by_value[v].score)


def test_insertion_order_does_not_affect_result_set() -> None:
    # The result SET must be identical regardless of construction order.
    fp1 = FrequencyPredictor({"hello": 10, "help": 5, "helium": 1})
    fp2 = FrequencyPredictor({"helium": 1, "help": 5, "hello": 10})

    r1 = {r.suggestion.value for r in fp1.predict(CompletionContext("he"))}
    r2 = {r.suggestion.value for r in fp2.predict(CompletionContext("he"))}

    assert r1 == r2


def test_rejects_empty_frequencies() -> None:
    with pytest.raises(ValueError, match="frequencies must not be empty"):
        FrequencyPredictor({})

def test_rejects_all_zero_frequencies() -> None:
    with pytest.raises(ValueError, match="positive"):
        FrequencyPredictor({"hello": 0, "help": 0})

class TestFrequencyPredictorIndexCorrectness:
    """FrequencyPredictor index must exclude exact matches at build time."""

    def test_exact_match_not_in_index(self) -> None:
        """A word must not appear in results when the prefix equals the word exactly."""
        from aac.predictors.frequency import FrequencyPredictor
        vocab = {"hello": 100, "help": 80}
        predictor = FrequencyPredictor(vocab)
        results = [s.value for s in predictor.predict("hello")]
        assert "hello" not in results

    def test_index_does_not_contain_exact_match_key(self) -> None:
        """The internal index must not store any word under its own full string."""
        from aac.predictors.frequency import FrequencyPredictor
        vocab = {"hi": 10, "hello": 100}
        predictor = FrequencyPredictor(vocab)
        assert "hi" not in predictor._index
        assert "hello" not in predictor._index


class TestFrequencyPredictorValidation:
    """FrequencyPredictor must reject invalid construction arguments."""

    def test_max_results_less_than_one_raises(self) -> None:
        from aac.predictors.frequency import FrequencyPredictor
        with pytest.raises(ValueError, match="max_results"):
            FrequencyPredictor({"hello": 1}, max_results=0)

    def test_empty_frequencies_raises(self) -> None:
        from aac.predictors.frequency import FrequencyPredictor
        with pytest.raises(ValueError, match="frequencies must not be empty"):
            FrequencyPredictor({})


# ---------------------------------------------------------------------------
# add_word() - runtime vocabulary growth
# ---------------------------------------------------------------------------

class TestFrequencyPredictorAddWord:
    """
    add_word() allows runtime vocabulary updates without rebuilding the index.

    Test strategy: verify each branch of the method's conditional logic.
    The implementation has four non-trivial paths:
      1. No-op for frequency <= 0
      2. No-op for empty string
      3. New word (prefix bucket doesn't exist yet)
      4. Update existing word:
         a. to higher frequency (word moves earlier in bucket)
         b. to lower frequency (word falls to end of bucket)
         c. exceeding current max_freq (updates log normaliser)
    """

    def test_zero_frequency_is_ignored(self) -> None:
        """add_word with frequency=0 must not add or change anything."""
        p = FrequencyPredictor({"hello": 10, "help": 5})
        before = [s.suggestion.value for s in p.predict(CompletionContext("he"))]
        p.add_word("newword", 0)
        after = [s.suggestion.value for s in p.predict(CompletionContext("ne"))]
        assert after == [], "zero-frequency word must not appear in suggestions"
        # hello/help ordering must be unchanged
        assert [s.suggestion.value for s in p.predict(CompletionContext("he"))] == before

    def test_negative_frequency_is_ignored(self) -> None:
        p = FrequencyPredictor({"hello": 10})
        p.add_word("bad", -5)
        assert p.predict(CompletionContext("ba")) == []

    def test_empty_string_is_ignored(self) -> None:
        """add_word with an empty string must not corrupt the index."""
        p = FrequencyPredictor({"hello": 10})
        p.add_word("", 100)
        # Index must be unchanged - no key "" should appear
        assert "" not in p._index

    def test_new_word_appears_in_suggestions(self) -> None:
        """A newly added word that shares a prefix with vocab words must be suggested."""
        p = FrequencyPredictor({"hello": 10, "help": 5})
        p.add_word("herald", 8)
        results = {s.suggestion.value for s in p.predict(CompletionContext("he"))}
        assert "herald" in results

    def test_new_word_with_novel_prefix(self) -> None:
        """A word whose prefix has no existing bucket must create that bucket."""
        p = FrequencyPredictor({"hello": 10})
        p.add_word("zebra", 50)
        results = {s.suggestion.value for s in p.predict(CompletionContext("ze"))}
        assert "zebra" in results

    def test_update_to_higher_frequency_moves_word_earlier(self) -> None:
        """
        Updating an existing word to a higher frequency must move it earlier
        in the bucket so it appears before lower-frequency words.

        Before: "help" (freq=5) < "hello" (freq=10) - hello ranks first
        After update "help" to freq=100: "help" must rank before "hello"
        """
        p = FrequencyPredictor({"hello": 10, "help": 5})
        results_before = [s.suggestion.value for s in p.predict(CompletionContext("hel"))]
        assert results_before[0] == "hello", "Pre-condition: hello ranks first (higher frequency)"

        p.add_word("help", 100)  # now help outranks hello

        results_after = [s.suggestion.value for s in p.predict(CompletionContext("hel"))]
        assert results_after[0] == "help", (
            "After boosting 'help' to freq=100, it must rank before 'hello' (freq=10)"
        )

    def test_update_existing_word_does_not_duplicate_it(self) -> None:
        """Updating an existing word must not create a duplicate entry."""
        p = FrequencyPredictor({"hello": 10, "help": 5})
        p.add_word("hello", 50)
        results = [s.suggestion.value for s in p.predict(CompletionContext("he"))]
        assert results.count("hello") == 1, "hello must appear exactly once after update"

    def test_update_exceeding_max_freq_updates_log_normaliser(self) -> None:
        """
        When add_word introduces a word with frequency > current max_freq,
        the log normaliser must update so all scores remain in (0, 1].

        If _log_max is not updated, the new high-frequency word would score > 1.0,
        violating the invariant that all scores are in (0, 1].
        """
        p = FrequencyPredictor({"hello": 10, "help": 5})
        old_log_max = p._log_max

        p.add_word("the", 100_000)  # much higher than current max (10)

        assert p._log_max > old_log_max, "log normaliser must update when max_freq increases"
        # All scores must remain in (0, 1]
        for s in p.predict(CompletionContext("t")):
            assert 0.0 < s.score <= 1.0, (
                f"Score {s.score} for '{s.suggestion.value}' out of (0,1] after max_freq update"
            )
        for s in p.predict(CompletionContext("hel")):
            assert 0.0 < s.score <= 1.0, (
                f"Score {s.score} for '{s.suggestion.value}' out of (0,1] after max_freq update"
            )

    def test_zero_frequency_in_constructor_skipped(self) -> None:
        """
        Words with frequency=0 in the constructor vocab must not appear in
        the index or in suggestions.

        This covers the `continue` branch in FrequencyPredictor.__init__
        that skips zero-frequency entries at build time.
        """
        p = FrequencyPredictor({"hello": 10, "ghost": 0, "help": 5})
        results = {s.suggestion.value for s in p.predict(CompletionContext("gh"))}
        assert "ghost" not in results, (
            "Zero-frequency word 'ghost' must be excluded from the index at build time"
        )

class TestFrequencyPredictorDefault:
    def test_default_max_results_is_100(self) -> None:
        """max_results must be 100, not 20. With 20, words ranked 21-100 in
        frequency were silently excluded - a correctness regression, not a
        performance trade-off."""
        assert _DEFAULT_MAX_RESULTS == 100, (
            "Default max_results must be 100 to prevent silent truncation "
            "of words that rank 21-100 in frequency for their prefix bucket"
        )

    def test_words_beyond_old_limit_20_are_returned(self) -> None:
        """'hello' ranks 22nd in frequency among 'he' words.
        With old default of 20 it was silently excluded."""
        from aac.data import load_english_frequencies
        freq = load_english_frequencies()
        p = FrequencyPredictor(freq)  # default max_results=100
        results = {s.suggestion.value for s in p.predict(CompletionContext("he"))}
        assert "hello" in results, (
            "'hello' ranks 22nd in frequency for prefix 'he'. "
            "With max_results=100 it must be included."
        )
