from __future__ import annotations

import pytest

from aac.domain.types import CompletionContext
from aac.predictors.frequency import FrequencyPredictor


@pytest.fixture()
def predictor() -> FrequencyPredictor:
    return FrequencyPredictor({"hello": 10, "help": 5, "helium": 1, "world": 20})


def test_returns_words_matching_prefix(predictor: FrequencyPredictor) -> None:
    results = predictor.predict(CompletionContext("he"))
    values = {r.suggestion.value for r in results}
    assert values == {"hello", "help", "helium"}


def test_scores_reflect_frequency(predictor: FrequencyPredictor) -> None:
    results = predictor.predict(CompletionContext("he"))
    by_value = {r.suggestion.value: r.score for r in results}
    assert by_value["hello"] == 10.0
    assert by_value["help"] == 5.0
    assert by_value["helium"] == 1.0


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


def test_confidence_is_relative_to_max_frequency(predictor: FrequencyPredictor) -> None:
    results = predictor.predict(CompletionContext("he"))
    by_value = {r.suggestion.value: r for r in results}

    # FrequencyPredictor always populates explanation - assert non-None
    # before accessing .confidence so the type checker knows it is safe.
    assert by_value["hello"].explanation is not None
    assert by_value["helium"].explanation is not None

    # Confidence is relative to global max frequency (world=20), not the
    # prefix-scoped max. hello (10/20=0.5), helium (1/20=0.05).
    assert by_value["hello"].explanation.confidence == 10 / 20
    assert by_value["helium"].explanation.confidence == 1 / 20


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
        # If the index contains "hi" as a key, exact-match filtering was done
        # at query time (old behaviour) rather than at build time (correct).
        # The key "hi" should not exist because range(1, len("hi")) = range(1,2) = [1]
        # which only generates prefix "h", not "hi".
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
