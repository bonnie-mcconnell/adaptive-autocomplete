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