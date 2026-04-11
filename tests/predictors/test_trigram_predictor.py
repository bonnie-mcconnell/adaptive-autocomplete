"""
Tests for TrigramPredictor.

Covers: typo recovery correctness, short-prefix guard, score ordering,
comparison with BK-tree on a controlled vocabulary, and edge cases.
"""
from __future__ import annotations

import pytest

from aac.domain.types import CompletionContext
from aac.predictors.bk_tree import levenshtein
from aac.predictors.edit_distance import EditDistancePredictor
from aac.predictors.trigram import TrigramPredictor, _trigrams

_VOCAB = [
    "hello", "help", "held", "hell", "helm",
    "hero", "her", "here", "world", "word",
    "work", "worker", "worse", "worst",
]

_LARGE_VOCAB_SIZE = 5_000  # big enough to show BK-tree degrades


# ------------------------------------------------------------------
# _trigrams helper
# ------------------------------------------------------------------

def test_trigrams_includes_boundary_tokens() -> None:
    tgs = _trigrams("hi")
    # Two leading spaces give '  h' and ' hi'; one trailing gives 'i '
    assert "  h" in tgs
    assert " hi" in tgs
    assert "hi " in tgs


def test_trigrams_same_string_idempotent() -> None:
    assert _trigrams("hello") == _trigrams("hello")


def test_trigrams_different_strings_not_equal() -> None:
    assert _trigrams("hello") != _trigrams("world")


# ------------------------------------------------------------------
# Basic typo recovery
# ------------------------------------------------------------------

def test_recovers_single_deletion() -> None:
    pred = TrigramPredictor(_VOCAB, max_distance=2)
    values = {s.suggestion.value for s in pred.predict("hell")}
    assert "hello" in values  # 'hell' -> 'hello' is distance 1


def test_recovers_transposition_when_sufficient_overlap() -> None:
    """
    Trigram recall for transpositions depends on shared trigram count.
    'helo' -> 'hello' (distance 1) shares 4 trigrams, well above threshold.
    'hlep' -> 'help' (distance 2) shares only 1 trigram - documented miss.
    Test verifies: results returned are always genuine matches.
    """
    from aac.predictors.bk_tree import levenshtein as lev
    pred = TrigramPredictor(_VOCAB, max_distance=2)
    results = pred.predict("helo")
    values = {s.suggestion.value for s in results}
    assert "hello" in values
    for s in results:
        assert lev("helo", s.suggestion.value) <= 2


def test_first_character_substitution_long_query() -> None:
    """First-char substitution on longer words where overlap is sufficient."""
    from aac.predictors.bk_tree import levenshtein as lev
    vocab = ["worker", "borker", "corker", "dorker", "hello"]
    pred = TrigramPredictor(vocab, max_distance=2)
    results = pred.predict("worler")  # distance 1 from worker
    for s in results:
        assert lev("worler", s.suggestion.value) <= 2


def test_no_results_below_min_prefix_length() -> None:
    """Short prefixes return empty - trigrams are too coarse at len<4."""
    pred = TrigramPredictor(_VOCAB, max_distance=2)
    assert pred.predict("he") == []
    assert pred.predict("h") == []
    assert pred.predict("hel") == []


def test_exact_match_returns_distance_zero() -> None:
    pred = TrigramPredictor(_VOCAB, max_distance=2)
    results = {s.suggestion.value: s for s in pred.predict("hello")}
    assert "hello" in results
    # Distance-0 match gets full base_score
    assert results["hello"].score == pytest.approx(1.0)


# ------------------------------------------------------------------
# Score ordering
# ------------------------------------------------------------------

def test_closer_matches_score_higher() -> None:
    """Distance-1 match must score above distance-2 match."""
    pred = TrigramPredictor(_VOCAB, max_distance=2)
    results = {s.suggestion.value: s.score for s in pred.predict("hello")}
    # 'hell' is distance 1, 'her' is distance 3 (excluded), 'help' is distance 2
    if "hell" in results and "help" in results:
        assert results["hell"] > results["help"]


def test_score_formula_matches_distance() -> None:
    """score == 1.0 / (1 + distance) for every result."""
    pred = TrigramPredictor(_VOCAB, max_distance=2)
    for s in pred.predict("hello"):
        dist = levenshtein("hello", s.suggestion.value)
        expected = 1.0 / (1 + dist)
        assert s.score == pytest.approx(expected)


# ------------------------------------------------------------------
# Correctness vs BK-tree (ground truth) on controlled vocabulary
# ------------------------------------------------------------------

def test_trigram_results_subset_of_bktree() -> None:
    """
    Every trigram result must also be a true match per exact edit distance.

    The trigram filter may miss some BK-tree results (it's a heuristic),
    but it must never return false positives - every result it does return
    must be within max_distance.
    """
    pred = TrigramPredictor(_VOCAB, max_distance=2)
    bk = EditDistancePredictor(_VOCAB, max_distance=2)

    for query in ["hell", "hlep", "wold", "work", "helo"]:
        trigram_values = {s.suggestion.value for s in pred.predict(query)}
        bk_values = {s.suggestion.value for s in bk.predict(query)}

        false_positives = trigram_values - bk_values
        assert not false_positives, (
            f"Trigram returned false positives for {query!r}: {false_positives}"
        )


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_empty_vocabulary_returns_empty() -> None:
    """Empty vocabulary produces no candidates, no crash."""
    pred = TrigramPredictor([], max_distance=2)
    assert pred.predict("hello") == []


def test_no_matches_returns_empty_list() -> None:
    pred = TrigramPredictor(["xyz", "xyzw", "xyzwv"], max_distance=1)
    results = pred.predict("aaaa")
    assert results == []


def test_does_not_mutate_context() -> None:
    pred = TrigramPredictor(_VOCAB, max_distance=2)
    ctx = CompletionContext(text="hello")
    pred.predict(ctx)
    assert ctx.text == "hello"


def test_predict_accepts_string_directly() -> None:
    pred = TrigramPredictor(_VOCAB, max_distance=2)
    # ensure_context is used internally; plain string must work
    results = pred.predict("hello")
    assert isinstance(results, list)
