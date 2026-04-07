"""
Tests for the BK-tree approximate string matching index.

Correctness property: BKTree.search(query, max_distance=t) must return
exactly the same set of words as a brute-force linear scan over the same
vocabulary using the same distance function.
"""
from __future__ import annotations

from aac.predictors.bk_tree import BKTree, _levenshtein


def _linear_search(
    query: str,
    words: list[str],
    max_distance: int,
) -> set[str]:
    """Brute-force reference implementation for correctness comparison."""
    return {w for w in words if _levenshtein(query, w) <= max_distance}


def _bk_search(tree: BKTree, query: str, max_distance: int) -> set[str]:
    return {w for w, _ in tree.search(query, max_distance=max_distance)}


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

def test_empty_vocabulary() -> None:
    tree = BKTree([])
    assert not tree
    assert len(tree) == 0
    assert list(tree.search("hello", max_distance=1)) == []


def test_single_word() -> None:
    tree = BKTree(["hello"])
    assert len(tree) == 1
    assert bool(tree)


def test_duplicates_are_deduplicated() -> None:
    tree = BKTree(["hello", "hello", "hello"])
    assert len(tree) == 1


def test_empty_strings_ignored() -> None:
    tree = BKTree(["hello", "", "world"])
    assert len(tree) == 2


# ------------------------------------------------------------------
# Correctness: BK-tree must match brute-force linear scan exactly
# ------------------------------------------------------------------

VOCAB = [
    "hello", "help", "helium", "hero", "hex", "heap",
    "world", "word", "work", "worry",
    "the", "there", "then", "them",
    "cat", "car", "card", "cart", "care",
]


def test_exact_match_found() -> None:
    tree = BKTree(VOCAB)
    results = _bk_search(tree, "hello", max_distance=0)
    assert results == {"hello"}


def test_exact_match_only_at_distance_zero() -> None:
    tree = BKTree(VOCAB)
    results = _bk_search(tree, "hello", max_distance=0)
    assert "help" not in results


def test_matches_brute_force_distance_1() -> None:
    tree = BKTree(VOCAB)
    for query in ["helo", "hlp", "wrold", "th", "cat"]:
        bk = _bk_search(tree, query, max_distance=1)
        linear = _linear_search(query, VOCAB, max_distance=1)
        assert bk == linear, f"Mismatch for {query!r}: BK={bk} linear={linear}"


def test_matches_brute_force_distance_2() -> None:
    tree = BKTree(VOCAB)
    for query in ["helo", "hlep", "wrold", "xyz", "teh", "wello"]:
        bk = _bk_search(tree, query, max_distance=2)
        linear = _linear_search(query, VOCAB, max_distance=2)
        assert bk == linear, f"Mismatch for {query!r}: BK={bk} linear={linear}"


def test_no_match_returns_empty() -> None:
    tree = BKTree(VOCAB)
    # 'zzzzz' is far from every word in the vocab
    results = _bk_search(tree, "zzzzz", max_distance=1)
    assert results == set()


def test_first_character_typo_detected() -> None:
    # BK-tree must catch first-character errors — unlike a first-char index.
    tree = BKTree(["hello", "help", "world"])
    results = _bk_search(tree, "wello", max_distance=2)
    # levenshtein('wello', 'hello') = 1 (w->h), so 'hello' must appear
    assert "hello" in results


def test_distance_values_are_correct() -> None:
    tree = BKTree(["hello", "help"])
    results = {w: d for w, d in tree.search("helo", max_distance=2)}
    assert results["hello"] == _levenshtein("helo", "hello")
    assert results["help"] == _levenshtein("helo", "help")


# ------------------------------------------------------------------
# Triangle inequality pruning (structural)
# ------------------------------------------------------------------

def test_search_empty_tree() -> None:
    tree = BKTree([])
    assert list(tree.search("anything", max_distance=2)) == []


def test_distance_zero_is_exact_match_only() -> None:
    tree = BKTree(["hello", "help"])
    results = _bk_search(tree, "hello", max_distance=0)
    assert results == {"hello"}


def test_large_max_distance_returns_all() -> None:
    words = ["hello", "world"]
    tree = BKTree(words)
    # max_distance=100 should return everything
    results = _bk_search(tree, "hello", max_distance=100)
    assert results == set(words)