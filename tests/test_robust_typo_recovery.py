from __future__ import annotations

from aac.domain.history import History
from aac.presets import get_preset

# Small vocabulary keeps BK-tree construction and search fast.
# The BK-tree's correctness against the full vocabulary is verified
# separately in tests/predictors/; this file tests preset wiring only.
_SMALL_VOCAB = {
    "hello": 100, "help": 80, "hero": 50,
    "her": 200, "here": 120, "heap": 40,
    "world": 300, "word": 150,
}


def test_robust_recovers_simple_typo() -> None:
    engine = get_preset("robust").build(History(), _SMALL_VOCAB)
    values = engine.suggest("helo")
    assert "hello" in values, f"Expected 'hello' in typo recovery results, got {values}"


def test_robust_does_not_pollute_exact_prefix() -> None:
    engine = get_preset("robust").build(History(), _SMALL_VOCAB)
    values = engine.suggest("he")
    assert "hello" in values


def test_robust_recovers_first_character_error() -> None:
    """BK-tree catches errors on the first character; trie/prefix indexes cannot."""
    engine = get_preset("robust").build(History(), _SMALL_VOCAB)
    values = engine.suggest("wello")
    # 'wello' is distance 1 from 'hello' (w→h substitution)
    assert "hello" in values, f"Expected first-char typo recovery, got {values}"
