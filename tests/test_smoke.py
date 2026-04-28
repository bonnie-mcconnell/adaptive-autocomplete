"""
Smoke tests: verify the engine initialises and produces output end-to-end.

These are not unit tests. They exist to catch import errors, missing data
files, and wiring failures that unit tests miss because they stub out
dependencies. If any of these fail, nothing else in the suite is reliable.
"""
from __future__ import annotations

from types import MappingProxyType

from aac.data import load_english_frequencies
from aac.presets import create_engine


def test_load_english_frequencies_returns_mapping() -> None:
    """Bundled vocabulary must load as a non-empty immutable mapping."""
    vocab = load_english_frequencies()
    assert isinstance(vocab, MappingProxyType)
    assert len(vocab) > 0
    # All keys must be lowercase strings, all values positive integers
    for word, freq in vocab.items():
        assert isinstance(word, str)
        assert isinstance(freq, int)
        assert freq > 0


def test_default_engine_produces_suggestions() -> None:
    engine = create_engine("default")
    results = engine.suggest("he")
    assert len(results) > 0, "Expected at least one suggestion for prefix 'he'"


def test_robust_engine_handles_typo() -> None:
    engine = create_engine("robust")
    values = engine.suggest("helo")
    assert "hello" in values, f"Expected 'hello' in typo recovery results, got {values}"


def test_engine_explain_returns_reconciled_scores() -> None:
    engine = create_engine("recency")
    explanations = engine.explain("he")
    assert explanations, "Expected at least one explanation"
    for exp in explanations:
        assert abs(exp.final_score - (exp.base_score + exp.history_boost)) < 1e-9, (
            f"Invariant broken for '{exp.value}': "
            f"{exp.base_score} + {exp.history_boost} != {exp.final_score}"
        )