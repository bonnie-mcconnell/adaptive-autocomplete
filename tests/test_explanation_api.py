"""
Tests for RankingExplanation helper methods.

These are used internally by rankers and exposed for testing/CLI output.
Covers: construction validation, to_dict, merge error path,
from_predictor, apply_history_boost, short_summary.
"""
from __future__ import annotations

import pytest

from aac.ranking.explanation import RankingExplanation


def _make(value: str = "hello", base: float = 10.0, boost: float = 0.0) -> RankingExplanation:
    return RankingExplanation(
        value=value,
        base_score=base,
        history_boost=boost,
        final_score=base + boost,
        source="score",
    )


def test_construction_rejects_inconsistent_final_score() -> None:
    with pytest.raises(ValueError, match="final_score"):
        RankingExplanation(
            value="hello",
            base_score=10.0,
            history_boost=2.0,
            final_score=999.0,  # wrong
            source="score",
        )


def test_to_dict_returns_serialisable_shape() -> None:
    exp = _make("hello", base=5.0, boost=2.0)
    d = exp.to_dict()
    assert d["value"] == "hello"
    assert d["base_score"] == 5.0
    assert d["history_boost"] == 2.0
    assert d["final_score"] == 7.0


def test_merge_rejects_different_values() -> None:
    a = _make("hello")
    b = _make("world")
    with pytest.raises(ValueError, match="different values"):
        a.merge(b)


def test_merge_adds_scores() -> None:
    a = _make("hello", base=5.0, boost=1.0)
    b = RankingExplanation(
        value="hello",
        base_score=0.0,
        history_boost=3.0,
        final_score=3.0,
        source="decay",
    )
    merged = a.merge(b)
    assert merged.base_score == 5.0
    assert merged.history_boost == 4.0
    assert merged.final_score == 9.0


def test_from_predictor_factory() -> None:
    exp = RankingExplanation.from_predictor(value="hello", score=8.5, source="frequency")
    assert exp.value == "hello"
    assert exp.base_score == 8.5
    assert exp.history_boost == 0.0
    assert exp.final_score == 8.5
    assert exp.source == "frequency"


def test_apply_history_boost() -> None:
    base = _make("hello", base=10.0, boost=0.0)
    boosted = base.apply_history_boost(boost=3.0, source="learning")
    assert boosted.history_boost == 3.0
    assert boosted.final_score == 13.0
    assert boosted.base_score == 10.0


def test_short_summary_contains_value_and_scores() -> None:
    exp = _make("hello", base=5.0, boost=2.0)
    summary = exp.short_summary()
    assert "hello" in summary
    assert "5.00" in summary
    assert "2.00" in summary
    assert "7.00" in summary


def test_repr_is_concise_and_contains_key_fields() -> None:
    """__repr__ produces a readable short form, not raw dataclass output."""
    exp = _make("hello", base=1.4063, boost=1.5)
    r = repr(exp)
    assert "hello" in r
    assert "1.4063" in r
    assert "+1.5000" in r
    assert "2.9063" in r
    # Must NOT contain the verbose internal fields
    assert "base_components" not in r
    assert "history_components" not in r
    assert "source" not in r
