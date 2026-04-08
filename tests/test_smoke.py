"""
Smoke test: verify the engine initialises and produces output end-to-end.

This is not a unit test. It exists to catch import errors, missing data
files, and wiring failures that unit tests would miss because they stub
out dependencies. If this test fails, nothing else in the suite is reliable.
"""
from __future__ import annotations

from aac.presets import create_engine


def test_default_engine_produces_suggestions() -> None:
    engine = create_engine("default")
    results = engine.suggest("he")
    assert len(results) > 0, "Expected at least one suggestion for prefix 'he'"


def test_robust_engine_handles_typo() -> None:
    engine = create_engine("robust")
    results = engine.suggest("helo")
    values = [s.value for s in results]
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
