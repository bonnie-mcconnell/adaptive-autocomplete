from __future__ import annotations

from aac.domain.history import History
from aac.presets import get_preset


def test_robust_recovers_simple_typo() -> None:
    history = History()
    engine = get_preset("robust").build(history, None)

    suggestions = engine.suggest("helo")
    values = [s.value for s in suggestions]

    assert "hello" in values


def test_robust_does_not_pollute_exact_prefix() -> None:
    history = History()
    engine = get_preset("robust").build(history, None)

    exact = engine.suggest("he")
    values = [s.value for s in exact]

    assert "hello" in values