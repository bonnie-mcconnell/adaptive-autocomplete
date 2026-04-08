from __future__ import annotations

from aac.domain.history import History
from aac.presets import get_preset


def test_stateless_does_not_learn() -> None:
    """Stateless preset must produce identical output regardless of recorded history."""
    history = History()
    # Pass None as second arg (vocabulary), PresetBuilder requires both positional args
    engine = get_preset("stateless").build(history, None)

    before = [s.value for s in engine.suggest("he")]

    # Record many selections, a learning engine would change its output.
    for _ in range(10):
        history.record("he", "hero")

    after = [s.value for s in engine.suggest("he")]

    assert before == after, (
        "Stateless engine must not change output after history.record(). "
        f"Before: {before}, After: {after}"
    )