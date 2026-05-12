from __future__ import annotations

import pytest

from aac.engine import AutocompleteEngine
from aac.presets import PRESETS, available_presets, create_engine, get_preset

# Small controlled vocabulary used for all preset smoke tests.
# Avoids loading the 48k word list (slow FrequencyPredictor construction)
# and the BK-tree over that vocabulary (60ms per call).
_SMALL_VOCAB = {
    "hello": 100, "help": 80, "hero": 50,
    "her": 200, "here": 120, "heat": 40,
}


def test_presets_registry_is_non_empty() -> None:
    assert PRESETS, "Preset registry should not be empty"


def test_available_presets_sorted() -> None:
    names = available_presets()
    assert names == sorted(names)


@pytest.mark.parametrize("preset_name", available_presets())
def test_create_engine_returns_engine(preset_name: str) -> None:
    engine = get_preset(preset_name).build(None, _SMALL_VOCAB)
    assert isinstance(engine, AutocompleteEngine)


def test_create_engine_invalid_preset() -> None:
    with pytest.raises(ValueError):
        create_engine("does_not_exist")


@pytest.mark.parametrize("preset_name", available_presets())
def test_engine_can_suggest(preset_name: str) -> None:
    engine = get_preset(preset_name).build(None, _SMALL_VOCAB)
    suggestions = engine.suggest("he")
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0
    assert all(isinstance(s, str) for s in suggestions), (
        f"suggest() must return List[str], got {[type(s) for s in suggestions[:3]]}"
    )


def test_engine_describe_shape() -> None:
    preset_name = available_presets()[0]
    engine = get_preset(preset_name).build(None, _SMALL_VOCAB)
    info = engine.describe()

    assert "predictors" in info
    assert "rankers" in info
    assert "history_entries" in info

    assert isinstance(info["predictors"], list)
    assert isinstance(info["rankers"], list)



def test_bktree_preset_is_accessible_directly() -> None:
    """
    The 'bktree' preset is excluded from available_presets() (it degrades to O(n)
    at 48k+ words and is not recommended for production), but must still be
    accessible via create_engine("bktree") for users who need it explicitly
    - e.g. benchmarking BK-tree vs SymSpell performance.

    This test exercises _bktree_engine() with a small vocabulary so the
    O(n) characteristic doesn't make CI slow.
    """
    engine = get_preset("bktree").build(None, _SMALL_VOCAB)
    assert isinstance(engine, AutocompleteEngine)

    results = engine.suggest("hel")
    assert len(results) > 0, "bktree preset must produce suggestions for 'hel'"
    assert "hello" in results or "help" in results or "here" in results
    """describe()['history_entries'] must reflect the true count of recorded selections.

    Validates O(1) access: describe() now uses len(self._history) via
    History.__len__, which reads len(self._entries) directly without
    allocating a tuple copy.
    """
    engine = get_preset("stateless").build(None, _SMALL_VOCAB)

    assert engine.describe()["history_entries"] == 0

    engine.record_selection("pr", "program")
    assert engine.describe()["history_entries"] == 1

    engine.record_selection("pr", "programming")
    engine.record_selection("hel", "hello")
    assert engine.describe()["history_entries"] == 3
