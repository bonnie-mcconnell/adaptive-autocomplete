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
