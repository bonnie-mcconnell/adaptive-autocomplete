from __future__ import annotations

import pytest

from aac.engine import AutocompleteEngine
from aac.presets import PRESETS, available_presets, create_engine


def test_presets_registry_is_non_empty() -> None:
    assert PRESETS, "Preset registry should not be empty"


def test_available_presets_sorted() -> None:
    names = available_presets()
    assert names == sorted(names)


@pytest.mark.parametrize("preset_name", available_presets())
def test_create_engine_returns_engine(preset_name: str) -> None:
    engine = create_engine(preset_name)
    assert isinstance(engine, AutocompleteEngine)


def test_create_engine_invalid_preset() -> None:
    with pytest.raises(ValueError):
        create_engine("does_not_exist")


@pytest.mark.parametrize("preset_name", available_presets())
def test_engine_can_suggest(preset_name: str) -> None:
    engine = create_engine(preset_name)
    suggestions = engine.suggest("he")
    assert isinstance(suggestions, list)


def test_engine_describe_shape() -> None:
    engine = create_engine(available_presets()[0])
    info = engine.describe()

    assert "predictors" in info
    assert "rankers" in info
    assert "history_entries" in info

    assert isinstance(info["predictors"], list)
    assert isinstance(info["rankers"], list)
