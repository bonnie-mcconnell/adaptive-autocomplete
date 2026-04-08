from __future__ import annotations

from aac.domain.history import History
from aac.engine.engine import AutocompleteEngine
from aac.presets import get_preset


def build_engine(
    *,
    preset: str,
    history: History,
) -> AutocompleteEngine:
    """
    Construct an AutocompleteEngine from a named preset.

    The CLI/application layer owns persistence and hydration.
    Presets define structure only.
    """
    preset_def = get_preset(preset)
    return preset_def.build(history, None)
