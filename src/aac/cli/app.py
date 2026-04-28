from __future__ import annotations

from aac.domain.history import History
from aac.engine.engine import AutocompleteEngine
from aac.presets import get_preset


def build_engine(
    *,
    preset: str,
    history: History,
    vocabulary: dict[str, int] | None = None,
) -> AutocompleteEngine:
    """
    Construct an AutocompleteEngine from a named preset.

    The CLI/application layer owns persistence and hydration.
    Presets define structure only.

    Args:
        preset:     Preset name (e.g. 'production', 'default').
        history:    Pre-loaded History instance (from JsonHistoryStore or fresh).
        vocabulary: Optional custom vocabulary dict {word: frequency}.
                    If None, uses the preset's default (bundled 48k English vocab).
    """
    preset_def = get_preset(preset)
    return preset_def.build(history, vocabulary)
