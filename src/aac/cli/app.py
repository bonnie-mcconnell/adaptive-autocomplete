"""Shared engine factory used by all CLI subcommands."""
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
    """Build an engine from a preset name, history, and optional custom vocabulary."""
    preset_def = get_preset(preset)
    return preset_def.build(history, vocabulary)
