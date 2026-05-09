"""aac record subcommand: record a user selection so the engine learns from it."""
from __future__ import annotations

from aac.engine.engine import AutocompleteEngine
from aac.storage.json_store import JsonHistoryStore


def run(
    *,
    engine: AutocompleteEngine,
    store: JsonHistoryStore,
    text: str,
    value: str,
) -> None:
    engine.record_selection(text, value)
    store.save(engine.history)
    print(f"Recorded selection '{value}' for input '{text}'")
