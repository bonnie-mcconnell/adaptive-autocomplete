"""aac suggest subcommand: print ranked suggestions to stdout, one per line."""
from __future__ import annotations

from aac.engine.engine import AutocompleteEngine


def run(*, engine: AutocompleteEngine, text: str, limit: int) -> None:
    suggestions = engine.suggest(text, limit=limit)

    if not suggestions:
        print("(no suggestions available)")
        return

    for suggestion in suggestions:
        print(suggestion)
