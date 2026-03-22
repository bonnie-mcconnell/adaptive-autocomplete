from __future__ import annotations

from aac.engine.engine import AutocompleteEngine


def run(
    *,
    engine: AutocompleteEngine,
    text: str,
    limit: int,
) -> None:
    """
    Render ranking explanations for a given input.

    This is a presentation-layer adapter:
    - No scoring logic lives here
    - Output is intentionally human-readable
    """
    explanations = engine.explain(text)[:limit]

    if not explanations:
        print("(no explanations available - no suggestions produced)")
        return

    for exp in explanations:
        history = exp.history_boost
        sign = "+" if history > 0 else ""

        print(
            f"{exp.value:12} "
            f"base={exp.base_score:6.2f} "
            f"history={sign}{history:6.2f} "
            f"= {exp.final_score:6.2f} "
            f"[source={exp.source}]"
        )
