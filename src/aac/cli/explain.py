"""aac explain subcommand: show per-suggestion score breakdowns in human-readable format."""
from __future__ import annotations

from aac.engine.engine import AutocompleteEngine


def run(
    *,
    engine: AutocompleteEngine,
    text: str,
    limit: int,
) -> None:
    """Print per-suggestion score breakdowns. score=base+boost; percentages relative to top result."""
    explanations = engine.explain(text)
    if limit is not None:
        explanations = explanations[:limit]

    if not explanations:
        print("(no suggestions for this input)")
        return

    # Normalise against the highest final score so percentages are meaningful
    max_score = max(e.final_score for e in explanations) or 1.0

    for exp in explanations:
        pct = (exp.final_score / max_score) * 100

        print(
            f"{exp.value:14} "
            f"score={exp.final_score:9.2f} ({pct:5.1f}%)  "
            f"base={exp.base_score:9.2f}  "
            f"boost={exp.history_boost:+.2f}"
        )
