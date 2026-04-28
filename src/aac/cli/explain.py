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

    Scores are shown both as raw values and as a percentage of the
    top-ranked suggestion's score, so the relative contribution of
    each signal is readable at a glance.

    Column meanings:
        score   - final ranked score (base + ranker boosts)
        base    - aggregated predictor score before ranking
                  (sum of all predictor contributions: frequency,
                  history predictor, typo correction, etc.)
        boost   - total ranker boost applied on top of the base score
                  (from LearningRanker, DecayRanker, etc.)

    Use ``engine.explain_as_dicts()`` for a per-predictor breakdown
    of the base score (``base_components``) and per-ranker breakdown
    of the boost (``history_components``).

    This is a presentation-layer adapter:
    - No scoring logic lives here
    - Output is intentionally human-readable
    """
    explanations = engine.explain(text)[:limit]

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
