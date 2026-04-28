from __future__ import annotations

import math

from aac.domain.history import History
from aac.domain.types import (
    CompletionContext,
    Predictor,
    PredictorExplanation,
    ScoredSuggestion,
    Suggestion,
    ensure_context,
)


class HistoryPredictor(Predictor):
    """
    Recall-based predictor driven by user selection history.

    Emits candidates previously selected by the user for the current prefix.

    Score model:
        Scores are log-normalised to (0, 1] relative to the most-selected
        value for this prefix:

            score = log(1 + count) / log(1 + max_count_for_prefix)

        This keeps HistoryPredictor's scores in the same (0, 1] space as
        FrequencyPredictor, so that weights in WeightedPredictor are
        meaningful. A weight of 1.5 on HistoryPredictor means 1.5× the
        frequency signal, not 1.5 raw selection counts.

    Confidence reflects dominance among historical matches.
    """

    name = "history"

    def __init__(self, history: History) -> None:
        self._history = history

    def predict(self, ctx: CompletionContext | str) -> list[ScoredSuggestion]:
        ctx = ensure_context(ctx)
        prefix = ctx.prefix()

        if not prefix:
            return []

        counts = self._history.counts_for_prefix(prefix)
        if not counts:
            return []

        max_count = max(counts.values())
        log_max = math.log1p(max_count)
        results: list[ScoredSuggestion] = []

        for value, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            score = math.log1p(count) / log_max if log_max > 0 else 0.0
            confidence = score  # log-normalised score doubles as confidence

            results.append(
                ScoredSuggestion(
                    suggestion=Suggestion(value=value),
                    score=score,
                    explanation=PredictorExplanation(
                        value=value,
                        score=score,
                        source=self.name,
                        confidence=confidence,
                    ),
                )
            )

        return results
