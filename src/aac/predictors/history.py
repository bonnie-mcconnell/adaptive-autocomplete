"""HistoryPredictor: scores previously-selected values by selection frequency for the current prefix."""
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
    """Suggests previously-selected values, scored by log-normalised selection count."""

    name = "history"

    def __init__(self, history: History) -> None:
        self._history = history

    @property
    def history(self) -> History:
        return self._history

    @history.setter
    def history(self, value: History) -> None:
        self._history = value

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
