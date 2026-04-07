from __future__ import annotations

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

    Emits candidates previously selected by the user.
    Score reflects raw usage frequency.
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
        results: list[ScoredSuggestion] = []

        for value, count in counts.items():
            score = float(count)
            confidence = count / max_count if max_count > 0 else 0.0

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