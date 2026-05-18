"""EditDistancePredictor: BK-tree based fuzzy matching. Falls back to linear scan at large vocab."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from aac.domain.types import (
    CompletionContext,
    Predictor,
    PredictorExplanation,
    ScoredSuggestion,
    Suggestion,
    ensure_context,
)
from aac.predictors._scoring import build_freq_scores, distance_score, edit_confidence
from aac.predictors.bk_tree import BKTree, levenshtein

__all__ = ["EditDistancePredictor", "levenshtein"]


class EditDistancePredictor(Predictor):
    """BK-tree fuzzy matching. Degrades toward O(n) at large max_distance; use SymSpellPredictor for production."""

    name = "edit_distance"

    def __init__(
        self,
        vocabulary: Iterable[str],
        *,
        max_distance: int = 2,
        base_score: float = 1.0,
        frequencies: Mapping[str, int] | None = None,
    ) -> None:
        self._max_distance = max_distance
        self._base_score = base_score
        # BKTree filters empty strings internally; pass vocabulary directly.
        self._tree = BKTree(vocabulary)
        # Pre-computed log-normalised frequency scores.
        # Formula and FREQ_WEIGHT rationale: see aac.predictors._scoring
        self._freq_scores: dict[str, float] = build_freq_scores(frequencies)

    def predict(self, ctx: CompletionContext | str) -> list[ScoredSuggestion]:
        ctx = ensure_context(ctx)
        prefix = ctx.prefix()

        if not prefix:
            return []

        results: list[ScoredSuggestion] = []

        for word, dist in self._tree.search(prefix, max_distance=self._max_distance):
            freq_score = self._freq_scores.get(word, 0.0)
            score = distance_score(self._base_score, dist, freq_score)
            confidence = edit_confidence(dist, self._max_distance)

            results.append(
                ScoredSuggestion(
                    suggestion=Suggestion(value=word),
                    score=score,
                    explanation=PredictorExplanation(
                        value=word,
                        score=score,
                        source=self.name,
                        confidence=confidence,
                    ),
                )
            )

        return results
