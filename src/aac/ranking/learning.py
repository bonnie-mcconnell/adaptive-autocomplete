from __future__ import annotations

from collections.abc import Sequence

from aac.domain.history import History
from aac.domain.types import ScoredSuggestion
from aac.ranking.base import Ranker
from aac.ranking.contracts import LearnsFromHistory
from aac.ranking.explanation import RankingExplanation


class LearningRanker(Ranker, LearnsFromHistory):
    """
    Ranker that adapts suggestion ordering based on user selection history.

    Applies a linear boost to suggestions that the user has selected before,
    causing them to rise in the ranked output over time.

    When to use:
        Use LearningRanker when your predictor stack does not already
        incorporate history as a prediction signal. For example, pairing
        LearningRanker with FrequencyPredictor alone gives you a clean
        separation: frequency drives candidate generation, history drives
        re-ranking.

        If your engine includes HistoryPredictor, that predictor already
        emits history-scored candidates at the prediction layer. Adding
        LearningRanker on top would count history twice - once in prediction
        scores and again in the ranking boost. In that configuration, omit
        LearningRanker and rely on HistoryPredictor's weight instead.

    Learning model:
        boost = min(count * boost_param, dominance_ratio * base_score)
        final_score = base_score + boost

    Invariants:
    - No history signal => original order preserved
    - Learning is additive (never suppresses candidates)
    - Learning influence is bounded by dominance_ratio
    - Deterministic and stable
    - Does not mutate inputs or history
    """

    def __init__(
        self,
        history: History,
        *,
        boost: float = 1.0,
        dominance_ratio: float = 1.0,
    ) -> None:
        if boost < 0.0:
            raise ValueError("boost must be non-negative")

        if dominance_ratio < 0.0:
            raise ValueError("dominance_ratio must be non-negative")

        # Required by LearnsFromHistory
        self.history: History = history

        self._boost = boost
        self._dominance_ratio = dominance_ratio

    # --- learning internals ---

    def _compute_history_boost(self, *, count: int, base_score: float) -> float:
        """
        Compute a bounded linear learning boost.

        raw_boost = count * boost
        raw_boost <= dominance_ratio * base_score
        """
        if count <= 0:
            return 0.0

        boost = count * self._boost

        if base_score > 0.0:
            boost = min(boost, self._dominance_ratio * base_score)

        return boost

    def _compute_adjusted_score(
        self,
        *,
        value: str,
        base_score: float,
        counts: dict[str, int],
    ) -> float:
        count = counts.get(value, 0)
        boost = self._compute_history_boost(
            count=count,
            base_score=base_score,
        )
        return base_score + boost

    # --- ranking ---

    def rank(
        self,
        prefix: str,
        suggestions: Sequence[ScoredSuggestion],
    ) -> list[ScoredSuggestion]:
        if not suggestions:
            return []

        counts = self.history.counts_for_prefix(prefix)

        # Invariant: no history signal => preserve original order
        if not counts:
            return list(suggestions)

        scored: list[tuple[float, int, ScoredSuggestion]] = []

        for index, suggestion in enumerate(suggestions):
            final_score = self._compute_adjusted_score(
                value=suggestion.suggestion.value,
                base_score=suggestion.score,
                counts=counts,
            )
            # Return a new ScoredSuggestion with the boosted score so that
            # downstream rankers and predict_scored() see the updated value.
            boosted = ScoredSuggestion(
                suggestion=suggestion.suggestion,
                score=final_score,
                explanation=suggestion.explanation,
                trace=suggestion.trace + (
                    f"LearningRanker boost={final_score - suggestion.score:.4f}",
                ),
            )
            scored.append((final_score, index, boosted))

        # Stable: score desc, original index as tiebreaker
        scored.sort(key=lambda t: (-t[0], t[1]))

        return [suggestion for _, _, suggestion in scored]

    # --- explanation ---

    def explain(
        self,
        prefix: str,
        suggestions: Sequence[ScoredSuggestion],
    ) -> list[RankingExplanation]:
        # Build a lookup of pre-ranking (pre-boost) scores keyed by value.
        # We must compute the boost against the pre-boost base_score, not the
        # post-boost score from rank(). Using the post-boost score in
        # _compute_history_boost would produce a different (larger) dominance
        # cap than the one used during rank(), causing the explanation to
        # be arithmetically inconsistent with the actual ranking decision.
        pre_boost_scores = {s.suggestion.value: s.score for s in suggestions}
        counts = self.history.counts_for_prefix(prefix)

        # Produce explanations in rank() order so callers can rely on ordering.
        ranked = self.rank(prefix, suggestions)
        explanations: list[RankingExplanation] = []

        for s in ranked:
            count = counts.get(s.suggestion.value, 0)
            base_score = pre_boost_scores[s.suggestion.value]
            boost = self._compute_history_boost(
                count=count,
                base_score=base_score,
            )

            explanations.append(
                RankingExplanation(
                    value=s.suggestion.value,
                    base_score=base_score,
                    history_boost=boost,
                    final_score=base_score + boost,
                    source="learning",
                )
            )

        return explanations

    def explain_as_dicts(
        self,
        prefix: str,
        suggestions: Sequence[ScoredSuggestion],
    ) -> list[dict[str, float | str]]:
        """
        Export ranking explanations in a JSON-safe schema.
        """
        return [
            {
                "value": e.value,
                "base_score": e.base_score,
                "history_boost": e.history_boost,
                "final_score": e.final_score,
            }
            for e in self.explain(prefix, suggestions)
        ]