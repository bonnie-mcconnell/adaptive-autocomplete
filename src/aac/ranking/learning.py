"""LearningRanker: boosts suggestions with high selection history above their base frequency score."""
from __future__ import annotations

from collections.abc import Sequence

from aac.domain.history import History
from aac.domain.types import ScoredSuggestion
from aac.ranking.base import Ranker
from aac.ranking.contracts import LearnsFromHistory
from aac.ranking.explanation import RankingExplanation


class LearningRanker(Ranker, LearnsFromHistory):
    """
    Boosts suggestions the user has previously selected, causing them to rise
    in the ranked output over time.

        boost = min(count * boost_param, dominance_ratio * base_score)

    If your engine already includes HistoryPredictor, don't add this ranker -
    history would be counted twice. Use it when FrequencyPredictor is the only
    predictor and you want learning at the ranking layer instead.

    rank() and explain() share one counts_for_prefix() call per prefix.
    Cache is invalidated at the start of each rank() call.
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

        # Cache: most recent counts_for_prefix result.
        self._cached_prefix: str | None = None
        self._cached_counts: dict[str, int] = {}
        self._cache_valid: bool = False

    # --- cache ---

    def _counts(self, prefix: str) -> dict[str, int]:
        """Return counts for prefix, reusing cache when prefix is unchanged since last rank()."""
        if prefix == self._cached_prefix and self._cache_valid:
            return self._cached_counts
        counts = self.history.counts_for_prefix(prefix)
        self._cached_prefix = prefix
        self._cached_counts = counts
        self._cache_valid = True
        return counts

    def _invalidate_cache(self) -> None:
        self._cache_valid = False

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

    def _sort_key(
        self,
        s: ScoredSuggestion,
        index: int,
        counts: dict[str, int],
    ) -> tuple[float, int]:
        """Sort key shared by rank() and explain(): (-final_score, original_index)."""
        final_score = self._compute_adjusted_score(
            value=s.suggestion.value,
            base_score=s.score,
            counts=counts,
        )
        return (-final_score, index)

    def ranker_config(self) -> dict[str, float]:
        """Return parameters needed to reconstruct this ranker."""
        return {
            "boost": self._boost,
            "dominance_ratio": self._dominance_ratio,
        }

    # --- ranking ---

    def rank(
        self,
        prefix: str,
        suggestions: Sequence[ScoredSuggestion],
    ) -> list[ScoredSuggestion]:
        if not suggestions:
            return []

        # Invalidate cache so this rank() always fetches fresh history,
        # then store the result so explain() can reuse it without a second fetch.
        self._invalidate_cache()
        counts = self._counts(prefix)

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

        scored.sort(key=lambda t: self._sort_key(t[2], t[1], counts))

        return [suggestion for _, _, suggestion in scored]

    # --- explanation ---

    def explain(
        self,
        prefix: str,
        suggestions: Sequence[ScoredSuggestion],
    ) -> list[RankingExplanation]:
        # Use pre-boost scores so the dominance cap matches what rank() used.
        pre_boost_scores = {s.suggestion.value: s.score for s in suggestions}

        # Reuse the cache populated by rank(); don't call rank() here.
        counts = self._counts(prefix)

        # Sort using _sort_key() so ordering matches rank() exactly.
        ordered = sorted(
            enumerate(suggestions),
            key=lambda t: self._sort_key(t[1], t[0], counts),
        )

        explanations: list[RankingExplanation] = []
        for _, s in ordered:
            count = counts.get(s.suggestion.value, 0)
            base_score = pre_boost_scores[s.suggestion.value]
            boost = self._compute_history_boost(
                count=count,
                base_score=base_score,
            )
            # base_score=0.0: this ranker contributes boost only, not a base score.
            explanations.append(
                RankingExplanation(
                    value=s.suggestion.value,
                    base_score=0.0,
                    history_boost=boost,
                    final_score=boost,
                    source="learning",
                    base_components={},
                    history_components={"learning": boost} if boost > 0 else {},
                )
            )

        return explanations
