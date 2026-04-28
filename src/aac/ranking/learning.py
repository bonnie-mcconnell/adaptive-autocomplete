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

    Performance:
        ``rank()`` and ``explain()`` share a single ``counts_for_prefix()``
        call per prefix.  When the engine calls both in sequence - as
        ``explain()`` does - the history lookup runs once, not twice.
        The cache is keyed by prefix and invalidated on every ``rank()`` call.
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
        """Mark cache as stale.  Called at the start of rank() so explain() reuses
        the counts fetched during that rank() call, but the next rank() call always
        re-fetches in case history was updated between calls."""
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

        # Reuse cache from rank() if prefix matches.  Do NOT call self.rank()
        # here - that would invalidate the cache, defeating the optimisation,
        # and re-sort the same suggestions a second time.
        counts = self._counts(prefix)

        # Produce explanations in the same order rank() would produce them.
        # We replicate the sort key rather than calling rank() to avoid the
        # cache invalidation and the extra allocation of a fully boosted list.
        def _final_score(s: ScoredSuggestion) -> float:
            count = counts.get(s.suggestion.value, 0)
            boost = self._compute_history_boost(
                count=count,
                base_score=pre_boost_scores[s.suggestion.value],
            )
            return pre_boost_scores[s.suggestion.value] + boost

        ordered = sorted(
            enumerate(suggestions),
            key=lambda t: (-_final_score(t[1]), t[0]),
        )

        explanations: list[RankingExplanation] = []
        for _, s in ordered:
            count = counts.get(s.suggestion.value, 0)
            base_score = pre_boost_scores[s.suggestion.value]
            boost = self._compute_history_boost(
                count=count,
                base_score=base_score,
            )
            # LearningRanker is a boost ranker: it contributes history signal,
            # not a base score. Emitting base_score=0.0 ensures that when this
            # explanation is merged with ScoreRanker's explanation, the merged
            # base_score equals only ScoreRanker's value (the true pre-ranking
            # score) - not a doubled sum of both rankers' claims.
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
