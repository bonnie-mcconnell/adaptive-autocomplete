from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from aac.domain.history import History
from aac.domain.types import ScoredSuggestion
from aac.ranking.base import Ranker
from aac.ranking.contracts import LearnsFromHistory
from aac.ranking.explanation import RankingExplanation

# ---------------------------------------------------------------------
# Decay function
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class DecayFunction:
    """
    Time-based exponential decay.

    weight = 0.5 ** (elapsed_seconds / half_life_seconds)
    """
    half_life_seconds: float

    def weight(self, *, now: datetime, event_time: datetime) -> float:
        if event_time.tzinfo is None:
            raise ValueError("History timestamps must be timezone-aware")

        if event_time > now:
            return 1.0

        elapsed = (now - event_time).total_seconds()
        if elapsed <= 0:
            return 1.0

        return float(0.5 ** (elapsed / self.half_life_seconds))


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------

class DecayRanker(Ranker, LearnsFromHistory):
    """
    Ranker that boosts suggestions using recency-weighted history.

    Properties:
    - Deterministic
    - Bounded
    - Composable
    - Fully explainable

    Performance:
        ``rank()`` and ``explain()`` share a single ``_decayed_counts()``
        call per (prefix, now) pair.  When the engine calls both in
        sequence - as ``explain()`` does - the history scan runs once,
        not twice.  The cache is invalidated on every ``rank()`` call so
        a new ``now`` timestamp is always used.
    """

    def __init__(
        self,
        history: History,
        decay: DecayFunction,
        *,
        weight: float = 1.0,
        now: datetime | None = None,
    ) -> None:
        self.history = history
        self._decay = decay
        self._weight = weight
        self._now = now

        # Cache: stores result from the most recent rank() call.
        # explain() reuses it when called with the same prefix immediately after rank().
        # Invalidated at the start of every rank() to ensure fresh history reads.
        self._cached_prefix: str | None = None
        self._cached_now: datetime | None = None
        self._cached_counts: dict[str, float] = {}
        self._cache_valid: bool = False

    def _now_utc(self) -> datetime:
        return self._now if self._now is not None else utcnow()

    def _decayed_counts(self, prefix: str, now: datetime) -> dict[str, float]:
        """
        Return recency-weighted selection counts for prefix.

        Results are cached for the (prefix, now) pair from the most recent
        rank() call.  explain() hits the cache when called in the same
        engine pipeline pass, eliminating a redundant O(k) history scan.
        The cache is invalidated at the start of every rank() call so that
        history updates between rank() calls are always visible.
        """
        if self._cache_valid and prefix == self._cached_prefix and now == self._cached_now:
            return self._cached_counts

        counts: dict[str, float] = defaultdict(float)
        for entry in self.history.entries_for_prefix(prefix):
            counts[entry.value] += self._decay.weight(
                now=now,
                event_time=entry.timestamp,
            )

        result = dict(counts)
        self._cached_prefix = prefix
        self._cached_now = now
        self._cached_counts = result
        self._cache_valid = True
        return result

    def rank(
        self,
        prefix: str,
        suggestions: Sequence[ScoredSuggestion],
    ) -> list[ScoredSuggestion]:
        if not suggestions:
            return []

        # Invalidate so this rank() always re-fetches history, then cache
        # the result so explain() can reuse it without a second scan.
        self._cache_valid = False
        now = self._now_utc()
        decayed = self._decayed_counts(prefix, now)
        if not decayed:
            return list(suggestions)

        scored: list[tuple[float, int, ScoredSuggestion]] = []

        for index, s in enumerate(suggestions):
            boost = decayed.get(s.suggestion.value, 0.0) * self._weight
            final_score = s.score + boost

            # Add a trace entry when boost is non-zero so debug() shows
            # DecayRanker's contribution alongside LearningRanker's.
            new_trace = s.trace
            if boost > 0.0:
                new_trace = s.trace + (
                    f"DecayRanker boost={boost:.4f}",
                )

            scored.append((
                final_score,
                index,  # stable tiebreaker: preserve original order on equal scores
                ScoredSuggestion(
                    suggestion=s.suggestion,
                    score=final_score,
                    explanation=s.explanation,
                    trace=new_trace,
                ),
            ))

        # Stable sort: score desc, original index as tiebreaker.
        # Matches LearningRanker's sort contract so composing both rankers
        # produces deterministic output regardless of insertion order.
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [s for _, _, s in scored]

    def explain(
        self,
        prefix: str,
        suggestions: Sequence[ScoredSuggestion],
    ) -> list[RankingExplanation]:
        # Reuse the cached decayed counts from the rank() call that the engine
        # made immediately before this explain() call.  If the cache misses
        # (e.g. explain() called standalone), recompute.
        now = self._now_utc()
        decayed = self._decayed_counts(prefix, now)

        explanations: list[RankingExplanation] = []

        for s in suggestions:
            boost = decayed.get(s.suggestion.value, 0.0) * self._weight
            # DecayRanker is a boost ranker: it contributes recency-weighted
            # history signal, not a base score. base_score=0.0 here ensures
            # the merged explanation's base_score equals only ScoreRanker's
            # contribution - the true pre-ranking predictor score.
            explanations.append(
                RankingExplanation(
                    value=s.suggestion.value,
                    base_score=0.0,
                    history_boost=boost,
                    final_score=boost,
                    source="decay",
                    base_components={},
                    history_components={"decay": boost} if boost > 0 else {},
                )
            )

        return explanations
