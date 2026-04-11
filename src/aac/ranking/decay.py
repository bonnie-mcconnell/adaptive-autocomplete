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

    def _now_utc(self) -> datetime:
        return self._now if self._now is not None else utcnow()

    def _decayed_counts(self, prefix: str) -> dict[str, float]:
        now = self._now_utc()
        counts: dict[str, float] = defaultdict(float)

        for entry in self.history.entries_for_prefix(prefix):
            counts[entry.value] += self._decay.weight(
                now=now,
                event_time=entry.timestamp,
            )

        return dict(counts)

    def rank(
        self,
        prefix: str,
        suggestions: Sequence[ScoredSuggestion],
    ) -> list[ScoredSuggestion]:
        if not suggestions:
            return []

        decayed = self._decayed_counts(prefix)
        if not decayed:
            return list(suggestions)

        ranked: list[ScoredSuggestion] = []

        for s in suggestions:
            boost = decayed.get(s.suggestion.value, 0.0) * self._weight

            ranked.append(
                ScoredSuggestion(
                    suggestion=s.suggestion,
                    score=s.score + boost,
                    explanation=s.explanation,
                    trace=s.trace,
                )
            )

        ranked.sort(key=lambda s: s.score, reverse=True)
        return ranked

    def explain(
        self,
        prefix: str,
        suggestions: Sequence[ScoredSuggestion],
    ) -> list[RankingExplanation]:
        decayed = self._decayed_counts(prefix)

        explanations: list[RankingExplanation] = []

        for s in suggestions:
            boost = decayed.get(s.suggestion.value, 0.0) * self._weight

            explanations.append(
                RankingExplanation(
                    value=s.suggestion.value,
                    base_score=0.0,   # this ranker adds, does not define base
                    history_boost=boost,
                    final_score=boost,
                    source="decay",
                )
            )

        return explanations
