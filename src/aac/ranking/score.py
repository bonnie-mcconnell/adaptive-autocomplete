from collections.abc import Sequence

from aac.domain.types import ScoredSuggestion
from aac.ranking.base import Ranker
from aac.ranking.explanation import RankingExplanation


class ScoreRanker(Ranker):
    """
    Pure score-based ranker.

    Invariants:
    - Deterministic
    - Stable
    - Non-mutating
    - Idempotent
    """

    def rank(
        self,
        prefix: str,
        suggestions: Sequence[ScoredSuggestion],
    ) -> list[ScoredSuggestion]:
        # Sort by descending score, with original index as a stable tiebreaker.
        # Matches LearningRanker and DecayRanker's sort contract so that
        # composing rankers produces deterministic output regardless of order.
        indexed = list(enumerate(suggestions))
        indexed.sort(key=lambda t: (-t[1].score, t[0]))
        return [s for _, s in indexed]

    def explain(
        self,
        prefix: str,
        suggestions: Sequence[ScoredSuggestion],
    ) -> list[RankingExplanation]:
        ranked = self.rank(prefix, suggestions)
        return [
            RankingExplanation.from_predictor(
                value=s.suggestion.value,
                score=s.score,
                source=s.explanation.source if s.explanation else "score",
            )
            for s in ranked
        ]


def score_and_rank(
    suggestions: Sequence[ScoredSuggestion],
) -> list[ScoredSuggestion]:
    """
    Pure functional helper used by ranking invariant tests.
    """
    return ScoreRanker().rank("", suggestions)
