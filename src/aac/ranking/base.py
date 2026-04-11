from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from aac.domain.types import ScoredSuggestion
from aac.ranking.explanation import RankingExplanation


class Ranker(ABC):
    """
    Base contract for all ranking strategies.

    Rankers must be:
    - deterministic
    - stable
    - non-mutating
    - explanation-aligned (explain() matches rank() order)
    """

    @abstractmethod
    def rank(
        self,
        prefix: str,
        suggestions: Sequence[ScoredSuggestion],
    ) -> list[ScoredSuggestion]:
        """
        Return suggestions ordered by preference.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def explain(
        self,
        prefix: str,
        suggestions: Sequence[ScoredSuggestion],
    ) -> list[RankingExplanation]:
        """
        Return ranking explanations aligned exactly with rank().
        """
        raise NotImplementedError  # pragma: no cover
