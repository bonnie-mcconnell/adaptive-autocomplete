"""Ranker protocol: any class implementing rank(prefix, suggestions) -> suggestions satisfies it."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from aac.domain.types import ScoredSuggestion
from aac.ranking.explanation import RankingExplanation


class Ranker(ABC):
    """Base class for ranking strategies. Implementations must be deterministic and explanation-aligned."""

    @abstractmethod
    def rank(
        self,
        prefix: str,
        suggestions: Sequence[ScoredSuggestion],
    ) -> list[ScoredSuggestion]:
        """Return suggestions in ranked order."""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def explain(
        self,
        prefix: str,
        suggestions: Sequence[ScoredSuggestion],
    ) -> list[RankingExplanation]:
        """Return explanations in the same order as rank()."""
        raise NotImplementedError  # pragma: no cover
