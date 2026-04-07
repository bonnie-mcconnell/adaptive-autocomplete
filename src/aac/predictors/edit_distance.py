from __future__ import annotations

from collections.abc import Iterable

from aac.domain.types import (
    CompletionContext,
    Predictor,
    PredictorExplanation,
    ScoredSuggestion,
    Suggestion,
    ensure_context,
)
from aac.predictors.bk_tree import BKTree, _levenshtein


def levenshtein(a: str, b: str) -> int:
    """
    Compute Levenshtein edit distance.

    Public API for callers outside this module. Delegates to the
    shared implementation in bk_tree to avoid duplication.

    Cost model: insertion=1, deletion=1, substitution=1.
    """
    return _levenshtein(a, b)


class EditDistancePredictor(Predictor):
    """
    Error-tolerant predictor using edit distance.

    Intended for recovering from mid-word typos and near-miss prefixes.
    Emits a weak signal that should be combined with stronger predictors.

    Index:
        Uses a BK-tree built at construction time. The BK-tree exploits
        the triangle inequality property of Levenshtein distance to prune
        subtrees that cannot contain results without evaluating them.
        All words within max_distance edits are returned without exception,
        including cases where the first character differs from the query.

    Performance characteristics:
        BK-tree pruning degrades when max_distance is large relative to
        the query length. At max_distance=2 with 4-character prefixes,
        the search visits ~75% of nodes in a 482-word vocabulary — the
        search ball is large enough that triangle inequality pruning helps
        little. At max_distance=1 pruning is substantially more effective.

        At vocabularies over ~100k words, BK-trees become impractical.
        Production systems use trigram indexes or approximate nearest-
        neighbour structures over word embeddings.

    Correctness guarantee:
        Returns every word in the vocabulary within max_distance Levenshtein
        edits of the query prefix, without exception.
    """

    name = "edit_distance"

    def __init__(
        self,
        vocabulary: Iterable[str],
        *,
        max_distance: int = 2,
        base_score: float = 1.0,
    ) -> None:
        self._max_distance = max_distance
        self._base_score = base_score
        # BKTree filters empty strings internally; pass vocabulary directly.
        self._tree = BKTree(vocabulary)

    def predict(self, ctx: CompletionContext | str) -> list[ScoredSuggestion]:
        ctx = ensure_context(ctx)
        prefix = ctx.prefix()

        if not prefix:
            return []

        results: list[ScoredSuggestion] = []

        for word, distance in self._tree.search(prefix, max_distance=self._max_distance):
            score = self._base_score / (1 + distance)
            confidence = max(0.0, 1.0 - (distance / (self._max_distance + 1)))

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