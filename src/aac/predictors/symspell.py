"""
SymSpell approximate string matching: O(1) typo recovery via delete-neighbourhood index.

At construction, every vocabulary word's delete-neighbours (all strings reachable
by up to max_distance deletions) are stored in a hash map. At query time the same
delete-neighbours are computed for the query and looked up. 100% recall; no BK-tree
traversal. At 48k words, max_distance=2: ~1.5s build, ~50MB RAM, ~0.4ms/query.

Ref: Wolf Garbe (2012) https://wolfgarbe.medium.com/1000x-faster-spelling-correction-algorithm-using-symmetricdelete-spelling-correction
"""
from __future__ import annotations

from collections import defaultdict
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
from aac.predictors.bk_tree import levenshtein

_DEFAULT_MAX_DISTANCE = 2


def _deletes(word: str, max_distance: int) -> frozenset[str]:
    """All strings reachable from word by up to max_distance deletions. Original word excluded."""
    deletes: set[str] = set()
    queue: set[str] = {word}

    for _ in range(max_distance):
        next_level: set[str] = set()
        for w in queue:
            for i in range(len(w)):
                deleted = w[:i] + w[i + 1:]
                if deleted != word:
                    deletes.add(deleted)
                next_level.add(deleted)
        queue = next_level

    return frozenset(deletes)


class SymSpellPredictor(Predictor):
    """
    Approximate-match predictor using a delete-neighbourhood index.

    ~0.4ms/query at 48k words (vs ~60ms for BK-tree). Full recall for
    1-3 char queries where TrigramPredictor degrades. Build once at startup.
    """

    name = "symspell"

    def __init__(
        self,
        vocabulary: Iterable[str],
        *,
        max_distance: int = _DEFAULT_MAX_DISTANCE,
        base_score: float = 1.0,
        frequencies: Mapping[str, int] | None = None,
    ) -> None:
        self._max_distance = max_distance
        self._base_score = base_score
        delete_map: dict[str, set[str]] = defaultdict(set)

        vocab_list = list(vocabulary)
        for word in vocab_list:
            if not word:
                continue
            # Add the word itself under its own deletes so queries that
            # exactly match a word are found via the '' delete path.
            delete_map[word].add(word)
            for d in _deletes(word, max_distance):
                delete_map[d].add(word)

        # Freeze to regular dicts for faster lookup. '' is a valid key - see _deletes().
        self._delete_map: dict[str, frozenset[str]] = {
            k: frozenset(v) for k, v in delete_map.items()
        }

        # Pre-computed log-normalised frequency scores.
        # Formula and FREQ_WEIGHT rationale: see aac.predictors._scoring
        self._freq_scores: dict[str, float] = build_freq_scores(frequencies)

    def predict(self, ctx: CompletionContext | str) -> list[ScoredSuggestion]:
        ctx = ensure_context(ctx)
        prefix = ctx.prefix()

        if not prefix:
            return []

        # Collect all candidate vocabulary words by looking up every
        # delete-neighbour of the query in the index.
        candidates: dict[str, int] = {}  # word -> best distance found

        # The query itself is a candidate (distance=0)
        for word in self._delete_map.get(prefix, set()):
            dist = levenshtein(prefix, word)
            if dist <= self._max_distance:
                if word not in candidates or dist < candidates[word]:
                    candidates[word] = dist

        # All delete-neighbours of the query are also candidates
        for d in _deletes(prefix, self._max_distance):
            for word in self._delete_map.get(d, set()):
                dist = levenshtein(prefix, word)
                if dist <= self._max_distance:
                    if word not in candidates or dist < candidates[word]:
                        candidates[word] = dist

        if not candidates:
            return []

        results: list[ScoredSuggestion] = []
        for word, distance in candidates.items():
            # Skip exact matches - completing a word to itself is noise.
            if word == prefix:
                continue

            freq_score = self._freq_scores.get(word, 0.0)
            score = distance_score(self._base_score, distance, freq_score)
            confidence = edit_confidence(distance, self._max_distance)

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

        results.sort(key=lambda s: s.score, reverse=True)
        return results
