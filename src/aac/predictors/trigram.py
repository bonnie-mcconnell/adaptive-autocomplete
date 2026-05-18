"""
Trigram index for approximate string matching at scale.

Pre-filters Levenshtein candidates by shared trigrams, cutting BK-tree's ~60ms/query
to ~600µs at 48k words. Requires queries >= 4 chars; trigrams give poor discrimination
below that. For short-prefix typo recovery use SymSpellPredictor instead.

Build: O(V*L), ~258ms one-time. Query: O(Q*B + S*Q²) ≈ 600µs avg.

Ref: Gravano et al. (2001). "Approximate String Joins in a Database (Almost) for Free." VLDB.
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

# Minimum query length below which trigram pre-filtering gives
# poor discrimination and the shortlist degenerates to O(vocabulary).
_MIN_PREFIX_LENGTH = 4


def _trigrams(s: str) -> frozenset[str]:
    """
    Return the trigram set of a padded string.

    Padding with two leading spaces and one trailing space gives
    boundary trigrams that encode word-start and word-end position,
    so 'he' and 'she' share ' he' but not '  h'.

    Example:
        _trigrams('helo') == frozenset({'  h', ' he', 'hel', 'elo', 'lo '})
    """
    padded = f"  {s} "
    return frozenset(padded[i : i + 3] for i in range(len(padded) - 2))


class TrigramIndex:
    """
    Inverted trigram index over a fixed vocabulary.

    Separates index construction (slow, one-time) from candidate lookup
    (fast, per-query). Used by TrigramPredictor.
    """

    def __init__(self, vocabulary: Iterable[str]) -> None:
        raw: dict[str, list[str]] = defaultdict(list)

        for word in vocabulary:
            if not word:
                continue
            for tg in _trigrams(word):
                raw[tg].append(word)

        self._index: dict[str, list[str]] = dict(raw)

    def candidates(
        self,
        query: str,
        *,
        max_distance: int,
    ) -> list[tuple[str, int]]:
        """Return (word, distance) pairs within max_distance. Empty if len(query) < 4."""
        if len(query) < _MIN_PREFIX_LENGTH:
            return []

        query_tgs = _trigrams(query)
        query_len = len(query)

        # Conservative threshold: share at least this many trigrams to be
        # worth running exact Levenshtein on. Derived from the fact that
        # each edit operation destroys at most 3 trigrams (the ones
        # containing the changed character and its two neighbours).
        min_shared = max(1, len(query_tgs) - max_distance)

        overlap: dict[str, int] = defaultdict(int)
        for tg in query_tgs:
            for word in self._index.get(tg, []):
                overlap[word] += 1

        results: list[tuple[str, int]] = []

        for word, shared in overlap.items():
            if shared < min_shared:
                continue
            # Length difference is a free lower bound on edit distance.
            if abs(len(word) - query_len) > max_distance:
                continue
            dist = levenshtein(query, word)
            if dist <= max_distance:
                results.append((word, dist))

        # Sort by (distance asc, word asc) for deterministic output.
        # Without this, equal-distance results reflect hash-randomised dict order.
        results.sort(key=lambda t: (t[1], t[0]))
        return results


class TrigramPredictor(Predictor):
    """
    Approximate-match predictor backed by a trigram index (~600µs/query at 48k words).
    Returns empty for queries < 4 chars; use SymSpellPredictor there instead.
    """

    name = "trigram"

    def __init__(
        self,
        vocabulary: Iterable[str],
        *,
        max_distance: int = 2,
        base_score: float = 1.0,
        frequencies: Mapping[str, int] | None = None,
    ) -> None:
        self._index = TrigramIndex(vocabulary)
        self._max_distance = max_distance
        self._base_score = base_score
        # Pre-computed log-normalised frequency scores.
        # Formula and FREQ_WEIGHT rationale: see aac.predictors._scoring
        self._freq_scores: dict[str, float] = build_freq_scores(frequencies)

    def predict(self, ctx: CompletionContext | str) -> list[ScoredSuggestion]:
        ctx = ensure_context(ctx)
        prefix = ctx.prefix()

        if not prefix:
            return []

        results: list[ScoredSuggestion] = []

        for word, distance in self._index.candidates(
            prefix, max_distance=self._max_distance
        ):
            # Exact matches (word == prefix) are excluded - completing a word
            # to itself is noise.  Mirrors FrequencyPredictor and SymSpellPredictor.
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

        # Re-sort by descending score so frequency tiebreaking takes effect.
        results.sort(key=lambda s: s.score, reverse=True)
        return results
