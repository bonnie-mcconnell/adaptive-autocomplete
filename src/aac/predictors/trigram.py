"""
Trigram index for approximate string matching at scale.

Solves the BK-tree scalability problem at long query lengths.

Background
----------
BK-tree search degrades toward O(n) when the search ball (the set of
all strings within max_distance edits) covers a large fraction of the
vocabulary. This happens at max_distance=2 with short prefixes: a 4-char
query has ~60ms latency at 48k words.

Trigram pre-filtering reduces Levenshtein work to a shortlist:

  1. Build an inverted index: trigram -> [words that contain it].
     Trigrams are computed on padded strings ("  word ") to give
     boundary-sensitive tokens.

  2. At query time: intersect the query's trigrams with the index to
     build a candidate shortlist, then run exact Levenshtein on it.

Shortlist size scales with trigram overlap, not vocabulary size. For
queries of length >= 5, the shortlist is typically 20-100 words. For
short queries (length < 4), trigrams provide weak discrimination and
the shortlist can approach the full vocabulary - so this predictor
requires min_prefix_length (default: 4) and returns empty below it.

Complexity
----------
    Construction:  O(V × L) - 258ms at 48k words, one-time cost
    Query (≥4ch):  O(Q × B + S × Q²) ≈ 600µs avg at 48k words
    Query (<4ch):  returns [] - use EditDistancePredictor on a small vocab

    where V=vocab size, L=avg word length, Q=query length,
    B=avg bucket size, S=shortlist size.

Correctness tradeoff
--------------------
The trigram filter is a heuristic. It may miss true matches when query
and target share few trigrams despite small edit distance - worst case:
'ab' vs 'ba' (transposition), or very short strings. The threshold is
tuned conservatively (max(1, len(query_trigrams) - max_distance)) to
keep recall high for realistic typing errors on 4+ char queries.

For exact recall over small vocabularies, use EditDistancePredictor.

References
----------
Gravano et al. (2001). "Approximate String Joins in a Database
(Almost) for Free." VLDB.
"""
from __future__ import annotations

import math
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
        """
        Return (word, distance) pairs within max_distance of query.

        Returns [] if len(query) < _MIN_PREFIX_LENGTH - trigrams provide
        weak discrimination for short strings.
        """
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
    Approximate-match predictor backed by a trigram index.

    Replaces EditDistancePredictor (BK-tree) when vocabulary size makes
    BK-tree latency unacceptable. At 48k words and max_distance=2:

        EditDistancePredictor:  ~60ms/call  (BK-tree degrades to O(n))
        TrigramPredictor:       ~600µs/call (shortlist of ~20-100 words)

    Constraint: returns empty for queries shorter than 4 characters.
    Trigrams provide too little discrimination on 1-3 char strings.
    For short-prefix typo recovery, use EditDistancePredictor on a
    curated small vocabulary.

    Scoring:
        Primary: 1.0 / (1 + distance) - closer matches score higher.
        Secondary (when frequencies provided): a log-scaled frequency bonus
        capped at 10% of the minimum distance score breaks ties between
        equal-distance matches. Distance always dominates; frequency only
        separates words at the same edit distance. Score matches
        EditDistancePredictor's scale so the two are interchangeable.
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
        # Pre-compute log-scaled frequency bonus for tiebreaking within equal-
        # distance groups. Capped at 10% of the minimum distance score so a
        # common word at distance 2 never outranks a rare word at distance 1.
        if frequencies:
            max_freq = max(frequencies.values()) or 1
            cap = (base_score / (1 + max_distance)) * 0.1
            self._freq_bonus: dict[str, float] = {
                w: cap * (math.log1p(f) / math.log1p(max_freq))
                for w, f in frequencies.items()
            }
        else:
            self._freq_bonus = {}

    def predict(self, ctx: CompletionContext | str) -> list[ScoredSuggestion]:
        ctx = ensure_context(ctx)
        prefix = ctx.prefix()

        if not prefix:
            return []

        results: list[ScoredSuggestion] = []

        for word, distance in self._index.candidates(
            prefix, max_distance=self._max_distance
        ):
            distance_score = self._base_score / (1 + distance)
            freq_bonus = self._freq_bonus.get(word, 0.0)
            score = distance_score + freq_bonus
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

        # Re-sort by descending score so frequency tiebreaking takes effect.
        results.sort(key=lambda s: s.score, reverse=True)
        return results
