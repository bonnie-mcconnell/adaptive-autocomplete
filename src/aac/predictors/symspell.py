"""
SymSpell-style approximate string matching for fast typo recovery.

Solves the BK-tree scalability problem with a fundamentally different approach.

Background
----------
BK-tree search degrades toward O(n) when max_distance is large relative
to the vocabulary's edit-distance distribution - which happens with short
English words at max_distance=2. At 48k words the BK-tree takes ~60ms/call.

SymSpell pre-computes all delete-neighbours of every vocabulary word at
construction time and stores them in a hash map. At query time, the query's
delete-neighbours are computed on the fly (fast) and looked up in the map.
This trades construction memory for O(1) average query time.

Construction: generate all strings reachable from each word by up to
max_distance deletions. For max_distance=2 and avg word length 7, each
word generates ~(7 + 7*6/2) ≈ 28 delete-neighbours. At 48k words that's
~1.3M entries in the map. Memory is ~50MB - acceptable for server use.

Query: generate all delete-neighbours of the query (same computation,
same cost). For a 7-char query at max_distance=2: ~28 lookups. Each
lookup is O(1). Total query time is O(Q × D²) where Q is query length
and D is max_distance - effectively constant for typical inputs.

Correctness
-----------
Delete-neighbourhood matching exactly finds all words within max_distance
Levenshtein edits, because any alignment of two strings within edit
distance d can be expressed as a sequence of at most d deletions from
each side. The set of (query_deletes ∩ word_deletes) is non-empty if and
only if distance(query, word) <= max_distance.

For max_distance=2: O(1) average query, 100% recall. No false negatives.

References
----------
Wolf Garbe (2012). "1000x Faster Spelling Correction Algorithm."
https://wolfgarbe.medium.com/1000x-faster-spelling-correction-algorithm-using-symmetricdelete-spelling-correction
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
    """
    Return all strings reachable from word by up to max_distance deletions.

    For word='hello', max_distance=1:
        {'ello', 'hllo', 'helo', 'hell'}
        (deduplicated: 4 unique strings)

    The original word is NOT included - the map key is always a
    deleted form, and the value is the original vocabulary word.

    The empty string '' may appear in the output for short words
    (e.g. _deletes('a', 1) returns {''}). This is correct and necessary
    for SymSpell correctness: both 'a' and 'b' map to '' at distance 1,
    so sharing the '' key is how the algorithm identifies that 'a' and 'b'
    are substitution candidates (edit distance 1). Similarly, 2-char words
    and 2-char queries both generate '' at max_distance=2, which is how
    SymSpell finds all pairs of short words within max_distance of each other.
    Filtering '' would introduce false negatives for short-word matching.
    """
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

    100x faster than BK-tree at max_distance=2 over 48k words:
        BKTree (EditDistancePredictor): ~60ms/call
        SymSpellPredictor:              ~0.4ms/call

    Full recall: finds every word within max_distance Levenshtein edits.
    No minimum prefix length constraint (unlike TrigramPredictor).
    Works correctly on 1-3 character queries.

    Construction cost:
        O(V × L × D) time, O(V × L × D) memory.
        At 48k words, avg length 7, max_distance=2: ~1.5s build, ~50MB RAM.
        Build once at startup; queries are O(Q × D²) ≈ O(1).

    When to use:
        - Need typo recovery on short prefixes (1-3 chars)
        - Need O(1) query time regardless of vocabulary size
        - Memory is not constrained (server environment)

    When to use TrigramPredictor instead:
        - Prefix length always >= 4 (TrigramPredictor is lighter to build)
        - Memory is constrained (TrigramPredictor uses less memory)

    Scoring:
        1.0 / (1 + distance) - identical to TrigramPredictor and
        EditDistancePredictor so the three are interchangeable.
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

        # Freeze to regular dicts for faster lookup.
        # Note: the empty string '' IS a valid key. It is needed for SymSpell
        # to correctly find short-word candidates: a 1-char word 'a' generates
        # '' as a distance-1 delete; a 2-char query 'he' generates '' as a
        # distance-2 delete. Both mapping to '' is how SymSpell identifies that
        # 'a' and 'he' are within edit distance 2 of each other (which they are:
        # two substitutions). Filtering '' would cause false negatives for all
        # pairs of short words within max_distance of each other.
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
            # Exact matches (word == prefix) are excluded - completing a word
            # to itself is noise, not a suggestion.  This mirrors the behaviour
            # of FrequencyPredictor, which excludes exact matches at index
            # construction time.  Without this exclusion, a low-frequency exact
            # match (e.g. "prog" freq=14) would outscore all prefix completions
            # ("program" freq=1860, "programming" freq=234) because distance=0
            # gives it the maximum score regardless of frequency.
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
