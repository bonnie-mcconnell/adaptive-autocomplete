from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping

from aac.domain.types import (
    CompletionContext,
    Predictor,
    PredictorExplanation,
    ScoredSuggestion,
    Suggestion,
    ensure_context,
)

_DEFAULT_MAX_RESULTS = 20


class FrequencyPredictor(Predictor):
    """
    Suggests words based on observed global frequency.

    Builds a prefix index at construction time so that lookups are
    O(max_results) rather than O(vocabulary). Each prefix bucket is
    pre-sorted by descending frequency at construction time, so predict()
    can take the top-N slice without sorting at query time.

    This matters at real scale: with a 48k-word vocabulary, the prefix
    "s" matches 5,110 words. Sorting all of them on every keystroke
    adds ~3ms of unnecessary work. Pre-sorting the index at construction
    keeps predict() fast regardless of vocabulary size.

    Score model:
        Scores are log-normalised to the range (0, 1]:

            score = log(1 + freq) / log(1 + max_freq)

        Raw corpus counts span many orders of magnitude (e.g. "the" ≈50k,
        "zymurgy" ≈1). Emitting raw counts causes FrequencyPredictor to
        dominate the combined score by 4–5 orders of magnitude, making the
        weights on HistoryPredictor and typo predictors effectively meaningless.
        Log-normalisation keeps all predictors in a common (0, 1] space so
        that weights in WeightedPredictor are interpretable: weight=1.5 on
        HistoryPredictor means 1.5× the frequency signal, not 1.5 out of 50,000.

    Confidence reflects relative dominance among known frequencies.

    Args:
        frequencies: Mapping of word -> frequency count.
        max_results: Maximum number of candidates to return per query.
                     Limits the pre-sorted slice, not the index itself.
                     Default: 20.

    Design notes:
        - Exact matches (prefix == word) are excluded. If the user has
          already typed the complete word, completing it to itself is
          noise, not signal. This matches the behaviour of TriePrefixPredictor.
        - Input is case-sensitive. The bundled vocabulary is lowercase;
          callers are responsible for normalising case before prediction
          if case-insensitive matching is required.
        - Memory: O(sum of word lengths × avg words per prefix). For a
          48k-word vocabulary with avg length 7, this is approximately
          ~344k string references - around 2.8MB in CPython. Acceptable
          for a server process; for embedded use, limit vocabulary size.
    """

    name = "frequency"

    def __init__(
        self,
        frequencies: Mapping[str, int],
        *,
        max_results: int = _DEFAULT_MAX_RESULTS,
    ) -> None:
        if not frequencies:
            raise ValueError("frequencies must not be empty")
        if max_results < 1:
            raise ValueError(f"max_results must be >= 1, got {max_results}")

        self._frequencies = dict(frequencies)
        self._max_freq = max(frequencies.values())
        self._max_results = max_results

        if self._max_freq <= 0:
            raise ValueError(
                "frequencies must contain at least one positive value; "
                f"got max={self._max_freq}"
            )

        # log(1 + max_freq): denominator for normalisation, computed once.
        self._log_max = math.log1p(self._max_freq)

        # Prefix index: each prefix maps to words, pre-sorted by descending
        # frequency. Built once at construction; O(1) slice at query time.
        # Sorting here costs O(|bucket| log |bucket|) per prefix, but that
        # work is amortised over all queries against that prefix.
        # Index prefixes of length 1..len(word)-1.
        # Exact matches (prefix == word) are excluded here, not filtered in
        # predict(). Filtering at query time costs one comparison per candidate
        # per call. Excluding at build time costs nothing at query time and
        # keeps the index smaller: ~48k fewer entries for a 48k-word vocabulary.
        raw: dict[str, list[str]] = defaultdict(list)
        for word in frequencies:
            if frequencies[word] <= 0:
                continue  # exclude zero-frequency words - they contribute no signal
            for length in range(1, len(word)):   # excludes len(word) → no exact match
                raw[word[:length]].append(word)

        self._index: dict[str, list[str]] = {
            prefix: sorted(words, key=lambda w: frequencies[w], reverse=True)
            for prefix, words in raw.items()
        }

    def _log_score(self, freq: int) -> float:
        """Return the log-normalised score for a raw frequency count."""
        return math.log1p(freq) / self._log_max

    def add_word(self, word: str, frequency: int) -> None:
        """
        Add or update a word in the vocabulary without rebuilding the index.

        Useful for domain-specific vocabularies that grow at runtime - for
        example, adding user-defined terms or recently observed completions.

        If the word already exists, its frequency and index position are
        updated. Words with frequency <= 0 are ignored.

        Complexity:
            O(B) per prefix bucket, where B is the number of words already
            in that bucket. Each bucket is a list sorted by descending
            frequency; insertion requires a linear scan to find the correct
            position. Python's ``bisect`` module operates on a sorted sequence
            by value, but bucket entries are words (strings), not frequencies -
            a key function is unavailable without allocating a parallel list.
            Both approaches are O(B); the direct scan used here allocates
            nothing extra.

            In practice ``add_word`` is called during startup or infrequently
            at runtime, so O(B) per call is acceptable. For high-frequency
            runtime vocabulary growth (thousands of calls per second),
            consider rebuilding the predictor from scratch or using
            ``sortedcontainers.SortedKeyList``.

        Parameters:
            word:      The word to add.
            frequency: Corpus frequency count. Must be > 0.

        Example::

            engine = create_engine("production")
            engine.predictors[0].add_word("asyncio", 500)  # surface asyncio early
        """
        if frequency <= 0:
            return

        word = str(word)
        if not word:
            return

        self._frequencies[word] = frequency
        if frequency > self._max_freq:
            self._max_freq = frequency
            self._log_max = math.log1p(self._max_freq)
        # Note: _log_max only grows, never shrinks.  This is intentional.
        # Log-normalisation requires a stable denominator so that scores
        # already returned to callers remain consistent with future calls.
        # If a word with lower frequency is added, the denominator stays put
        # and the new word's score is computed relative to the existing max -
        # which is the correct semantics: the new word is less common than
        # the most common word already in the vocabulary.

        # Update prefix index for all prefixes of this word.
        for length in range(1, len(word)):
            prefix = word[:length]
            if prefix not in self._index:
                self._index[prefix] = [word]
            else:
                bucket = self._index[prefix]
                # Remove existing entry if word is already in bucket.
                try:
                    bucket.remove(word)
                except ValueError:
                    pass
                # Insert at the correct descending-frequency position.
                # Linear scan: find the first bucket entry whose frequency is
                # strictly less than ours and insert before it so the bucket
                # remains sorted highest-first. Falls through to append when
                # all existing entries have equal or higher frequency.
                insert_pos = len(bucket)
                for i, existing in enumerate(bucket):
                    if self._frequencies.get(existing, 0) < frequency:
                        insert_pos = i
                        break
                bucket.insert(insert_pos, word)

    def predict(self, ctx: CompletionContext | str) -> list[ScoredSuggestion]:
        ctx = ensure_context(ctx)
        prefix = ctx.prefix()

        if not prefix:
            return []

        results: list[ScoredSuggestion] = []

        # No exact-match guard needed here: the prefix index excludes
        # words under their own full string (built with range(1, len(word))).
        for word in self._index.get(prefix, []):
            freq = self._frequencies[word]
            score = self._log_score(freq)
            confidence = score  # log-normalised score doubles as confidence

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

            if len(results) >= self._max_results:
                break

        return results
