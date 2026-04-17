from __future__ import annotations

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

    Score reflects raw frequency magnitude.
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

        # Prefix index: each prefix maps to words, pre-sorted by descending
        # frequency. Built once at construction; O(1) slice at query time.
        # Sorting here costs O(|bucket| log |bucket|) per prefix, but that
        # work is amortised over all queries against that prefix.
        raw: dict[str, list[str]] = defaultdict(list)
        for word in frequencies:
            for length in range(1, len(word) + 1):
                raw[word[:length]].append(word)

        self._index: dict[str, list[str]] = {
            prefix: sorted(words, key=lambda w: frequencies[w], reverse=True)
            for prefix, words in raw.items()
        }

    def predict(self, ctx: CompletionContext | str) -> list[ScoredSuggestion]:
        ctx = ensure_context(ctx)
        prefix = ctx.prefix()

        if not prefix:
            return []

        results: list[ScoredSuggestion] = []

        for word in self._index.get(prefix, []):
            if word == prefix:
                # Exact match: the user has already typed this word.
                # Completing it to itself adds no information.
                continue

            count = self._frequencies[word]
            score = float(count)
            confidence = count / self._max_freq

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
