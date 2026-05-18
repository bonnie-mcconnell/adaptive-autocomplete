"""FrequencyPredictor: scores candidates by log-normalised word frequency from a frequency table."""
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

_DEFAULT_MAX_RESULTS = 100


class FrequencyPredictor(Predictor):
    """
    Suggests words by log-normalised corpus frequency.

    Prefix index is built and pre-sorted at construction so predict() is O(max_results).
    Scores are in (0, 1] via log(1 + freq) / log(1 + max_freq) so weights across
    predictors are interpretable (weight=1.5 means 1.5x the frequency signal).
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

        # Prefix index: prefix -> words sorted descending by frequency.
        # Exact matches excluded at build time (range ends before len(word)).
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
        """Add or update a word without rebuilding the index. O(B) per prefix bucket."""
        if frequency <= 0:
            return

        word = str(word)
        if not word:
            return

        self._frequencies[word] = frequency
        if frequency > self._max_freq:
            self._max_freq = frequency
            self._log_max = math.log1p(self._max_freq)

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
                # Find correct descending-frequency position.
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

        # No exact-match guard: index built with range(1, len(word)) excludes them.
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
