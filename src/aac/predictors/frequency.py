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


class FrequencyPredictor(Predictor):
    """
    Suggests words based on observed global frequency.

    Builds a prefix index at construction time so that lookups are
    O(results) rather than O(vocabulary). This matters at scale: a
    vocabulary of 100k words with a linear scan would be noticeably
    slow at interactive keystroke latency.

    Score reflects raw frequency magnitude.
    Confidence reflects relative dominance among known frequencies.

    Design notes:
        - Exact matches (prefix == word) are excluded. If the user has
          already typed the complete word, completing it to itself is
          noise, not signal. This matches the behaviour of TriePrefixPredictor.
        - Input is case-sensitive. The bundled vocabulary is lowercase;
          callers are responsible for normalising case before prediction
          if case-insensitive matching is required.
    """

    name = "frequency"

    def __init__(self, frequencies: Mapping[str, int]) -> None:
        if not frequencies:
            raise ValueError("frequencies must not be empty")

        self._frequencies = dict(frequencies)
        self._max_freq = max(frequencies.values())

        if self._max_freq <= 0:
            raise ValueError(
                "frequencies must contain at least one positive value; "
                f"got max={self._max_freq}"
            )

        # Prefix index: each prefix of each word maps to the matching words.
        # Built once at construction, used for every predict() call.
        # Memory: O(sum of word lengths × avg words per prefix).
        self._index: dict[str, list[str]] = defaultdict(list)
        for word in frequencies:
            for length in range(1, len(word) + 1):
                self._index[word[:length]].append(word)

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

        return results