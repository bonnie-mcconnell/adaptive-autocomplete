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


class StaticPrefixPredictor(Predictor):
    """
    Deterministic prefix-based predictor over a fixed vocabulary.

    Intended for use in tests and examples where a controlled,
    fully predictable predictor is needed without frequency data.
    Not part of the public API - see FrequencyPredictor or
    TriePrefixPredictor for production use.
    """

    name = "static_prefix"

    def __init__(self, vocabulary: Iterable[str]) -> None:
        self._vocabulary = tuple(dict.fromkeys(vocabulary))

    def predict(self, ctx: CompletionContext | str) -> list[ScoredSuggestion]:
        ctx = ensure_context(ctx)
        prefix = ctx.prefix()

        if not prefix:
            return []

        results: list[ScoredSuggestion] = []

        for word in self._vocabulary:
            if word == prefix or not word.startswith(prefix):
                continue

            results.append(
                ScoredSuggestion(
                    suggestion=Suggestion(value=word),
                    score=1.0,
                    explanation=PredictorExplanation(
                        value=word,
                        score=1.0,
                        confidence=1.0,
                        source=self.name,
                    ),
                    trace=(
                        f"prefix='{prefix}'",
                        f"matched='{word}'",
                    ),
                )
            )

        return results