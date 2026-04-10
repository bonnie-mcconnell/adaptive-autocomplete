from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class CompletionContext:
    text: str
    cursor_pos: int | None = None

    def prefix(self) -> str:
        """
        Return the current completion prefix, normalised to lowercase.

        Normalisation:
            All prefixes are lowercased before being returned. The bundled
            vocabulary is lowercase; case-normalising here means typing
            "He" and "he" produce the same completions. Callers that need
            case-sensitive behaviour should construct CompletionContext with
            pre-normalised text.

        Cursor position rules:
            If cursor_pos is provided, the character immediately before the
            cursor is considered in-progress and excluded from the prefix.
            cursor_pos is clamped to [0, len(text)], so out-of-range values
            are safe and return the longest valid prefix.

            Examples:
                text="git checkout", cursor_pos=7  ->  "ch"   (cursor mid-word)
                text="git checkout", cursor_pos=12 ->  "checkou"  (last char in-progress)
                text="he", cursor_pos=None         ->  "he"   (full text, no cursor)
        """
        if self.cursor_pos is not None:
            before = self.text[: self.cursor_pos]
            parts = before.split()

            if not parts:
                return ""

            token = parts[-1]
            result = token[:-1] if token else ""
            return result.lower()

        # No cursor: use full text
        parts = self.text.split()
        return parts[-1].lower() if parts else ""


@dataclass(frozen=True)
class Suggestion:
    """
    A candidate completion.
    """

    value: str


@dataclass(frozen=True)
class ScoredSuggestion:
    """
    A suggestion with an associated score.
    """

    suggestion: Suggestion
    score: float
    explanation: PredictorExplanation | None = None
    trace: list[str] = field(default_factory=list)

    @property
    def value(self) -> str:
        return self.suggestion.value


class Predictor(Protocol):
    """
    Contract implemented by all predictors.
    """

    name: str

    def predict(self, ctx: CompletionContext) -> list[ScoredSuggestion]:
        ...


@dataclass(frozen=True)
class PredictorExplanation:
    """
    Explanation produced by a single predictor.

    Represents a raw signal before any ranking,
    normalization, or aggregation occurs.
    """

    value: str
    score: float
    source: str
    confidence: float


@dataclass(frozen=True)
class WeightedPredictor:
    """
    Predictor paired with a weight applied during aggregation.
    """

    predictor: Predictor
    weight: float = 1.0

    @property
    def name(self) -> str:
        return self.predictor.name


def ensure_context(ctx: CompletionContext | str) -> CompletionContext:
    """
    Normalizes raw text or CompletionContext into CompletionContext.
    """
    if isinstance(ctx, CompletionContext):
        return ctx
    return CompletionContext(text=ctx)