"""Core domain types: Suggestion, ScoredSuggestion, CompletionContext, WeightedPredictor, Ranker protocol."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class CompletionContext:
    text: str
    cursor_pos: int | None = None

    def prefix(self) -> str:
        """
        The completion prefix, normalised to lowercase.

        When cursor_pos is None, returns the last whitespace-delimited token.
        When cursor_pos is set, excludes the character under the cursor
        (treating it as still being typed):

            text="git ch", cursor_pos=6  ->  "c"
            text="git ch", cursor_pos=5  ->  ""
            text="he",     cursor_pos=None -> "he"
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
    """A candidate completion returned by a predictor."""

    value: str


@dataclass(frozen=True)
class ScoredSuggestion:
    """A suggestion with a score and optional diagnostic trace."""

    suggestion: Suggestion
    score: float
    explanation: PredictorExplanation | None = None
    trace: tuple[str, ...] = field(default_factory=tuple)

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
    """Raw score signal from a single predictor, before ranking."""

    value: str
    score: float
    source: str
    confidence: float


@dataclass(frozen=True)
class WeightedPredictor:
    """Predictor with a weight applied during score aggregation. Weight must be > 0."""

    predictor: Predictor
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0.0:
            raise ValueError(
                f"WeightedPredictor weight must be > 0, got {self.weight!r}. "
                f"To silence a predictor, remove it from the predictor list."
            )

    @property
    def name(self) -> str:
        return self.predictor.name


def ensure_context(ctx: CompletionContext | str) -> CompletionContext:
    """
    Normalises raw text or CompletionContext into CompletionContext.
    """
    if isinstance(ctx, CompletionContext):
        return ctx
    return CompletionContext(text=ctx)