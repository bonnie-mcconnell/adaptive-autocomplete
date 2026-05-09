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

            Examples (cursor_pos counts characters from the start of text):
                text="git ch", cursor_pos=6  ->  "c"   (cursor at end: "h" in-progress)
                text="git ch", cursor_pos=5  ->  ""    (cursor before "c": whole token in-progress)
                text="he", cursor_pos=None   ->  "he"  (full text, no cursor)
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

    ``trace`` is an immutable tuple of diagnostic strings recording which
    predictors and rankers contributed to the score.  Using ``tuple``
    rather than ``list`` makes the immutability guarantee of
    ``frozen=True`` complete: callers cannot mutate the trace without
    constructing a new ``ScoredSuggestion``.
    """

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
    Predictor paired with a weight applied during score aggregation.

    Weights must be positive. A weight of 0.0 would silence the predictor
    entirely - remove it from the predictor list instead. Weights above ~5.0
    are rarely useful: they push the weighted sum to be dominated by this
    predictor alone, defeating the purpose of combining signals. The practical
    useful range is (0.0, 5.0]. Default is 1.0.

    The positivity constraint is enforced at construction time. There is no
    upper-bound enforcement - very high weights are unusual but not necessarily
    wrong (e.g. a domain-specific predictor intentionally overwhelming the
    general frequency signal).
    """

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
    Normalizes raw text or CompletionContext into CompletionContext.
    """
    if isinstance(ctx, CompletionContext):
        return ctx
    return CompletionContext(text=ctx)