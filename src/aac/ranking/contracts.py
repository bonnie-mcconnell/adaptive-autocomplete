"""Shared invariants that every Ranker implementation must satisfy."""

from typing import Protocol, runtime_checkable

from aac.domain.history import History
from aac.domain.types import CompletionContext


@runtime_checkable
class LearnsFromHistory(Protocol):
    """Ranker that holds a History instance shared with the engine."""

    history: History


@runtime_checkable
class PredictorLearnsFromHistory(Protocol):
    """Predictor with a reassignable history property (for reset_history() propagation)."""

    @property
    def history(self) -> History:
        ...

    @history.setter
    def history(self, value: History) -> None:
        ...


@runtime_checkable
class PredictorAcceptsRecord(Protocol):
    """Predictor that maintains private state updated on each selection."""

    def record(self, ctx: CompletionContext, value: str) -> None:
        ...
