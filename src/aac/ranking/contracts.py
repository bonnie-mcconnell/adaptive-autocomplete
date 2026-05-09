"""Shared invariants that every Ranker implementation must satisfy."""

from typing import Protocol, runtime_checkable

from aac.domain.history import History
from aac.domain.types import CompletionContext


@runtime_checkable
class LearnsFromHistory(Protocol):
    """
    Contract for rankers that adapt based on user feedback.

    Any ranker implementing this protocol must expose
    a shared History instance used by the engine.
    """

    history: History


@runtime_checkable
class PredictorLearnsFromHistory(Protocol):
    """
    Contract for predictors that hold a History reference.

    Predictors implementing this protocol expose a ``history`` property
    that can be reassigned.  The engine uses this to propagate a new
    History instance to all relevant predictors when ``reset_history()``
    is called, ensuring no predictor continues reading from the discarded
    History after a reset.

    Implement this only if your predictor needs to read from History
    directly (as HistoryPredictor does).  Predictors that are stateless
    or that receive their signals through the shared History indirectly
    do not need to implement this protocol.
    """

    @property
    def history(self) -> History:
        ...

    @history.setter
    def history(self, value: History) -> None:
        ...


@runtime_checkable
class PredictorAcceptsRecord(Protocol):
    """
    Contract for predictors that maintain private state beyond the shared History.

    The engine calls ``record(ctx, value)`` on every predictor that satisfies
    this protocol after each user selection.  This hook is intentionally narrow:
    most predictors are stateless and should NOT implement it.

    The canonical use case is a predictor that maintains its own frequency table
    or n-gram model that updates in response to selections - state that lives
    inside the predictor and is not exposed through the shared History.

    Note on HistoryPredictor:
        HistoryPredictor reads directly from the shared History instance.
        The engine already writes to History in ``record_selection()``, so
        HistoryPredictor must NOT implement this protocol.  If it did, each
        selection would be recorded twice, doubling the learning signal.

    Implementing this protocol:
        Add a ``record(self, ctx: CompletionContext, value: str) -> None``
        method to your predictor.  The engine will call it automatically.
        No registration is required - duck typing via ``isinstance()`` is used.
    """

    def record(self, ctx: CompletionContext, value: str) -> None:
        ...
