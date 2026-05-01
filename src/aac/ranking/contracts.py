from typing import Protocol, runtime_checkable

from aac.domain.history import History


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
