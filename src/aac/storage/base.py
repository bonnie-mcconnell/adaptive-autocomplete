from __future__ import annotations

from abc import ABC, abstractmethod

from aac.domain.history import History


class HistoryStore(ABC):
    """
    Persistence boundary for History.
    """

    @abstractmethod
    def load(self) -> History:
        """
        Load history from storage.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def save(self, history: History) -> None:
        """
        Persist history to storage.
        """
        raise NotImplementedError  # pragma: no cover
