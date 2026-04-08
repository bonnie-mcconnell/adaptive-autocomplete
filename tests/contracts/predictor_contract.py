from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from aac.domain.types import CompletionContext, ScoredSuggestion, Suggestion


class PredictorContractTestMixin(ABC):
    """
    Shared contract tests for all Predictor implementations.

    Any predictor must satisfy these invariants. Add a subclass to
    tests/contracts/test_predictor_contracts.py for each new predictor.
    """

    @abstractmethod
    def make_predictor(self) -> Any:
        """Return a predictor instance under test."""
        raise NotImplementedError

    def test_has_name(self) -> None:
        predictor = self.make_predictor()
        assert isinstance(predictor.name, str)
        assert predictor.name.strip() != ""

    def test_predict_returns_list(self) -> None:
        predictor = self.make_predictor()
        result = predictor.predict(CompletionContext(text="h"))
        assert isinstance(result, list)

    def test_predict_returns_scored_suggestions(self) -> None:
        predictor = self.make_predictor()
        for item in predictor.predict(CompletionContext(text="h")):
            assert isinstance(item, ScoredSuggestion)
            assert isinstance(item.suggestion, Suggestion)
            assert isinstance(item.suggestion.value, str)
            assert isinstance(item.score, float)

    def test_predict_does_not_mutate_context(self) -> None:
        predictor = self.make_predictor()
        ctx = CompletionContext(text="hello")
        predictor.predict(ctx)
        assert ctx.text == "hello"