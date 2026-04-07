from __future__ import annotations

from abc import ABC, abstractmethod

from aac.domain.types import CompletionContext, ScoredSuggestion, Suggestion


class PredictorContractTestMixin(ABC):
    """
    Shared contract tests for all Predictor implementations.

    Any predictor (real or dummy) must satisfy these invariants.
    """
    
    @abstractmethod
    def make_predictor(self):
        """
        Must return a predictor instance under test.
        """
        raise NotImplementedError

    def test_has_name(self):
        predictor = self.make_predictor()
        assert isinstance(predictor.name, str)
        assert predictor.name.strip() != ""

    def test_predict_returns_list(self):
        predictor = self.make_predictor()
        ctx = CompletionContext(text="h")
        result = predictor.predict(ctx)
        assert isinstance(result, list)

    def test_predict_returns_scored_suggestions(self):
        predictor = self.make_predictor()
        ctx = CompletionContext(text="h")

        results = predictor.predict(ctx)

        for item in results:
            assert isinstance(item, ScoredSuggestion)
            assert isinstance(item.suggestion, Suggestion)
            assert isinstance(item.suggestion.value, str)
            assert isinstance(item.score, float)

    def test_predict_does_not_mutate_context(self):
        predictor = self.make_predictor()
        ctx = CompletionContext(text="hello")

        predictor.predict(ctx)

        assert ctx.text == "hello"


