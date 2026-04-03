from aac.domain.types import ScoredSuggestion, Suggestion, CompletionContext
from aac.engine import AutocompleteEngine


class FakePredictor:
    def __init__(self, name: str, suggestions: list[ScoredSuggestion]):
        self.name = name
        self._suggestions = suggestions

    def predict(self, ctx: CompletionContext) -> list[ScoredSuggestion]:
        return self._suggestions


def test_autocomplete_engine_aggregates_and_sorts():
    p1 = FakePredictor(
        "p1",
        [
            ScoredSuggestion(suggestion=Suggestion("foo"), score=0.2),
            ScoredSuggestion(suggestion=Suggestion("bar"), score=0.9),
        ],
    )

    p2 = FakePredictor(
        "p2",
        [
            ScoredSuggestion(suggestion=Suggestion("baz"), score=0.5),
        ],
    )

    engine = AutocompleteEngine([p1, p2])

    suggestions = engine.suggest("anything")

    assert [s.value for s in suggestions] == ["bar", "baz", "foo"]
