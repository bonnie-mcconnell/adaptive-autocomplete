from __future__ import annotations

from aac.domain.types import CompletionContext, ScoredSuggestion, Suggestion


def test_prefix_without_cursor() -> None:
    ctx = CompletionContext(text="git che")
    assert ctx.prefix() == "che"


def test_prefix_with_cursor() -> None:
    ctx = CompletionContext(text="git checkout", cursor_pos=7)
    assert ctx.prefix() == "ch"


def test_prefix_is_lowercased() -> None:
    assert CompletionContext("He").prefix() == "he"
    assert CompletionContext("HELP").prefix() == "help"
    assert CompletionContext("Git Checkout").prefix() == "checkout"


def test_prefix_with_cursor_is_lowercased() -> None:
    ctx = CompletionContext(text="GIT CHECKOUT", cursor_pos=7)
    assert ctx.prefix() == "ch"


def test_empty_text_returns_empty_prefix() -> None:
    assert CompletionContext("").prefix() == ""
    assert CompletionContext("   ").prefix() == ""


def test_prefix_whitespace_only_returns_empty() -> None:
    assert CompletionContext(text="   ", cursor_pos=2).prefix() == ""


def test_scored_suggestion_holds_score() -> None:
    s = Suggestion("checkout")
    scored = ScoredSuggestion(suggestion=s, score=0.8)
    assert scored.suggestion.value == "checkout"
    assert scored.score == 0.8


def test_scored_suggestion_value_property() -> None:
    scored = ScoredSuggestion(suggestion=Suggestion("hello"), score=1.0)
    assert scored.value == "hello"