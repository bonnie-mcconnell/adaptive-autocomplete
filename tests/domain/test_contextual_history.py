"""
Tests for ContextualHistory - the multi-domain history container.

ContextualHistory lets callers maintain separate History objects per
named domain (e.g. "python", "finance", "chat") while recording through
a single interface. Tests verify domain isolation, live view semantics,
and the iteration helpers.
"""
from __future__ import annotations

from aac.domain.contextual_history import ContextualHistory
from aac.domain.history import History
from aac.presets import create_engine

_VOCAB = {
    "hello": 100, "help": 80, "hero": 50, "her": 200,
    "here": 120, "heap": 40, "world": 300, "word": 150,
    "programming": 500, "program": 400, "progress": 300,
}


class TestContextualHistory:
    def test_domain_isolation(self) -> None:
        ctx = ContextualHistory()
        ctx.record("prog", "programming", domain="python")
        ctx.record("prog", "progress",    domain="pm")

        python_h = ctx.for_domain("python")
        pm_h = ctx.for_domain("pm")

        assert python_h.counts_for_prefix("prog").get("programming") == 1
        assert python_h.counts_for_prefix("prog").get("progress", 0) == 0
        assert pm_h.counts_for_prefix("prog").get("progress") == 1
        assert pm_h.counts_for_prefix("prog").get("programming", 0) == 0

    def test_engine_built_on_domain_uses_domain_history(self) -> None:
        ctx = ContextualHistory()
        for _ in range(5):
            ctx.record("he", "heap", domain="finance")
        for _ in range(5):
            ctx.record("he", "hello", domain="chat")

        finance_engine = create_engine("default", vocabulary=_VOCAB, history=ctx.for_domain("finance"))
        chat_engine    = create_engine("default", vocabulary=_VOCAB, history=ctx.for_domain("chat"))

        assert finance_engine.suggest("he")[0] == "heap", (
            "finance domain engine should suggest 'heap' first"
        )
        assert chat_engine.suggest("he")[0] == "hello", (
            "chat domain engine should suggest 'hello' first"
        )

    def test_default_domain_used_when_no_domain_specified(self) -> None:
        ctx = ContextualHistory()
        ctx.record("he", "hero")    # no domain → default
        h = ctx.for_domain()        # no domain → same default
        assert h.counts_for_prefix("he").get("hero") == 1

    def test_for_domain_returns_live_history(self) -> None:
        """Mutations via record() are immediately visible in the view returned
        by for_domain() - it is a live reference, not a snapshot."""
        ctx = ContextualHistory()
        h = ctx.for_domain("shell")
        ctx.record("git", "git commit", domain="shell")
        assert h.counts_for_prefix("git").get("git commit") == 1

    def test_domain_names_sorted(self) -> None:
        ctx = ContextualHistory()
        ctx.for_domain("zzz")
        ctx.for_domain("aaa")
        ctx.for_domain("mmm")
        assert ctx.domain_names() == ["aaa", "mmm", "zzz"]

    def test_domains_iterator(self) -> None:
        ctx = ContextualHistory()
        ctx.record("a", "apple", domain="fruit")
        ctx.record("b", "banana", domain="fruit")
        ctx.record("c", "car", domain="vehicle")

        domains = dict(ctx.domains())
        assert set(domains.keys()) == {"fruit", "vehicle"}
        assert domains["fruit"].counts_for_prefix("a").get("apple") == 1

    def test_total_entries(self) -> None:
        ctx = ContextualHistory()
        ctx.record("a", "apple", domain="d1")
        ctx.record("b", "banana", domain="d1")
        ctx.record("c", "cat", domain="d2")
        assert ctx.total_entries() == 3

    def test_load_domain_replaces_history(self) -> None:
        ctx = ContextualHistory()
        ctx.record("he", "hello", domain="test")

        new_h = History()
        new_h.record("he", "help")
        ctx.load_domain("test", new_h)

        h = ctx.for_domain("test")
        assert h.counts_for_prefix("he").get("help") == 1
        assert h.counts_for_prefix("he").get("hello", 0) == 0

    def test_repr_shows_domain_entry_counts(self) -> None:
        ctx = ContextualHistory()
        ctx.record("a", "apple", domain="fruit")
        r = repr(ctx)
        assert "fruit" in r
        assert "ContextualHistory" in r
