"""
ContextualHistory: domain-keyed learning.

Problem
-------
The base ``History`` class keys every entry by prefix alone.  In a
multi-domain application - a code editor, a CLI with subcommands, a
search box that serves different product categories - the user's selection
of "python" in a shell context and "python" in a documentation search
context represent different intentions.  Mixing them into a single prefix
key produces noisy suggestions: a user who always types "git" → "git
commit" in their terminal starts seeing "git commit" suggested in their
docs search box.

Solution
--------
``ContextualHistory`` wraps a ``History`` instance per domain.  Each domain
is an arbitrary string (e.g. "shell", "python", "search:products").
``record()`` and ``suggest()`` accept an optional ``domain`` parameter.
When domain is ``None``, all calls fall through to a shared default
``History``, making ``ContextualHistory`` a drop-in replacement for
``History`` in single-domain applications.

The engine API accepts ``ContextualHistory`` wherever ``History`` is
accepted.  Domain selection happens at the call site, not inside the
engine, so no engine internals need to change.

Usage
-----
::

    from aac.domain.contextual_history import ContextualHistory
    from aac.presets import create_engine

    ctx_history = ContextualHistory()
    engine = create_engine("production", history=ctx_history.for_domain("shell"))

    # Record with domain
    ctx_history.record("git", "git commit", domain="shell")
    ctx_history.record("git", "git push",   domain="shell")
    ctx_history.record("git", "gitignore",  domain="search")

    # Suggest using shell domain engine
    engine.suggest("git")   # "git commit" leads, not "gitignore"

    # Switch to search domain engine
    search_engine = create_engine("production", history=ctx_history.for_domain("search"))
    search_engine.suggest("git")   # "gitignore" leads

    # Cross-domain comparison
    for domain, hist in ctx_history.domains():
        print(domain, list(hist.entries()))

Storage
-------
``ContextualHistory`` can be persisted per-domain via ``JsonHistoryStore``::

    from aac.storage.json_store import JsonHistoryStore

    for domain, hist in ctx_history.domains():
        store = JsonHistoryStore(path / f"{domain}.json")
        store.save(hist)

    # Restore
    ctx_history = ContextualHistory()
    for path in history_dir.glob("*.json"):
        domain = path.stem
        store = JsonHistoryStore(path)
        ctx_history.load_domain(domain, store.load())
"""
from __future__ import annotations

from collections.abc import Iterator

from aac.domain.history import History

_DEFAULT_DOMAIN = "__default__"


class ContextualHistory:
    """
    Domain-partitioned history for multi-context autocomplete.

    Each domain is an independent ``History`` instance.  Domains are
    created lazily on first access - no upfront registration needed.

    Parameters:
        default_domain: Name used when no domain is specified.  Defaults
                        to ``"__default__"``.  Change this if you want a
                        more meaningful name in serialised output.

    Example::

        ctx = ContextualHistory()
        ctx.record("prog", "programming", domain="python")
        ctx.record("prog", "progress",    domain="pm")

        python_engine = create_engine("production", history=ctx.for_domain("python"))
        python_engine.suggest("prog")   # programming leads

        pm_engine = create_engine("production", history=ctx.for_domain("pm"))
        pm_engine.suggest("prog")   # progress leads
    """

    def __init__(self, *, default_domain: str = _DEFAULT_DOMAIN) -> None:
        self._default_domain = default_domain
        self._histories: dict[str, History] = {}

    def _get_or_create(self, domain: str | None) -> History:
        key = domain if domain is not None else self._default_domain
        if key not in self._histories:
            self._histories[key] = History()
        return self._histories[key]

    def record(self, prefix: str, value: str, *, domain: str | None = None) -> None:
        """
        Record a selection under the given domain.

        Parameters:
            prefix: The input prefix at the time of selection.
            value:  The completion that was selected.
            domain: Domain key.  If None, uses the default domain.
        """
        self._get_or_create(domain).record(prefix, value)

    def for_domain(self, domain: str | None = None) -> History:
        """
        Return the ``History`` for a specific domain.

        Creates the domain lazily if it does not exist yet.  The returned
        ``History`` is the live instance - mutations via ``record()`` on
        either the ``ContextualHistory`` or the returned ``History`` are
        immediately visible to both.

        Pass this to ``create_engine()`` or ``AutocompleteEngine()`` to
        build a domain-specific engine::

            engine = create_engine("production", history=ctx.for_domain("shell"))

        Parameters:
            domain: Domain key.  If None, uses the default domain.
        """
        return self._get_or_create(domain)

    def domains(self) -> Iterator[tuple[str, History]]:
        """
        Iterate over all domains and their History instances.

        Yields ``(domain_name, history)`` pairs.  Useful for persistence::

            for domain, hist in ctx.domains():
                JsonHistoryStore(path / f"{domain}.json").save(hist)
        """
        yield from self._histories.items()

    def domain_names(self) -> list[str]:
        """Return a sorted list of all domain names."""
        return sorted(self._histories.keys())

    def load_domain(self, domain: str, history: History) -> None:
        """
        Load a pre-populated ``History`` into the given domain.

        Use this when restoring persisted history from disk::

            store = JsonHistoryStore(path / f"{domain}.json")
            ctx.load_domain(domain, store.load())

        Overwrites any existing History for that domain.

        Parameters:
            domain:  Domain key.
            history: Pre-populated ``History`` instance.
        """
        self._histories[domain] = history

    def total_entries(self) -> int:
        """Total number of recorded entries across all domains. O(1)."""
        return sum(len(h) for h in self._histories.values())

    def __repr__(self) -> str:
        domain_counts = {d: len(h) for d, h in self._histories.items()}
        return f"ContextualHistory(domains={domain_counts})"
