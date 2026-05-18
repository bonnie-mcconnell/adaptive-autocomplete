"""
ContextualHistory: domain-partitioned learning for multi-context autocomplete.

Each domain (e.g. "shell", "search", "python") gets its own History instance.
Pass ctx.for_domain("shell") to create_engine() to build a domain-specific engine.
Domains are created lazily; no upfront registration needed.
"""
from __future__ import annotations

from collections.abc import Iterator

from aac.domain.history import History

_DEFAULT_DOMAIN = "__default__"


class ContextualHistory:
    """Domain-partitioned history. Each domain is an independent History instance."""

    def __init__(self, *, default_domain: str = _DEFAULT_DOMAIN) -> None:
        self._default_domain = default_domain
        self._histories: dict[str, History] = {}

    def _get_or_create(self, domain: str | None) -> History:
        key = domain if domain is not None else self._default_domain
        if key not in self._histories:
            self._histories[key] = History()
        return self._histories[key]

    def record(self, prefix: str, value: str, *, domain: str | None = None) -> None:
        """Record a selection. domain=None uses the default domain."""
        self._get_or_create(domain).record(prefix, value)

    def for_domain(self, domain: str | None = None) -> History:
        """Return the History for a domain, creating it lazily. domain=None uses the default."""
        return self._get_or_create(domain)

    def domains(self) -> Iterator[tuple[str, History]]:
        """Yield (domain_name, history) pairs for all domains."""
        yield from self._histories.items()

    def domain_names(self) -> list[str]:
        """Return a sorted list of all domain names."""
        return sorted(self._histories.keys())

    def load_domain(self, domain: str, history: History) -> None:
        """Load a pre-populated History into a domain. Overwrites any existing entry."""
        self._histories[domain] = history

    def total_entries(self) -> int:
        """Total recorded entries across all domains."""
        return sum(len(h) for h in self._histories.values())

    def __repr__(self) -> str:
        domain_counts = {d: len(h) for d, h in self._histories.items()}
        return f"ContextualHistory(domains={domain_counts})"
