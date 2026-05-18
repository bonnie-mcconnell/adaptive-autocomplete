"""Append-only selection history: records (prefix, value, timestamp) triples and exposes prefix-keyed counts."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class HistoryEntry:
    """A single completion selection: (prefix, value, timestamp)."""
    prefix: str
    value: str
    timestamp: datetime

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(
                "HistoryEntry.timestamp must be timezone-aware (UTC)"
            )


class History:
    """
    Append-only store of user completion events - the shared learning state.

    Entries are immutable once recorded; predictors and rankers can safely
    read concurrently. For concurrent writes, use ThreadSafeHistory.

    A prefix index keeps prefix-scoped reads O(k) rather than O(n).
    Cost is one extra dict write per record() call.
    """

    def __init__(self) -> None:
        self._entries: list[HistoryEntry] = []
        # Prefix index: maps prefix -> entries with that prefix, in
        # insertion order. Maintained incrementally on every record().
        self._by_prefix: defaultdict[str, list[HistoryEntry]] = defaultdict(list)

    # ------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------

    def record(
        self,
        prefix: str,
        value: str,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a completion selection. timestamp defaults to now (UTC)."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        if timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")

        entry = HistoryEntry(
            prefix=str(prefix),
            value=str(value),
            timestamp=timestamp,
        )
        self._entries.append(entry)
        # Maintain prefix index incrementally. One dict write per record(),
        # zero overhead on reads.
        self._by_prefix[entry.prefix].append(entry)

    # ------------------------------------------------------------
    # Read APIs
    # ------------------------------------------------------------

    def __len__(self) -> int:
        """Number of recorded entries. O(1)."""
        return len(self._entries)

    def entries(self) -> Sequence[HistoryEntry]:
        """All recorded entries in insertion order."""
        return tuple(self._entries)

    def entries_for_prefix(self, prefix: str) -> Sequence[HistoryEntry]:
        """Return all history entries for a prefix. O(k) via the prefix index."""
        prefix = str(prefix)
        return tuple(self._by_prefix.get(prefix, []))

    def counts_for_prefix(self, prefix: str) -> dict[str, int]:
        """Selection counts per value for a given prefix. O(k) via the prefix index."""
        prefix = str(prefix)
        counts: dict[str, int] = defaultdict(int)

        for e in self._by_prefix.get(prefix, []):
            counts[e.value] += 1

        return dict(counts)

    def counts_for_prefix_since(
        self,
        prefix: str,
        since: datetime,
    ) -> dict[str, int]:
        """Like counts_for_prefix but filtered to entries at or after since."""
        if since.tzinfo is None:
            raise ValueError("since must be timezone-aware")

        prefix = str(prefix)
        counts: dict[str, int] = defaultdict(int)

        for e in self._by_prefix.get(prefix, []):
            if e.timestamp >= since:
                counts[e.value] += 1

        return dict(counts)

    def count(self, value: str) -> int:
        """
        Total selections for a value across all prefixes. O(n) full scan.

        If you know the prefix (you usually do), prefer:
            counts_for_prefix(prefix).get(value, 0)

        Not used internally - intended for diagnostics and tests.
        """
        value = str(value)
        return sum(
            1 for e in self._entries
            if e.value == value
        )

    # ------------------------------------------------------------
    # Persistence boundary
    # ------------------------------------------------------------

    def snapshot(self) -> dict[str, dict[str, int]]:
        """
        Deprecated. Use snapshot_counts() for inspection or JsonHistoryStore
        for persistence (snapshot() omits timestamps, breaking DecayRanker).
        """
        import warnings
        warnings.warn(
            "History.snapshot() is deprecated and will be removed in a future version. "
            "Use snapshot_counts() for inspection, or JsonHistoryStore.save() for persistence.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.snapshot_counts()

    def snapshot_counts(self) -> dict[str, dict[str, int]]:
        """
        Count-only view of history: {prefix: {value: count}}.

        Do not use for persistence - timestamps are omitted, which breaks
        DecayRanker on reload. Use JsonHistoryStore.save() instead.
        """
        snapshot: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for e in self._entries:
            snapshot[e.prefix][e.value] += 1

        return {
            prefix: dict(values)
            for prefix, values in snapshot.items()
        }

    def __repr__(self) -> str:
        """History(entries=3, prefixes=2)"""
        return (
            f"History("
            f"entries={len(self._entries)}, "
            f"prefixes={len(self._by_prefix)})"
        )

    def copy(self) -> History:
        """Return an independent copy. Modifications to either do not affect the other."""
        new_history = History()
        for entry in self._entries:
            new_history.record(entry.prefix, entry.value, timestamp=entry.timestamp)
        return new_history