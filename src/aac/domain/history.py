from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class HistoryEntry:
    """
    A single observed completion event.

    Attributes:
        prefix: The user input prefix at the time of selection.
        value: The completion value selected by the user.
        timestamp: When the selection occurred (UTC, timezone-aware).

    Notes:
        - Entries are immutable once created.
        - Timestamps are always stored in UTC.
        - Enables future extensions such as recency decay, session analysis,
          or time-windowed learning strategies.
    """
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
    Append-only store of user completion events.

    This is the single source of truth for all learning signals.

    Design guarantees:
        - Entries are immutable once recorded
        - No deletion or in-place mutation
        - Safe to share across predictors and rankers
        - Persistence-friendly via explicit snapshot export

    This class intentionally separates:
        - In-memory domain representation (HistoryEntry)
        - Serialized representation (snapshot)
    """

    def __init__(self) -> None:
        self._entries: list[HistoryEntry] = []

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
        """
        Record a completion selection.

        Parameters:
            prefix: The user input prefix.
            value: The completion selected by the user.
            timestamp: Optional explicit timestamp. If omitted,
                       the current UTC time is used.

        Notes:
            - This operation is append-only.
            - Callers should treat History as write-once per event.
            - Prefix and value are coerced to strings to enforce
              persistence and serialization invariants.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        if timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")

        self._entries.append(
            HistoryEntry(
                prefix=str(prefix),
                value=str(value),
                timestamp=timestamp,
            )
        )

    # ------------------------------------------------------------
    # Read APIs
    # ------------------------------------------------------------

    def entries(self) -> Sequence[HistoryEntry]:
        """
        Immutable view of all recorded history entries.

        Returns:
            A tuple of HistoryEntry objects in insertion order.
        """
        return tuple(self._entries)

    def entries_for_prefix(self, prefix: str) -> Sequence[HistoryEntry]:
        """
        Return all history entries matching a given prefix.

        Parameters:
            prefix: The prefix to filter by.

        Returns:
            A tuple of HistoryEntry objects.
        """
        prefix = str(prefix)
        return tuple(
            e for e in self._entries
            if e.prefix == prefix
        )

    def counts_for_prefix(self, prefix: str) -> dict[str, int]:
        """
        Count how often each value was selected for a given prefix.

        This is a backwards-compatible, time-agnostic API intended
        for simple learning strategies.

        Parameters:
            prefix: The prefix to aggregate counts for.

        Returns:
            Mapping of completion value -> selection count.
        """
        prefix = str(prefix)
        counts: dict[str, int] = defaultdict(int)

        for e in self._entries:
            if e.prefix == prefix:
                counts[e.value] += 1

        return dict(counts)

    def counts_for_prefix_since(
        self,
        prefix: str,
        since: datetime,
    ) -> dict[str, int]:
        """
        Count selections for a prefix occurring at or after a timestamp.

        Enables recency-aware or time-decayed learning strategies.

        Parameters:
            prefix: The prefix to aggregate counts for.
            since: Lower bound (inclusive) for entry timestamps.

        Returns:
            Mapping of completion value -> selection count.
        """
        if since.tzinfo is None:
            raise ValueError("since must be timezone-aware")

        prefix = str(prefix)
        counts: dict[str, int] = defaultdict(int)

        for e in self._entries:
            if e.prefix != prefix:
                continue
            if e.timestamp < since:
                continue
            counts[e.value] += 1

        return dict(counts)

    def count(self, value: str) -> int:
        """
        Total count for a specific value across all prefixes.

        Parameters:
            value: Completion value to count.

        Returns:
            Number of times the value was selected.
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
        Return a count-only snapshot of history data.

        Format:
            {
                "<prefix>": {
                    "<value>": count
                }
            }

        Notes:
            - Timestamps are omitted. This format is the v1 storage
              schema and is kept for inspection and backwards compatibility.
              For full persistence including timestamps, use JsonHistoryStore
              which writes the v2 format.
        """
        snapshot: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for e in self._entries:
            snapshot[e.prefix][e.value] += 1

        return {
            prefix: dict(values)
            for prefix, values in snapshot.items()
        }