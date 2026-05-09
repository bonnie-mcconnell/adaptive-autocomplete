"""
Tests for the History domain object.

Covers: recording, read APIs, timestamp validation,
time-bounded queries, snapshot format, and immutability guarantees.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aac.domain.history import History, HistoryEntry

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_HOUR = timedelta(hours=1)


# ------------------------------------------------------------------
# HistoryEntry construction
# ------------------------------------------------------------------

def test_history_entry_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        HistoryEntry(prefix="he", value="hello", timestamp=datetime(2024, 1, 1))


def test_history_entry_accepts_aware_timestamp() -> None:
    entry = HistoryEntry(prefix="he", value="hello", timestamp=_NOW)
    assert entry.timestamp.tzinfo is not None


# ------------------------------------------------------------------
# Recording
# ------------------------------------------------------------------

def test_record_stores_entry() -> None:
    h = History()
    h.record("he", "hello", timestamp=_NOW)
    assert len(list(h.entries())) == 1


def test_record_auto_timestamps_in_utc() -> None:
    h = History()
    h.record("he", "hello")
    entry = list(h.entries())[0]
    assert entry.timestamp.tzinfo is not None


def test_record_rejects_naive_explicit_timestamp() -> None:
    h = History()
    with pytest.raises(ValueError):
        h.record("he", "hello", timestamp=datetime(2024, 1, 1))


def test_multiple_records_preserved_in_order() -> None:
    h = History()
    h.record("he", "hello", timestamp=_NOW)
    h.record("he", "hero", timestamp=_NOW + _HOUR)
    h.record("wo", "world", timestamp=_NOW + 2 * _HOUR)
    entries = list(h.entries())
    assert [e.value for e in entries] == ["hello", "hero", "world"]


# ------------------------------------------------------------------
# entries_for_prefix
# ------------------------------------------------------------------

def test_entries_for_prefix_filters_correctly() -> None:
    h = History()
    h.record("he", "hello", timestamp=_NOW)
    h.record("wo", "world", timestamp=_NOW)
    h.record("he", "hero", timestamp=_NOW)
    result = list(h.entries_for_prefix("he"))
    assert len(result) == 2
    assert all(e.prefix == "he" for e in result)


def test_entries_for_prefix_empty_when_no_match() -> None:
    h = History()
    h.record("he", "hello", timestamp=_NOW)
    assert list(h.entries_for_prefix("wo")) == []


# ------------------------------------------------------------------
# counts_for_prefix
# ------------------------------------------------------------------

def test_counts_for_prefix_aggregates_correctly() -> None:
    h = History()
    h.record("he", "hello", timestamp=_NOW)
    h.record("he", "hello", timestamp=_NOW)
    h.record("he", "hero", timestamp=_NOW)
    counts = h.counts_for_prefix("he")
    assert counts == {"hello": 2, "hero": 1}


# ------------------------------------------------------------------
# counts_for_prefix_since
# ------------------------------------------------------------------

def test_counts_for_prefix_since_filters_by_time() -> None:
    h = History()
    cutoff = _NOW
    h.record("he", "old", timestamp=_NOW - _HOUR)
    h.record("he", "recent", timestamp=_NOW + _HOUR)
    counts = h.counts_for_prefix_since("he", since=cutoff)
    assert "recent" in counts
    assert "old" not in counts


def test_counts_for_prefix_since_rejects_naive_timestamp() -> None:
    h = History()
    with pytest.raises(ValueError):
        h.counts_for_prefix_since("he", since=datetime(2024, 1, 1))


# ------------------------------------------------------------------
# count (global, across prefixes)
# ------------------------------------------------------------------

def test_count_totals_across_prefixes() -> None:
    h = History()
    h.record("he", "hello", timestamp=_NOW)
    h.record("hel", "hello", timestamp=_NOW)
    h.record("he", "hero", timestamp=_NOW)
    assert h.count("hello") == 2
    assert h.count("hero") == 1
    assert h.count("world") == 0


# ------------------------------------------------------------------
# snapshot
# ------------------------------------------------------------------

def test_snapshot_format() -> None:
    import pytest
    h = History()
    h.record("he", "hello", timestamp=_NOW)
    h.record("he", "hello", timestamp=_NOW)
    h.record("wo", "world", timestamp=_NOW)
    with pytest.warns(DeprecationWarning, match="snapshot_counts"):
        snap = h.snapshot()
    assert snap["he"]["hello"] == 2
    assert snap["wo"]["world"] == 1


def test_snapshot_does_not_include_timestamps() -> None:
    import pytest
    h = History()
    h.record("he", "hello", timestamp=_NOW)
    with pytest.warns(DeprecationWarning):
        snap = h.snapshot()
    assert isinstance(snap["he"]["hello"], int)


# ------------------------------------------------------------------
# Immutability guarantees
# ------------------------------------------------------------------

def test_entries_returns_tuple() -> None:
    h = History()
    h.record("he", "hello", timestamp=_NOW)
    result = h.entries()
    assert isinstance(result, tuple)


def test_modifying_returned_entries_does_not_affect_history() -> None:
    h = History()
    h.record("he", "hello", timestamp=_NOW)
    entries = list(h.entries())
    entries.clear()
    assert len(list(h.entries())) == 1


def test_counts_for_prefix_since_skips_non_matching_prefix() -> None:
    """Entries for other prefixes must be ignored even if timestamp qualifies."""
    h = History()
    h.record("wo", "world", timestamp=_NOW + _HOUR)
    counts = h.counts_for_prefix_since("he", since=_NOW)
    assert counts == {}

# ------------------------------------------------------------------
# Prefix index correctness
# ------------------------------------------------------------------

def test_prefix_index_returns_same_as_full_scan() -> None:
    """entries_for_prefix via index must match a brute-force full scan."""
    h = History()
    h.record("he", "hello")
    h.record("he", "hero")
    h.record("sh", "shell")
    h.record("he", "help")

    indexed = list(h.entries_for_prefix("he"))
    brute = [e for e in h.entries() if e.prefix == "he"]

    assert len(indexed) == len(brute)
    assert all(a == b for a, b in zip(indexed, brute, strict=False))


def test_counts_for_prefix_via_index() -> None:
    """counts_for_prefix must aggregate correctly from the prefix index."""
    h = History()
    for _ in range(3):
        h.record("he", "hello")
    for _ in range(2):
        h.record("he", "hero")
    h.record("sh", "shell")

    counts = h.counts_for_prefix("he")
    assert counts == {"hello": 3, "hero": 2}
    assert "sh" not in counts


def test_prefix_index_empty_prefix_returns_empty() -> None:
    h = History()
    h.record("he", "hello")
    assert list(h.entries_for_prefix("xx")) == []
    assert h.counts_for_prefix("xx") == {}


def test_prefix_index_consistent_after_many_records() -> None:
    """Index stays consistent across a large number of mixed-prefix records."""
    from datetime import datetime, timezone
    h = History()
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

    prefixes = ["a", "ab", "b", "c"]
    for i in range(200):
        p = prefixes[i % len(prefixes)]
        h.record(p, f"word{i}", timestamp=ts)

    for p in prefixes:
        indexed = list(h.entries_for_prefix(p))
        brute = [e for e in h.entries() if e.prefix == p]
        assert indexed == brute, f"Index mismatch for prefix {p!r}"


# ---------------------------------------------------------------------------
# History.copy()
# ---------------------------------------------------------------------------

class TestHistoryCopy:
    def test_copy_is_independent(self) -> None:
        h = History()
        h.record("he", "hello")
        h2 = h.copy()

        h2.record("he", "help")  # modify copy
        assert len(list(h.entries())) == 1, "original must be unaffected"
        assert len(list(h2.entries())) == 2, "copy should have both entries"

    def test_copy_has_same_entries(self) -> None:
        h = History()
        h.record("prog", "programming")
        h.record("prog", "program")
        h2 = h.copy()
        assert set(e.value for e in h2.entries()) == {"programming", "program"}

    def test_copy_prefix_index_works(self) -> None:
        h = History()
        h.record("prog", "programming")
        h2 = h.copy()
        assert h2.counts_for_prefix("prog").get("programming") == 1

    def test_copy_of_empty_history_is_empty(self) -> None:
        h2 = History().copy()
        assert len(list(h2.entries())) == 0

    def test_original_unaffected_by_copy_mutations(self) -> None:
        h = History()
        h.record("he", "her")
        h2 = h.copy()
        for _ in range(5):
            h2.record("he", "help")

        assert h.counts_for_prefix("he").get("help", 0) == 0, (
            "Recording into the copy must not affect the original's counts"
        )


# ---------------------------------------------------------------------------
# History.__repr__
# ---------------------------------------------------------------------------

class TestHistoryRepr:
    def test_empty_history_repr(self) -> None:
        h = History()
        assert repr(h) == "History(entries=0, prefixes=0)"

    def test_repr_reflects_entry_and_prefix_counts(self) -> None:
        h = History()
        h.record("prog", "programming", timestamp=_NOW)
        h.record("prog", "program", timestamp=_NOW)
        h.record("hel", "hello", timestamp=_NOW)
        assert repr(h) == "History(entries=3, prefixes=2)"

    def test_repr_same_prefix_multiple_records(self) -> None:
        h = History()
        for _ in range(5):
            h.record("he", "hello", timestamp=_NOW)
        # 5 entries, 1 distinct prefix
        assert repr(h) == "History(entries=5, prefixes=1)"

    def test_repr_does_not_mutate(self) -> None:
        """Calling repr() must not change the history state."""
        h = History()
        h.record("a", "apple", timestamp=_NOW)
        _ = repr(h)
        assert len(h.entries()) == 1
