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
    h = History()
    h.record("he", "hello", timestamp=_NOW)
    h.record("he", "hello", timestamp=_NOW)
    h.record("wo", "world", timestamp=_NOW)
    snap = h.snapshot()
    assert snap["he"]["hello"] == 2
    assert snap["wo"]["world"] == 1


def test_snapshot_does_not_include_timestamps() -> None:
    h = History()
    h.record("he", "hello", timestamp=_NOW)
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
