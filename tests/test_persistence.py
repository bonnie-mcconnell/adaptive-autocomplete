"""
Tests for JsonHistoryStore persistence.

Covers: round-trip correctness, timestamp preservation, v1 migration,
decay interaction after reload, and error handling.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aac.domain.history import History
from aac.storage.json_store import JsonHistoryStore

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


# ------------------------------------------------------------------
# Round-trip
# ------------------------------------------------------------------

def test_empty_history_round_trips(tmp_path: Path) -> None:
    store = JsonHistoryStore(tmp_path / "history.json")
    store.save(History())
    assert list(store.load().entries()) == []


def test_entries_survive_round_trip(tmp_path: Path) -> None:
    history = History()
    history.record("he", "hello", timestamp=_NOW)
    history.record("he", "hero", timestamp=_NOW - timedelta(hours=1))

    store = JsonHistoryStore(tmp_path / "history.json")
    store.save(history)
    loaded = store.load()

    assert len(list(loaded.entries())) == 2
    counts = loaded.counts_for_prefix("he")
    assert counts == {"hello": 1, "hero": 1}


def test_timestamps_survive_round_trip(tmp_path: Path) -> None:
    history = History()
    t1 = _NOW - timedelta(hours=2)
    t2 = _NOW - timedelta(minutes=30)
    history.record("he", "hello", timestamp=t1)
    history.record("he", "hero", timestamp=t2)

    store = JsonHistoryStore(tmp_path / "history.json")
    store.save(history)
    loaded = store.load()

    entries = {e.value: e.timestamp for e in loaded.entries()}
    assert abs((entries["hello"] - t1).total_seconds()) < 1
    assert abs((entries["hero"] - t2).total_seconds()) < 1


def test_multiple_prefixes_survive_round_trip(tmp_path: Path) -> None:
    history = History()
    history.record("he", "hello", timestamp=_NOW)
    history.record("wo", "world", timestamp=_NOW)
    history.record("pr", "problem", timestamp=_NOW)

    store = JsonHistoryStore(tmp_path / "history.json")
    store.save(history)
    loaded = store.load()

    assert loaded.counts_for_prefix("he") == {"hello": 1}
    assert loaded.counts_for_prefix("wo") == {"world": 1}
    assert loaded.counts_for_prefix("pr") == {"problem": 1}


def test_file_format_is_version_2(tmp_path: Path) -> None:
    history = History()
    history.record("he", "hello", timestamp=_NOW)

    path = tmp_path / "history.json"
    JsonHistoryStore(path).save(history)
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert raw["version"] == 2
    assert isinstance(raw["entries"], list)
    entry = raw["entries"][0]
    assert "prefix" in entry
    assert "value" in entry
    assert "timestamp" in entry


# ------------------------------------------------------------------
# v1 migration (count-only format)
# ------------------------------------------------------------------

def test_v1_format_loads_correctly(tmp_path: Path) -> None:
    path = tmp_path / "v1.json"
    path.write_text(
        json.dumps({"he": {"hello": 3, "hero": 1}}),
        encoding="utf-8",
    )
    loaded = JsonHistoryStore(path).load()
    assert loaded.counts_for_prefix("he") == {"hello": 3, "hero": 1}


def test_v1_entries_get_epoch_timestamps(tmp_path: Path) -> None:
    """Migrated entries are stamped at epoch so decay treats them as maximally stale."""
    path = tmp_path / "v1.json"
    path.write_text(json.dumps({"he": {"hello": 2}}), encoding="utf-8")
    loaded = JsonHistoryStore(path).load()
    for entry in loaded.entries():
        assert entry.timestamp == _EPOCH


def test_v1_epoch_timestamps_decay_to_near_zero() -> None:
    """
    Verifies that v1 migrated entries are treated as stale by DecayRanker.
    A selection from epoch (1970) with a 1-hour half-life decays to zero.
    """
    from aac.ranking.decay import DecayFunction
    decay = DecayFunction(half_life_seconds=3600)
    weight = decay.weight(now=_NOW, event_time=_EPOCH)
    assert weight < 1e-10, f"Expected near-zero weight for epoch timestamp, got {weight}"


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------

def test_missing_file_returns_empty_history(tmp_path: Path) -> None:
    store = JsonHistoryStore(tmp_path / "does_not_exist.json")
    assert list(store.load().entries()) == []


def test_corrupt_json_returns_empty_history(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text("this is not valid json {{{", encoding="utf-8")
    assert list(JsonHistoryStore(path).load().entries()) == []


def test_wrong_top_level_type_returns_empty_history(tmp_path: Path) -> None:
    path = tmp_path / "wrong_type.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert list(JsonHistoryStore(path).load().entries()) == []


def test_malformed_entries_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "malformed.json"
    payload = {
        "version": 2,
        "entries": [
            {"prefix": "he", "value": "hello", "timestamp": "2024-01-01T12:00:00+00:00"},
            {"prefix": "he"},                                            # missing fields
            "not a dict",                                                # wrong type
            {"prefix": "he", "value": "hero", "timestamp": "bad-date"}, # bad timestamp
            {"prefix": "he", "value": "help", "timestamp": "2024-01-01T11:00:00+00:00"},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = JsonHistoryStore(path).load()

    assert len(list(loaded.entries())) == 2
    counts = loaded.counts_for_prefix("he")
    assert "hello" in counts
    assert "help" in counts
    assert "hero" not in counts


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "history.json"
    store = JsonHistoryStore(nested)
    store.save(History())
    assert nested.exists()


# ------------------------------------------------------------------
# v1 migration: defensive branches
# ------------------------------------------------------------------

def test_v1_non_dict_values_skipped(tmp_path: Path) -> None:
    """Prefix entries that are not dicts are silently skipped in v1 format."""
    path = tmp_path / "v1.json"
    path.write_text(
        json.dumps({"he": ["not", "a", "dict"], "wo": {"world": 1}}),
        encoding="utf-8",
    )
    loaded = JsonHistoryStore(path).load()
    assert loaded.counts_for_prefix("wo") == {"world": 1}
    assert loaded.counts_for_prefix("he") == {}


def test_v1_non_integer_count_skipped(tmp_path: Path) -> None:
    """Non-integer counts in v1 format are silently skipped."""
    path = tmp_path / "v1.json"
    path.write_text(
        json.dumps({"he": {"hello": "not-a-number", "help": 2}}),
        encoding="utf-8",
    )
    loaded = JsonHistoryStore(path).load()
    counts = loaded.counts_for_prefix("he")
    assert counts.get("help") == 2
    assert "hello" not in counts


def test_v1_version_key_skipped(tmp_path: Path) -> None:
    """The literal key 'version' in v1 data must not be treated as a prefix."""
    path = tmp_path / "v1.json"
    path.write_text(
        json.dumps({"version": {"something": 1}, "he": {"hello": 1}}),
        encoding="utf-8",
    )
    loaded = JsonHistoryStore(path).load()
    assert loaded.counts_for_prefix("version") == {}
    assert loaded.counts_for_prefix("he") == {"hello": 1}

class TestDataLoadValidation:
    """load_english_frequencies raises on malformed data."""

    def test_non_dict_json_raises(self, tmp_path: Path) -> None:
        """If the JSON root is not a dict, load_english_frequencies raises ValueError."""
        import json
        from unittest.mock import patch

        from aac.data import load_english_frequencies

        bad_json = json.dumps([1, 2, 3])  # list, not dict
        with patch("aac.data._DATA_DIR", tmp_path):
            (tmp_path / "english_frequencies.json").write_text(bad_json, encoding="utf-8")
            with pytest.raises((ValueError, Exception)):
                load_english_frequencies.cache_clear()  # type: ignore[attr-defined]
                load_english_frequencies()


class TestJsonStoreExceptionCleanup:
    """JsonHistoryStore.save() must clean up the temp file if write fails."""

    def test_temp_file_cleaned_up_on_write_failure(self, tmp_path: Path) -> None:
        """If writing the temp file raises, no orphaned .tmp file is left.

        Patches ``aac.storage.json_store.os.fdopen`` rather than the global
        ``os.fdopen``.  Patching the global would intercept the call at the
        wrong point in the import chain and behave differently across
        platforms (on Windows the fd is already open before fdopen is called,
        leaving an orphaned temp file).  Patching the name as imported by the
        module under test guarantees the exception fires at the right point
        regardless of platform.
        """
        from unittest.mock import patch

        from aac.domain.history import History
        from aac.storage.json_store import JsonHistoryStore

        store = JsonHistoryStore(tmp_path / "history.json")
        history = History()
        history.record("he", "hello")

        # Patch os.fdopen as imported by json_store, not the global os module.
        # This is the call that actually opens the fd for writing; raising here
        # exercises the except/unlink cleanup path on every platform.
        with patch("aac.storage.json_store.os.fdopen", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                store.save(history)

        # No orphaned temp files should remain
        tmp_files = list(tmp_path.glob(".aac_history_*.tmp"))
        assert tmp_files == [], f"Orphaned temp files: {tmp_files}"
