from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from aac.domain.history import History
from aac.storage.base import HistoryStore

# Storage format version. Increment when the schema changes in a
# backwards-incompatible way and add a migration branch in _load_data().
_CURRENT_VERSION = 2


class JsonHistoryStore(HistoryStore):
    """
    JSON-backed persistence for History.

    Persists full HistoryEntry objects including timestamps, so that
    recency-aware rankers (DecayRanker) continue to work correctly
    after a process restart.

    Format (version 2):
        {
          "version": 2,
          "entries": [
            {
              "prefix": "he",
              "value": "hero",
              "timestamp": "2024-01-15T09:32:11+00:00"
            },
            ...
          ]
        }

    Migration:
        Version 1 files (the old count-only format) are loaded with
        timestamps set to the Unix epoch. This means all migrated
        entries are treated as maximally stale by decay-based rankers —
        they contribute counts but carry no recency signal. This is the
        safest migration: old data can only boost, never mislead.

    Design notes:
        - Domain objects remain I/O-free; all serialisation lives here.
        - Malformed entries are skipped, not fatal.
        - Atomic write (write to temp file, rename) is not implemented
          but would be the correct production approach.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> History:
        """
        Load history from disk.

        Returns an empty History if the file does not exist.
        Malformed entries are skipped; a partially corrupt file
        returns whatever entries were valid.
        """
        if not self._path.exists():
            return History()

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return History()

        return _load_data(data)

    def save(self, history: History) -> None:
        """
        Persist all history entries to disk, including timestamps.

        Creates parent directories if they do not exist.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)

        entries = [
            {
                "prefix": entry.prefix,
                "value": entry.value,
                "timestamp": entry.timestamp.isoformat(),
            }
            for entry in history.entries()
        ]

        payload = {
            "version": _CURRENT_VERSION,
            "entries": entries,
        }

        self._path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


# ------------------------------------------------------------------
# Internal loading helpers
# ------------------------------------------------------------------

def _load_data(data: object) -> History:
    """
    Dispatch to the appropriate loader based on format version.
    """
    if not isinstance(data, dict):
        return History()

    version = data.get("version", 1)

    if version == 2:
        return _load_v2(data)

    # Any unrecognised version without an explicit 'entries' key is
    # treated as the original count-only format (version 1).
    return _load_v1(data)


def _load_v2(data: dict[str, object]) -> History:
    """
    Load format version 2: full entries with timestamps.
    """
    history = History()
    raw_entries = data.get("entries", [])

    if not isinstance(raw_entries, list):
        return history

    for item in raw_entries:
        if not isinstance(item, dict):
            continue

        prefix = item.get("prefix")
        value = item.get("value")
        raw_ts = item.get("timestamp")

        if not isinstance(prefix, str) or not isinstance(value, str):
            continue
        if not isinstance(raw_ts, str):
            continue

        try:
            ts = datetime.fromisoformat(raw_ts)
        except ValueError:
            continue

        # Ensure timezone-aware — fromisoformat on Python 3.10 may
        # return naive datetimes for strings without offset info.
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        history.record(prefix, value, timestamp=ts)

    return history


def _load_v1(data: dict[str, object]) -> History:
    """
    Load format version 1: count-only {prefix: {value: count}}.

    Timestamps are set to the Unix epoch so that all migrated entries
    are treated as maximally stale by decay-based rankers. They still
    contribute to count-based ranking but carry no recency signal.
    """
    history = History()
    _EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

    for prefix, values in data.items():
        if prefix == "version":
            continue
        if not isinstance(values, dict):
            continue

        prefix_str = str(prefix)

        for value, count in values.items():
            value_str = str(value)

            try:
                count_int = int(count)
            except (TypeError, ValueError):
                continue

            for _ in range(count_int):
                history.record(prefix_str, value_str, timestamp=_EPOCH)

    return history