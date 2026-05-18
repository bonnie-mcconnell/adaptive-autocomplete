"""JsonHistoryStore: persists History to a JSON file. Thread-safe with atomic rename-on-write."""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from aac.domain.history import History
from aac.storage.base import HistoryStore

_log = logging.getLogger(__name__)

# Storage format version. Increment when the schema changes in a
# backwards-incompatible way and add a migration branch in _load_data().
_CURRENT_VERSION = 2


class JsonHistoryStore(HistoryStore):
    """
    JSON-backed persistence for History.

    Stores full entries with timestamps (v2 format) so DecayRanker works
    correctly after restarts. v1 files (count-only) are migrated on load
    with epoch timestamps - they contribute counts but no recency signal.
    save() uses atomic rename on POSIX; best-effort backup rotation on Windows.
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
        except json.JSONDecodeError as e:
            # Corrupted file (partial write or external corruption). Start fresh.
            _log.warning(
                "History file %s is not valid JSON (%s). "
                "Starting with empty history. Rename or delete the file to silence this.",
                self._path, e,
            )
            return History()
        except OSError as e:
            # File exists but can't be read (permissions or lock issue).
            _log.warning(
                "Cannot read history file %s (%s). Starting with empty history.",
                self._path, e,
            )
            return History()

        return _load_data(data)

    def save(self, history: History) -> None:
        """
        Atomically persist all history entries to disk.

        Writes to a temp file in the same directory (same filesystem,
        so rename never crosses a device boundary), then renames it over
        the target.

        On POSIX, ``rename()`` is atomic - a reader sees either the old
        file or the new one, never a partial write.

        On Windows, ``Path.replace()`` is a delete-then-rename and is not
        atomic.  To mitigate crash risk, the previous file is first moved
        to ``<path>.bak``, then the temp file is renamed into place.  If a
        crash occurs between those two steps the ``.bak`` file contains the
        previous history.  The loader handles a missing target file by
        returning an empty ``History``.

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

        content = json.dumps(payload, indent=2, sort_keys=True)

        # Write to a temp file in the same directory so the rename is
        # guaranteed to be on the same filesystem (cross-device rename fails).
        fd, tmp_path_str = tempfile.mkstemp(
            dir=self._path.parent,
            prefix=".aac_history_",
            suffix=".tmp",
        )
        # Track whether os.fdopen() has taken ownership of fd.
        # If fdopen raises, fd is still open (on Windows an open file cannot
        # be unlinked). We must close it explicitly before calling unlink.
        fd_owned_by_file = False
        tmp_path = Path(tmp_path_str)

        # Windows backup rotation.
        # Path.replace() is not atomic on Windows when the destination exists:
        # it deletes the target then renames the temp file, leaving a window
        # where a crash produces no file at the target path.  We mitigate
        # this by keeping the previous file as a .bak alongside the target.
        # On POSIX, rename() is atomic and no backup is needed; we skip the
        # rotation to avoid the extra syscall.
        backup_path: Path | None = None
        if sys.platform == "win32" and self._path.exists():  # pragma: no cover
            # Windows-only: Path.replace() is not atomic when the destination
            # exists. Keep the old file as .bak so a crash between the delete
            # and rename leaves the .bak as a recovery option.
            # Tested by the Windows CI matrix; unreachable on Linux runners.
            backup_path = self._path.with_suffix(".bak")
            try:
                self._path.replace(backup_path)
            except OSError:
                backup_path = None

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                fd_owned_by_file = True   # f now owns fd; closing f closes it
                f.write(content)
            # POSIX: atomic. Windows: best-effort (backup kept as fallback).
            tmp_path.replace(self._path)
            # Replace succeeded - remove stale backup if present.
            if backup_path is not None:  # pragma: no cover
                # Only set on win32 (see block above).
                try:
                    backup_path.unlink()
                except OSError:
                    pass
        except Exception:
            # Catch-all here is intentional: we need to clean up the temp
            # file regardless of what went wrong (OSError, PermissionError,
            # interrupted write, etc.). After cleanup we always re-raise so
            # the caller sees the original exception unmodified.
            _log.debug("save() failed; removing temp file %s", tmp_path_str, exc_info=True)
            if not fd_owned_by_file:
                # os.fdopen() raised before taking ownership of fd.
                # The fd is still open; close it before unlink so that
                # Windows (mandatory file locking) can delete it.
                try:
                    os.close(fd)
                except OSError:  # pragma: no cover
                    # os.close() failing is pathological (invalid fd, EINTR).
                    # Can't be triggered reliably in a test without OS mocking.
                    pass
            try:
                os.unlink(tmp_path_str)
            except OSError:  # pragma: no cover
                # Temp file already gone (race with external cleanup) or
                # unlink permission denied. Nothing we can do; re-raise below.
                pass
            raise


# ------------------------------------------------------------------
# Internal loading helpers
# ------------------------------------------------------------------

def _load_data(data: object) -> History:
    """Route to the appropriate loader based on the file format version."""
    if not isinstance(data, dict):
        return History()

    version = data.get("version", 1)

    if version == 2:
        return _load_v2(data)

    # Any unrecognised version without an explicit 'entries' key is
    # treated as the original count-only format (version 1).
    return _load_v1(data)


def _load_v2(data: dict[str, object]) -> History:
    """Load v2 format: full entries with timestamps."""
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

        # Ensure timezone-aware - fromisoformat on Python 3.10 may
        # return naive datetimes for strings without offset info.
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        history.record(prefix, value, timestamp=ts)

    return history


def _load_v1(data: dict[str, object]) -> History:
    """Load v1 count-only format. Timestamps set to epoch so decay treats entries as stale."""
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