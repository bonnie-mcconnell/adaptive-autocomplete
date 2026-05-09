"""
Thread-safe wrapper around History for concurrent server environments.

Usage::

    from aac.domain.thread_safe_history import ThreadSafeHistory
    from aac.storage.json_store import JsonHistoryStore
    from aac.presets import create_engine

    store = JsonHistoryStore(path)
    history = ThreadSafeHistory(store.load())
    engine = create_engine("production", history=history)

    # Safe to call from multiple threads simultaneously:
    engine.record_selection("prog", "programming")
    store.save(history.snapshot_history())
"""
from __future__ import annotations

import threading
from collections.abc import Sequence
from datetime import datetime

from aac.domain.history import History, HistoryEntry


class ThreadSafeHistory(History):
    """
    A History subclass that is safe for concurrent reads and writes.

    Uses a ``threading.RWLock``-style pattern: a ``threading.Condition``
    for write serialisation, reference-counted readers for read concurrency.
    Multiple readers run simultaneously; writes wait for active readers to
    finish. Matches the engine's access pattern: many concurrent predict()
    reads, occasional record_selection() writes.

    Concurrency guarantees:
        - Concurrent ``record()`` calls are fully serialised; no corruption.
        - Read methods run concurrently with each other and see a
          consistent, fully-committed view.
        - Writes wait for active reads; reads wait while a write is pending.

    Works on CPython, PyPy, and free-threaded CPython (PEP 703 / 3.13+).
    Does not rely on GIL guarantees.

    Writer starvation: under sustained heavy read load a writer may wait
    for a quiet moment. In practice writes are rare (one per selection),
    so this isn't a concern. If it ever is, switch to a fair queued lock.
    """

    def __init__(self, source: History | None = None) -> None:
        """
        Create a ThreadSafeHistory, optionally pre-populated from an
        existing History (e.g. loaded from JsonHistoryStore).

        Parameters:
            source: Existing History to copy entries from.  If None,
                    starts empty.
        """
        super().__init__()

        # Condition used for both read and write coordination.
        # _readers tracks the number of active concurrent readers.
        # _write_pending is True while a writer is waiting or writing.
        self._lock = threading.Condition(threading.Lock())
        self._readers = 0
        self._write_pending = False

        if source is not None:
            for entry in source.entries():
                # Bypass locking during construction - no other threads
                # can reference this object yet.
                super().record(
                    entry.prefix,
                    entry.value,
                    timestamp=entry.timestamp,
                )

    # ------------------------------------------------------------------
    # Read-lock helpers
    # ------------------------------------------------------------------

    def _acquire_read(self) -> None:
        with self._lock:
            # Wait until no writer is active or pending.
            while self._write_pending:
                self._lock.wait()
            self._readers += 1

    def _release_read(self) -> None:
        with self._lock:
            self._readers -= 1
            if self._readers == 0:
                self._lock.notify_all()  # wake any waiting writer

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def record(
        self,
        prefix: str,
        value: str,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Thread-safe record(). Waits for active readers before writing."""
        with self._lock:
            self._write_pending = True
            # Wait until all active readers have finished.
            while self._readers > 0:
                self._lock.wait()
            try:
                super().record(prefix, value, timestamp=timestamp)
            finally:
                self._write_pending = False
                self._lock.notify_all()  # wake waiting readers

    # ------------------------------------------------------------------
    # Read paths (all guarded by read-lock)
    # ------------------------------------------------------------------

    def entries(self) -> Sequence[HistoryEntry]:
        self._acquire_read()
        try:
            return super().entries()
        finally:
            self._release_read()

    def entries_for_prefix(self, prefix: str) -> Sequence[HistoryEntry]:
        self._acquire_read()
        try:
            return super().entries_for_prefix(prefix)
        finally:
            self._release_read()

    def counts_for_prefix(self, prefix: str) -> dict[str, int]:
        self._acquire_read()
        try:
            return super().counts_for_prefix(prefix)
        finally:
            self._release_read()

    def counts_for_prefix_since(
        self,
        prefix: str,
        since: datetime,
    ) -> dict[str, int]:
        self._acquire_read()
        try:
            return super().counts_for_prefix_since(prefix, since)
        finally:
            self._release_read()

    def count(self, value: str) -> int:
        self._acquire_read()
        try:
            return super().count(value)
        finally:
            self._release_read()

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Deprecated. Thread-safe wrapper; use snapshot_counts() instead."""
        import warnings
        warnings.warn(
            "ThreadSafeHistory.snapshot() is deprecated. "
            "Use snapshot_counts() for inspection, or JsonHistoryStore.save() for persistence.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._acquire_read()
        try:
            return super().snapshot_counts()  # bypass super().snapshot() to avoid double-warn
        finally:
            self._release_read()

    def snapshot_counts(self) -> dict[str, dict[str, int]]:
        """Thread-safe snapshot_counts(). Returns a deep copy under a read lock."""
        self._acquire_read()
        try:
            return super().snapshot_counts()
        finally:
            self._release_read()

    def copy(self) -> History:
        """Thread-safe copy(). Returns a new non-thread-safe History snapshot."""
        self._acquire_read()
        try:
            return super().copy()
        finally:
            self._release_read()

    # ------------------------------------------------------------------
    # Snapshot for persistence
    # ------------------------------------------------------------------

    def snapshot_history(self) -> History:
        """
        Return a plain (non-thread-safe) snapshot of the current history.

        Takes a consistent point-in-time copy under a read lock.  Entries
        recorded after snapshot_history() returns are not included.

        Useful for passing to ``JsonHistoryStore.save()`` without holding
        a lock during I/O::

            store.save(ts_history.snapshot_history())
        """
        self._acquire_read()
        try:
            snap = History()
            for entry in super().entries():
                snap.record(entry.prefix, entry.value, timestamp=entry.timestamp)
            return snap
        finally:
            self._release_read()

    # ------------------------------------------------------------------
    # Expose the write lock for compound atomic operations
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        self._acquire_read()
        try:
            n = len(self._entries)
        finally:
            self._release_read()
        return f"ThreadSafeHistory(entries={n})"

    @property
    def lock(self) -> threading.Condition:
        """
        The underlying coordination lock.

        This is an advanced escape hatch for compound atomic operations.
        In normal use you should never need it - ``record()`` and all read
        methods are already individually thread-safe.

        The only correct use case is a compound read-then-write that must
        be atomic - for example, capping the number of recordings for a
        prefix.  The pattern requires calling the *internal* (un-locked)
        methods via ``super()``, because calling the public locked methods
        while holding ``lock`` will deadlock::

            # WRONG - deadlocks: counts_for_prefix() tries to acquire
            # a read-lock while the write-lock is already held.
            with history.lock:
                count = history.counts_for_prefix("he")   # DEADLOCK

            # CORRECT - use a subclass or internal access for the read,
            # then call the public record() after releasing the lock.
            # In practice this pattern is rarely needed; if you find
            # yourself reaching for it, consider whether a higher-level
            # lock in your application code is a cleaner solution.

        Warning: acquiring ``lock`` while holding a read-lock will deadlock.
        Only use this for write-side compound operations via internal methods.
        """
        return self._lock
