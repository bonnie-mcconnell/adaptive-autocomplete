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

    Uses a ``threading.RWLock``-style pattern implemented with a
    ``threading.Lock`` for writes and a reference-counted read-lock using
    a ``threading.Condition``.  This allows multiple concurrent readers
    while serialising writers, which is the access pattern of a running
    autocomplete engine: many predict() calls read history simultaneously,
    while record_selection() writes occasionally.

    Correctness guarantee:
        - Concurrent calls to ``record()`` from any number of threads will
          not corrupt internal state.  Each write is fully serialised and
          waits for all active readers to finish.
        - All read methods (``entries()``, ``entries_for_prefix()``,
          ``counts_for_prefix()``, etc.) can execute concurrently with each
          other and see a consistent, fully-committed view of history.
        - A write cannot begin until all active reads have completed.
          A read cannot begin while a write is in progress or waiting.

    Compatibility:
        Safe on CPython, PyPy, and free-threaded CPython (PEP 703 / 3.13+).
        Does not rely on GIL guarantees.  The previous implementation relied
        on CPython's GIL making list.append() atomic, which is not guaranteed
        under PyPy or the no-GIL build.

    Writer starvation:
        Writers signal a ``Condition`` when they finish, allowing waiting
        readers to proceed.  Under heavy read load a writer may wait until
        a quiet moment.  In practice, autocomplete read traffic is bursty
        and writes are rare (one per user selection), so starvation is not
        a concern.  If it ever becomes one, switch to a fair queued lock.
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
        self._acquire_read()
        try:
            return super().snapshot()
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
        try:
            self._acquire_read()
            n = len(self._entries)
        finally:
            self._release_read()
        return f"ThreadSafeHistory(entries={n})"

    @property
    def lock(self) -> threading.Condition:
        """
        The underlying coordination lock.

        Acquire this directly only when you need a compound atomic
        operation spanning multiple calls - for example, reading the
        current count and conditionally recording a selection::

            with history.lock:
                count = history.counts_for_prefix("he").get("hello", 0)
                if count < 100:
                    history.record("he", "hello")

        Warning: acquiring ``lock`` while holding a read-lock will deadlock.
        Only use this for write-side compound operations.
        """
        return self._lock
