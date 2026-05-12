"""
Tests for ThreadSafeHistory: both API correctness and concurrency safety.

Structure:
  TestThreadSafeHistoryAPI  - correctness of all public methods on the single-
                              threaded fast path, including the source-copy
                              constructor, all read methods, deprecated snapshot(),
                              __repr__, and the lock property.
  TestThreadSafeHistoryConcurrency - concurrent access patterns that expose races
                                     in naive implementations.

Concurrency tests are probabilistic: a correct implementation should pass
deterministically, but a broken implementation will fail with high probability
due to race conditions manifesting as assertion errors, incorrect counts, or
(on free-threaded builds) memory corruption.
"""
from __future__ import annotations

import threading
import warnings

from aac.domain.history import History
from aac.domain.thread_safe_history import ThreadSafeHistory


class TestThreadSafeHistoryAPI:
    """Single-threaded API correctness tests for all public methods."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def test_empty_constructor_starts_empty(self) -> None:
        history = ThreadSafeHistory()
        assert len(history) == 0
        assert history.entries() == ()

    def test_source_constructor_copies_entries(self) -> None:
        """
        ThreadSafeHistory(source=existing) must copy all entries from source.

        This is the primary use case: loading history from JsonHistoryStore
        and wrapping it for thread-safe access.
        """
        source = History()
        source.record("he", "hello")
        source.record("he", "help")
        source.record("wo", "world")

        ts = ThreadSafeHistory(source=source)

        assert ts.counts_for_prefix("he") == {"hello": 1, "help": 1}
        assert ts.counts_for_prefix("wo") == {"world": 1}
        assert len(ts) == 3

    def test_source_constructor_independent_of_source(self) -> None:
        """
        Mutations to source after construction must not affect ThreadSafeHistory.

        ThreadSafeHistory copies entries at construction time; it does not hold
        a reference to the source.
        """
        source = History()
        source.record("he", "hello")
        ts = ThreadSafeHistory(source=source)

        source.record("he", "help")  # mutate original after copy

        assert "help" not in ts.counts_for_prefix("he"), (
            "ThreadSafeHistory must not share state with the source History"
        )

    # ------------------------------------------------------------------
    # entries() and entries_for_prefix()
    # ------------------------------------------------------------------

    def test_entries_returns_all_recorded_entries(self) -> None:
        """entries() must return every recorded entry across all prefixes."""
        history = ThreadSafeHistory()
        history.record("he", "hello")
        history.record("wo", "world")

        all_entries = history.entries()
        values = {e.value for e in all_entries}
        prefixes = {e.prefix for e in all_entries}

        assert values == {"hello", "world"}
        assert prefixes == {"he", "wo"}

    def test_entries_for_prefix_filters_correctly(self) -> None:
        """entries_for_prefix() must return only entries for the given prefix."""
        history = ThreadSafeHistory()
        history.record("he", "hello")
        history.record("he", "help")
        history.record("wo", "world")

        he_entries = history.entries_for_prefix("he")
        assert all(e.prefix == "he" for e in he_entries)
        assert {e.value for e in he_entries} == {"hello", "help"}

    def test_entries_for_unknown_prefix_returns_empty(self) -> None:
        history = ThreadSafeHistory()
        history.record("he", "hello")
        assert history.entries_for_prefix("xyz") == ()

    # ------------------------------------------------------------------
    # counts_for_prefix_since()
    # ------------------------------------------------------------------

    def test_counts_for_prefix_since_filters_by_time(self) -> None:
        """
        counts_for_prefix_since() must only count entries at or after `since`.

        Uses explicit timestamps to make the time boundary deterministic.
        """
        from datetime import datetime, timezone

        t0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 1, 2, 0, 0, tzinfo=timezone.utc)

        history = ThreadSafeHistory()
        history.record("he", "hello", timestamp=t0)   # before cutoff
        history.record("he", "help",  timestamp=t1)   # at cutoff
        history.record("he", "hero",  timestamp=t2)   # after cutoff

        counts = history.counts_for_prefix_since("he", t1)
        assert "hello" not in counts, "Entry before cutoff must be excluded"
        assert counts.get("help", 0) == 1, "Entry at cutoff must be included"
        assert counts.get("hero", 0) == 1, "Entry after cutoff must be included"

    # ------------------------------------------------------------------
    # count()
    # ------------------------------------------------------------------

    def test_count_returns_total_across_all_prefixes(self) -> None:
        """count(value) returns total times `value` was selected, any prefix."""
        history = ThreadSafeHistory()
        history.record("he", "hello")
        history.record("hel", "hello")   # same value, different prefix
        history.record("wo", "world")

        assert history.count("hello") == 2
        assert history.count("world") == 1
        assert history.count("nothere") == 0

    # ------------------------------------------------------------------
    # snapshot() deprecated
    # ------------------------------------------------------------------

    def test_snapshot_deprecated_emits_deprecation_warning(self) -> None:
        """
        snapshot() is deprecated in favour of snapshot_counts().
        Calling it must emit a DeprecationWarning so callers know to migrate.
        """
        history = ThreadSafeHistory()
        history.record("he", "hello")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            history.snapshot()  # return value tested in test_snapshot_deprecated_returns_correct_data

        assert len(caught) == 1
        assert issubclass(caught[0].category, DeprecationWarning)
        assert "snapshot_counts" in str(caught[0].message)

    def test_snapshot_deprecated_returns_correct_data(self) -> None:
        """snapshot() must still return correct data despite being deprecated."""
        history = ThreadSafeHistory()
        history.record("he", "hello")
        history.record("he", "hello")
        history.record("wo", "world")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            snap = history.snapshot()

        assert snap["he"]["hello"] == 2
        assert snap["wo"]["world"] == 1

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def test_repr_shows_entry_count(self) -> None:
        """
        __repr__ must include the entry count so repr(history) is informative
        in debuggers and test output.
        """
        history = ThreadSafeHistory()
        assert "0" in repr(history)

        history.record("he", "hello")
        history.record("wo", "world")
        assert "2" in repr(history)

    def test_repr_identifies_class(self) -> None:
        history = ThreadSafeHistory()
        assert "ThreadSafeHistory" in repr(history)

    # ------------------------------------------------------------------
    # lock property
    # ------------------------------------------------------------------

    def test_lock_property_returns_condition(self) -> None:
        """
        The lock property exposes the internal Condition for advanced
        compound atomic operations. It must return a threading.Condition.
        """
        history = ThreadSafeHistory()
        import threading as _threading
        assert isinstance(history.lock, _threading.Condition)

    def test_lock_is_same_object_across_calls(self) -> None:
        """The lock property must return the same object, not a new one each time."""
        history = ThreadSafeHistory()
        assert history.lock is history.lock

_THREAD_COUNT = 10
_WRITES_PER_THREAD = 50
_READS_PER_THREAD = 200


class TestThreadSafeHistoryConcurrency:
    def test_concurrent_writes_produce_correct_count(self) -> None:
        """
        N threads each write W entries for the same prefix/value.
        The final count must equal N * W exactly.

        A naive History using unsynchronised list.append() might lose
        writes if two appends race. A coarse mutex would produce the correct
        count but serialise reads unnecessarily.
        """
        history = ThreadSafeHistory()
        barrier = threading.Barrier(_THREAD_COUNT)

        def writer() -> None:
            barrier.wait()
            for _ in range(_WRITES_PER_THREAD):
                history.record("he", "hello")

        threads = [threading.Thread(target=writer) for _ in range(_THREAD_COUNT)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = history.counts_for_prefix("he").get("hello", 0)
        expected = _THREAD_COUNT * _WRITES_PER_THREAD
        assert total == expected, (
            f"Concurrent writes lost some records. "
            f"Expected {expected}, got {total}."
        )

    def test_concurrent_reads_during_write_see_consistent_state(self) -> None:
        """
        Concurrent readers must never see a partially-written state.

        Each HistoryEntry is written atomically (under the write lock).
        Readers acquire the read lock, which prevents them from reading
        mid-write. The test verifies that no reader ever sees a count
        that is not an integer, not non-negative, or otherwise corrupted.
        """
        history = ThreadSafeHistory()
        errors: list[str] = []
        stop = threading.Event()

        def writer() -> None:
            for i in range(200):
                history.record("prefix", f"word{i % 10}")

        def reader() -> None:
            while not stop.is_set():
                counts = history.counts_for_prefix("prefix")
                for word, count in counts.items():
                    if not isinstance(count, int) or count < 0:
                        errors.append(
                            f"Corrupt count for {word!r}: {count!r}"
                        )

        reader_threads = [
            threading.Thread(target=reader, daemon=True)
            for _ in range(_THREAD_COUNT)
        ]
        writer_thread = threading.Thread(target=writer)

        for t in reader_threads:
            t.start()
        writer_thread.start()
        writer_thread.join()
        stop.set()
        for t in reader_threads:
            t.join()

        assert not errors, "Concurrent read saw corrupt state:\n" + "\n".join(errors[:5])

    def test_snapshot_history_is_consistent(self) -> None:
        """
        snapshot_history() must return a consistent point-in-time copy.

        While a writer is running, snapshots may reflect any prefix count,
        but must never reflect a partial write (i.e. the count for any word
        must be a non-negative integer).
        """
        history = ThreadSafeHistory()
        stop = threading.Event()
        errors: list[str] = []

        def writer() -> None:
            for i in range(300):
                history.record("snap", f"word{i % 5}")

        def snapshotter() -> None:
            while not stop.is_set():
                snap = history.snapshot_history()
                counts = snap.counts_for_prefix("snap")
                for word, count in counts.items():
                    if not isinstance(count, int) or count < 0:
                        errors.append(f"Corrupt snapshot count {word!r}: {count!r}")

        snap_threads = [threading.Thread(target=snapshotter, daemon=True) for _ in range(5)]
        writer_thread = threading.Thread(target=writer)

        for t in snap_threads:
            t.start()
        writer_thread.start()
        writer_thread.join()
        stop.set()
        for t in snap_threads:
            t.join()

        assert not errors, "snapshot_history() returned inconsistent data:\n" + "\n".join(errors[:5])

    def test_snapshot_counts_and_copy_not_racy(self) -> None:
        """
        snapshot_counts() and copy() (previously un-overridden in
        ThreadSafeHistory) must not race with concurrent writes.

        Regression test: before snapshot_counts() was overridden, calling it
        during a write could return a partially-written dict.
        """
        history = ThreadSafeHistory()
        stop = threading.Event()
        errors: list[str] = []

        def writer() -> None:
            for i in range(300):
                history.record("test", f"word{i % 8}")

        def reader() -> None:
            while not stop.is_set():
                sc = history.snapshot_counts()
                if not isinstance(sc, dict):
                    errors.append(f"snapshot_counts returned non-dict: {type(sc)}")
                cp = history.copy()
                if not isinstance(cp.counts_for_prefix("test"), dict):
                    errors.append("copy() produced invalid history")

        reader_threads = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
        writer_thread = threading.Thread(target=writer)

        for t in reader_threads:
            t.start()
        writer_thread.start()
        writer_thread.join()
        stop.set()
        for t in reader_threads:
            t.join()

        assert not errors, "Thread safety regression in snapshot_counts/copy:\n" + "\n".join(errors[:5])

    def test_no_writer_starvation_under_heavy_read_load(self) -> None:
        """
        A writer must eventually complete even under sustained read pressure.

        If the RW-lock implementation has writer starvation, the writer thread
        will never acquire the write lock and the test will timeout.
        """
        history = ThreadSafeHistory()
        write_completed = threading.Event()
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                history.counts_for_prefix("prefix")

        def writer() -> None:
            history.record("prefix", "written")
            write_completed.set()

        reader_threads = [
            threading.Thread(target=reader, daemon=True)
            for _ in range(_THREAD_COUNT)
        ]
        writer_thread = threading.Thread(target=writer)

        for t in reader_threads:
            t.start()
        writer_thread.start()

        completed = write_completed.wait(timeout=5.0)
        stop.set()
        writer_thread.join(timeout=1.0)
        for t in reader_threads:
            t.join(timeout=0.5)

        assert completed, (
            "Writer did not complete within 5 seconds under heavy read load. "
            "Possible writer starvation in ThreadSafeHistory."
        )
