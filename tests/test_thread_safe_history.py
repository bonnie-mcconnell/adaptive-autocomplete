"""
Concurrency tests for ThreadSafeHistory.

These tests deliberately exercise concurrent access patterns that can expose
races in naive implementations:
  - Many readers with an interleaved writer
  - Many writers with no coordination outside the class
  - snapshot_counts() and copy() under concurrent mutation
  - snapshot_history() consistency under load

These tests are probabilistic: a correct implementation should pass
deterministically, but a broken implementation will fail with high
probability due to race conditions manifesting as assertion errors,
incorrect counts, or (on free-threaded builds) memory corruption.
"""
from __future__ import annotations

import threading

from aac.domain.thread_safe_history import ThreadSafeHistory

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
