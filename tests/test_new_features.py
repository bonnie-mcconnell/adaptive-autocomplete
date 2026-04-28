"""
Tests for new features in 0.2.0:
  - suggest(limit=N)
  - async suggest/explain/record_selection
  - FrequencyPredictor.add_word()
  - ThreadSafeHistory
  - SymSpellPredictor
  - bktree preset
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from aac.domain.history import History
from aac.domain.thread_safe_history import ThreadSafeHistory
from aac.predictors.frequency import FrequencyPredictor
from aac.predictors.symspell import SymSpellPredictor
from aac.presets import create_engine, get_preset

_VOCAB = {
    "hello": 100, "help": 80, "hero": 50, "her": 200,
    "here": 120, "heap": 40, "world": 300, "word": 150,
    "helo": 5,
}


# ------------------------------------------------------------------
# suggest(limit=N)
# ------------------------------------------------------------------

def test_suggest_limit_parameter() -> None:
    engine = create_engine("stateless", vocabulary=_VOCAB)
    all_results = engine.suggest("he")
    limited = engine.suggest("he", limit=3)
    assert len(limited) == 3
    assert limited == all_results[:3]


def test_suggest_limit_none_returns_all() -> None:
    engine = create_engine("stateless", vocabulary=_VOCAB)
    assert engine.suggest("he", limit=None) == engine.suggest("he")


def test_suggest_limit_larger_than_results() -> None:
    engine = create_engine("stateless", vocabulary=_VOCAB)
    results = engine.suggest("he", limit=1000)
    all_results = engine.suggest("he")
    assert results == all_results


def test_suggest_limit_zero_returns_empty() -> None:
    engine = create_engine("stateless", vocabulary=_VOCAB)
    assert engine.suggest("he", limit=0) == []


# ------------------------------------------------------------------
# Async API
# ------------------------------------------------------------------

def test_suggest_async_returns_same_as_sync() -> None:
    engine = create_engine("stateless", vocabulary=_VOCAB)
    sync_result = engine.suggest("he", limit=5)
    async_result = asyncio.run(engine.suggest_async("he", limit=5))
    assert async_result == sync_result


def test_explain_async_returns_same_as_sync() -> None:
    engine = create_engine("stateless", vocabulary=_VOCAB)
    sync_exps = engine.explain("he")
    async_exps = asyncio.run(engine.explain_async("he"))
    assert [e.value for e in async_exps] == [e.value for e in sync_exps]


def test_record_selection_async_learning_visible() -> None:
    """record_selection_async() writes to engine history — visible in next suggest.

    Use a vocabulary where 'heap' (freq=40) is below 'heal' (freq=45) initially.
    After 5 recordings, heap should overtake heal.
    """
    vocab = {
        "her": 200, "here": 120, "hello": 100, "help": 80,
        "hero": 50, "heal": 45, "heap": 40,
    }
    engine = create_engine("default", vocabulary=vocab)
    before = engine.suggest("he")
    assert before.index("heap") > before.index("heal"), (
        "heap must start below heal (lower freq)"
    )

    async def record_and_check() -> list[str]:
        for _ in range(5):
            await engine.record_selection_async("he", "heap")
        return engine.suggest("he")

    after = asyncio.run(record_and_check())
    assert after.index("heap") < after.index("heal"), (
        f"heap should beat heal after 5 recordings. got: {after}"
    )


# ------------------------------------------------------------------
# FrequencyPredictor.add_word()
# ------------------------------------------------------------------

def test_add_word_surfaces_new_word() -> None:
    from aac.domain.types import CompletionContext
    predictor = FrequencyPredictor({"hello": 100, "help": 80})
    results_before = [s.suggestion.value for s in predictor.predict(CompletionContext("zy"))]
    assert "zymurgy" not in results_before

    predictor.add_word("zymurgy", 50)
    results_after = [s.suggestion.value for s in predictor.predict(CompletionContext("zy"))]
    assert "zymurgy" in results_after


def test_add_word_updates_frequency() -> None:
    predictor = FrequencyPredictor({"hello": 100, "help": 80, "hero": 1})
    from aac.domain.types import CompletionContext

    before = [s.suggestion.value for s in predictor.predict(CompletionContext("he"))]
    assert before.index("hello") < before.index("hero")

    predictor.add_word("hero", 9999)
    after = [s.suggestion.value for s in predictor.predict(CompletionContext("he"))]
    assert after[0] == "hero", f"Expected hero first after freq update, got {after}"


def test_add_word_zero_frequency_ignored() -> None:
    predictor = FrequencyPredictor({"hello": 100})
    predictor.add_word("ghost", 0)
    from aac.domain.types import CompletionContext
    results = [s.suggestion.value for s in predictor.predict(CompletionContext("gh"))]
    assert "ghost" not in results


def test_add_word_empty_string_ignored() -> None:
    predictor = FrequencyPredictor({"hello": 100})
    predictor.add_word("", 100)  # should not raise


# ------------------------------------------------------------------
# ThreadSafeHistory
# ------------------------------------------------------------------

def test_thread_safe_history_concurrent_writes() -> None:
    """100 threads recording simultaneously must not corrupt history."""
    history = ThreadSafeHistory()
    n_threads = 100
    n_records_per_thread = 10

    def worker(prefix: str) -> None:
        for i in range(n_records_per_thread):
            history.record(prefix, f"word_{i}")

    threads = [threading.Thread(target=worker, args=(f"pf{i}",)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(history.entries()) == n_threads * n_records_per_thread


def test_thread_safe_history_loads_from_existing() -> None:
    """ThreadSafeHistory pre-populated from an existing History has the same entries."""
    source = History()
    source.record("he", "hello")
    source.record("he", "help")
    source.record("pr", "programming")

    ts = ThreadSafeHistory(source)
    assert len(ts.entries()) == 3
    assert ts.counts_for_prefix("he")["hello"] == 1
    assert ts.counts_for_prefix("pr")["programming"] == 1


def test_thread_safe_history_snapshot_is_independent() -> None:
    """snapshot_history() returns an independent copy — later writes don't affect it."""
    ts = ThreadSafeHistory()
    ts.record("he", "hello")

    snap = ts.snapshot_history()
    ts.record("he", "help")  # write after snapshot

    assert len(snap.entries()) == 1
    assert len(ts.entries()) == 2


def test_thread_safe_history_is_history_subclass() -> None:
    """ThreadSafeHistory is a drop-in replacement — passes isinstance checks."""
    ts = ThreadSafeHistory()
    assert isinstance(ts, History)


# ------------------------------------------------------------------
# SymSpellPredictor
# ------------------------------------------------------------------

def test_symspell_recovers_single_substitution() -> None:
    predictor = SymSpellPredictor(["hello", "help", "hero", "world"], max_distance=1)
    from aac.domain.types import CompletionContext
    results = [s.suggestion.value for s in predictor.predict(CompletionContext("helo"))]
    assert "hello" in results


def test_symspell_recovers_deletion() -> None:
    predictor = SymSpellPredictor(["programming", "program"], max_distance=2)
    from aac.domain.types import CompletionContext
    results = [s.suggestion.value for s in predictor.predict(CompletionContext("programing"))]
    assert "programming" in results


def test_symspell_works_on_short_prefix() -> None:
    """SymSpell works on 1-3 char prefixes — unlike TrigramPredictor."""
    predictor = SymSpellPredictor(["he", "her", "here", "hello"], max_distance=1)
    from aac.domain.types import CompletionContext
    # 'hx' is distance 1 from 'he' — should find it
    results = [s.suggestion.value for s in predictor.predict(CompletionContext("hx"))]
    assert "he" in results


def test_symspell_exact_match_distance_zero() -> None:
    """Exact match must have the highest score (distance=0)."""
    predictor = SymSpellPredictor(["hello", "help", "hero"], max_distance=2)
    from aac.domain.types import CompletionContext
    results = predictor.predict(CompletionContext("hello"))
    by_value = {s.suggestion.value: s.score for s in results}
    assert by_value["hello"] > by_value.get("help", 0)


def test_symspell_no_results_beyond_max_distance() -> None:
    predictor = SymSpellPredictor(["world"], max_distance=1)
    from aac.domain.types import CompletionContext
    # 'xyz' is far from 'world' — should return empty
    results = predictor.predict(CompletionContext("xyz"))
    assert results == []


# ------------------------------------------------------------------
# bktree preset
# ------------------------------------------------------------------

def test_bktree_preset_exists_and_recovers_typo() -> None:
    engine = get_preset("bktree").build(None, _VOCAB)
    assert "hello" in engine.suggest("helo", limit=20)


# ------------------------------------------------------------------
# robust preset now uses SymSpell
# ------------------------------------------------------------------

def test_robust_preset_uses_symspell() -> None:
    engine = get_preset("robust").build(None, _VOCAB)
    # Should recover typo on a short prefix (TrigramPredictor cannot do this)
    results = engine.suggest("helo", limit=20)
    assert "hello" in results


def test_robust_preset_works_on_two_char_prefix() -> None:
    """SymSpell-based robust preset handles 1-2 char prefixes correctly."""
    engine = get_preset("robust").build(None, _VOCAB)
    results = engine.suggest("wo", limit=20)
    assert "world" in results or "word" in results


# ------------------------------------------------------------------
# Coverage gap fixes
# ------------------------------------------------------------------

def test_thread_safe_history_lock_property() -> None:
    """lock property returns the threading.Condition used for coordination."""
    import threading
    ts = ThreadSafeHistory()
    assert isinstance(ts.lock, threading.Condition)


def test_symspell_empty_word_in_vocabulary_skipped() -> None:
    """Empty strings in the vocabulary are silently skipped."""
    predictor = SymSpellPredictor(["hello", "", "help"], max_distance=1)
    from aac.domain.types import CompletionContext
    results = predictor.predict(CompletionContext("hel"))
    values = [s.suggestion.value for s in results]
    assert "" not in values
    assert "hello" in values or "help" in values


def test_symspell_empty_prefix_returns_empty() -> None:
    """Empty prefix returns empty list — no suggestions possible."""
    predictor = SymSpellPredictor(["hello", "help"], max_distance=1)
    from aac.domain.types import CompletionContext
    results = predictor.predict(CompletionContext(""))
    assert results == []


# ---------------------------------------------------------------------------
# Score normalisation: all predictors in (0, 1] space
# ---------------------------------------------------------------------------

def test_frequency_predictor_scores_in_unit_interval() -> None:
    """FrequencyPredictor must emit scores in (0, 1]."""
    from aac.domain.types import CompletionContext
    from aac.predictors.frequency import FrequencyPredictor

    predictor = FrequencyPredictor({"the": 50000, "a": 10000, "zymurgy": 1})
    for prefix in ("t", "a", "z"):
        for s in predictor.predict(CompletionContext(prefix)):
            assert 0.0 < s.score <= 1.0, f"score {s.score!r} out of (0,1] for {s.value!r}"


def test_history_predictor_scores_in_unit_interval() -> None:
    """HistoryPredictor must emit scores in (0, 1]."""
    from aac.domain.history import History
    from aac.predictors.history import HistoryPredictor

    history = History()
    for _ in range(10):
        history.record("he", "hello")
    for _ in range(3):
        history.record("he", "help")
    history.record("he", "hero")

    predictor = HistoryPredictor(history)
    scores = {s.suggestion.value: s.score for s in predictor.predict("he")}

    for value, score in scores.items():
        assert 0.0 < score <= 1.0, f"score {score!r} out of (0,1] for {value!r}"

    # Most-selected word must score exactly 1.0
    assert scores["hello"] == pytest.approx(1.0)

    # Less-selected words must score below 1.0
    assert scores["help"] < 1.0
    assert scores["hero"] < scores["help"]


def test_weight_semantics_with_normalised_scores() -> None:
    """
    With log-normalised scores, HistoryPredictor(weight=1.5) should override
    FrequencyPredictor(weight=1.0) after a handful of selections — not thousands.
    """
    from aac.domain.history import History
    from aac.domain.types import WeightedPredictor
    from aac.engine.engine import AutocompleteEngine
    from aac.predictors.frequency import FrequencyPredictor
    from aac.predictors.history import HistoryPredictor

    # "the" is far more common than "test"
    vocab = {"the": 50000, "test": 100, "tell": 200}
    history = History()
    engine = AutocompleteEngine(
        predictors=[
            WeightedPredictor(FrequencyPredictor(vocab), weight=1.0),
            WeightedPredictor(HistoryPredictor(history), weight=1.5),
        ],
        history=history,
    )

    # Before any selection, "the" should lead (we just check after)
    engine.suggest("te")
    # After a few "test" selections, history signal should surface it
    for _ in range(3):
        engine.record_selection("te", "test")

    after = engine.suggest("te")
    assert after[0] == "test", (
        f"Expected 'test' to lead after 3 selections, got: {after}"
    )


# ---------------------------------------------------------------------------
# ThreadSafeHistory: read-write lock correctness
# ---------------------------------------------------------------------------

def test_thread_safe_history_concurrent_writes_do_not_corrupt() -> None:
    """Concurrent record() calls from multiple threads must not corrupt state."""
    import threading

    from aac.domain.thread_safe_history import ThreadSafeHistory

    ts = ThreadSafeHistory()
    errors: list[Exception] = []
    n_threads = 10
    writes_per_thread = 100

    def writer(thread_id: int) -> None:
        try:
            for i in range(writes_per_thread):
                ts.record("prefix", f"value_{thread_id}_{i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent writes raised: {errors}"
    # All writes must be visible
    assert len(ts.entries()) == n_threads * writes_per_thread


def test_thread_safe_history_reads_see_all_writes() -> None:
    """Every recorded entry must be visible to concurrent readers."""
    import threading

    from aac.domain.thread_safe_history import ThreadSafeHistory

    ts = ThreadSafeHistory()
    # Pre-populate
    for i in range(50):
        ts.record("p", f"v{i}")

    read_counts: list[int] = []
    errors: list[Exception] = []

    def reader() -> None:
        try:
            read_counts.append(len(ts.entries()))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # All readers must see at least the 50 pre-populated entries
    for count in read_counts:
        assert count >= 50


# ---------------------------------------------------------------------------
# History.snapshot_counts()
# ---------------------------------------------------------------------------

def test_snapshot_counts_matches_snapshot() -> None:
    """snapshot_counts() must return the same data as snapshot()."""
    from aac.domain.history import History

    h = History()
    h.record("he", "hello")
    h.record("he", "hello")
    h.record("he", "help")
    h.record("wo", "world")

    assert h.snapshot_counts() == h.snapshot()


def test_snapshot_counts_is_count_only() -> None:
    """snapshot_counts() must omit timestamps."""
    from aac.domain.history import History

    h = History()
    h.record("he", "hello")
    counts = h.snapshot_counts()
    assert counts == {"he": {"hello": 1}}


# ---------------------------------------------------------------------------
# explain_as_dicts: richer schema
# ---------------------------------------------------------------------------

def test_explain_as_dicts_includes_sources_and_components() -> None:
    """explain_as_dicts() must show correct per-predictor base_components breakdown."""
    import math

    from aac.domain.history import History
    from aac.domain.types import WeightedPredictor
    from aac.engine.engine import AutocompleteEngine
    from aac.predictors.frequency import FrequencyPredictor
    from aac.predictors.history import HistoryPredictor
    from aac.ranking.learning import LearningRanker

    history = History()
    history.record("he", "help")
    history.record("he", "help")

    vocab = {"hello": 1000, "help": 50, "hero": 200}
    engine = AutocompleteEngine(
        predictors=[
            WeightedPredictor(FrequencyPredictor(vocab), weight=1.0),
            WeightedPredictor(HistoryPredictor(history), weight=1.5),
        ],
        ranker=LearningRanker(history, boost=0.3),
        history=history,
    )
    dicts = engine.explain_as_dicts("he")
    by_value = {d["value"]: d for d in dicts}

    # "help" was selected twice — it gets both predictor scores and a ranker boost.
    help_d = by_value["help"]
    assert "frequency" in help_d["base_components"], (
        f"base_components missing 'frequency': {help_d['base_components']}"
    )
    assert "history" in help_d["base_components"], (
        f"base_components missing 'history' predictor contribution: {help_d['base_components']}"
    )

    # frequency contribution = log(1+50)/log(1+1000) * 1.0
    log_max = math.log1p(1000)
    expected_freq = math.log1p(50) / log_max
    assert help_d["base_components"]["frequency"] == pytest.approx(expected_freq, rel=1e-6)

    # history contribution = 1.0 (max for this prefix) * 1.5 weight = 1.5
    assert help_d["base_components"]["history"] == pytest.approx(1.5, rel=1e-6)

    # ranker boost is in history_components
    assert help_d["history_boost"] > 0
    assert "learning" in help_d["history_components"]

    # "hello" has no history — only frequency in base_components
    hello_d = by_value["hello"]
    assert list(hello_d["base_components"].keys()) == ["frequency"]
    assert hello_d["history_components"] == {}

    # All dicts have the required schema keys
    for d in dicts:
        for key in ("value", "base_score", "history_boost", "final_score",
                    "sources", "base_components", "history_components"):
            assert key in d, f"missing key {key!r} in {d}"


def test_weighted_predictor_rejects_zero_weight() -> None:
    """WeightedPredictor must reject weight <= 0 at construction time."""
    from aac.domain.types import WeightedPredictor
    from aac.predictors.static_prefix import StaticPrefixPredictor

    p = StaticPrefixPredictor(["hello"])
    with pytest.raises(ValueError, match="weight must be > 0"):
        WeightedPredictor(predictor=p, weight=0.0)

    with pytest.raises(ValueError, match="weight must be > 0"):
        WeightedPredictor(predictor=p, weight=-1.0)


# ---------------------------------------------------------------------------
# DecayRanker / LearningRanker: cache correctness after history update
# ---------------------------------------------------------------------------

def test_learning_ranker_cache_invalidates_after_record() -> None:
    """LearningRanker must reflect history changes between rank() calls."""
    from aac.domain.history import History
    from aac.domain.types import ScoredSuggestion, Suggestion
    from aac.ranking.learning import LearningRanker

    history = History()
    ranker = LearningRanker(history, boost=1.0)

    suggestions = [
        ScoredSuggestion(Suggestion("hello"), 1.0),
        ScoredSuggestion(Suggestion("help"), 1.0),
    ]

    # First rank: no history, order unchanged
    ranked1 = ranker.rank("he", suggestions)
    assert ranked1[0].suggestion.value == "hello"

    # Update history
    history.record("he", "help")

    # Second rank: must see the new history entry
    ranked2 = ranker.rank("he", suggestions)
    assert ranked2[0].suggestion.value == "help"
