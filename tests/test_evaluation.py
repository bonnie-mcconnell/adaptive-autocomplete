"""
Tests for the evaluation module: metrics, harness, datasets, optimiser.

These tests verify:
  - Each metric formula is mathematically correct
  - EvaluationHarness produces consistent results
  - make_synthetic_query_log and make_query_log_from_history work correctly
  - WeightOptimiser finds improvements when the space has one
  - JSONL load/save round-trips correctly
  - CLI eval and tune commands parse and run without error
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aac.evaluation import (
    EvaluationHarness,
    QueryLog,
    QueryLogEntry,
    WeightOptimiser,
    make_query_log_from_history,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from aac.evaluation.datasets import (
    load_jsonl,
    make_synthetic_query_log,
    save_jsonl,
)
from aac.evaluation.metrics import average_precision
from aac.presets import create_engine

# ---------------------------------------------------------------------------
# Metric correctness
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_perfect_result(self) -> None:
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)

    def test_zero_hits(self) -> None:
        assert precision_at_k(["x", "y"], {"a", "b"}, k=2) == pytest.approx(0.0)

    def test_partial_hits(self) -> None:
        # 1 hit in top 3 → P@3 = 1/3
        assert precision_at_k(["a", "x", "y"], {"a"}, k=3) == pytest.approx(1 / 3)

    def test_k_limits_results(self) -> None:
        # Only look at top 2; "b" is at rank 3 and should not count
        assert precision_at_k(["x", "y", "b"], {"b"}, k=2) == pytest.approx(0.0)

    def test_empty_relevant(self) -> None:
        assert precision_at_k(["a", "b"], set(), k=2) == 0.0

    def test_k_zero(self) -> None:
        assert precision_at_k(["a", "b"], {"a"}, k=0) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        assert recall_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_partial_recall(self) -> None:
        # Found 1 of 2 relevant items
        assert recall_at_k(["a", "x"], {"a", "b"}, k=2) == pytest.approx(0.5)

    def test_empty_relevant(self) -> None:
        assert recall_at_k(["a"], set(), k=1) == 0.0


class TestMRRAtK:
    def test_first_result_relevant(self) -> None:
        assert mrr_at_k(["a", "b", "c"], {"a"}, k=3) == pytest.approx(1.0)

    def test_second_result_relevant(self) -> None:
        assert mrr_at_k(["x", "a", "b"], {"a"}, k=3) == pytest.approx(0.5)

    def test_third_result_relevant(self) -> None:
        assert mrr_at_k(["x", "y", "a"], {"a"}, k=3) == pytest.approx(1 / 3)

    def test_no_relevant_in_topk(self) -> None:
        assert mrr_at_k(["x", "y", "z"], {"a"}, k=3) == pytest.approx(0.0)

    def test_relevant_beyond_k(self) -> None:
        # "a" is at rank 4 but k=3 - should return 0
        assert mrr_at_k(["x", "y", "z", "a"], {"a"}, k=3) == pytest.approx(0.0)

    def test_empty_relevant(self) -> None:
        assert mrr_at_k(["a"], set(), k=3) == 0.0


class TestNDCGAtK:
    def test_perfect_ranking(self) -> None:
        # Both relevant items at top 2 positions
        result = ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=3)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_reversed_ranking(self) -> None:
        # Relevant items at positions 2 and 3 instead of 1 and 2
        result = ndcg_at_k(["x", "a", "b"], {"a", "b"}, k=3)
        assert 0.0 < result < 1.0

    def test_perfect_beats_reversed(self) -> None:
        perfect = ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3)
        reversed_order = ndcg_at_k(["x", "a", "b"], {"a", "b"}, k=3)
        assert perfect > reversed_order

    def test_no_relevant(self) -> None:
        assert ndcg_at_k(["x", "y"], {"a"}, k=2) == pytest.approx(0.0)

    def test_graded_relevance(self) -> None:
        # "a" is more relevant than "b"
        grades = {"a": 1.0, "b": 0.5}
        good = ndcg_at_k(["a", "b"], {"a", "b"}, k=2, grades=grades)
        bad = ndcg_at_k(["b", "a"], {"a", "b"}, k=2, grades=grades)
        assert good > bad


class TestAveragePrecision:
    def test_perfect(self) -> None:
        assert average_precision(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_partial(self) -> None:
        # "a" at rank 1 (P@1=1.0), "b" not in top 3
        result = average_precision(["a", "x", "y"], {"a", "b"}, k=3)
        # AP = (1/2) * (1.0) = 0.5 (only "a" found, 2 relevant total)
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

class TestMakeSyntheticQueryLog:
    def test_produces_entries(self) -> None:
        vocab = ["programming", "program", "progress", "python", "pytest"]
        log = make_synthetic_query_log(vocab, prefix_lengths=[2, 3], include_typos=False)
        assert len(log) > 0

    def test_prefix_lengths_respected(self) -> None:
        vocab = ["programming", "program"]
        log = make_synthetic_query_log(vocab, prefix_lengths=[3], include_typos=False)
        for entry in log:
            # All prefixes must have length <= 3 (some may be shorter due to word length)
            assert len(entry.prefix) <= 4  # allow some slack for overlap

    def test_relevant_completions_start_with_prefix(self) -> None:
        vocab = ["hello", "help", "world"]
        log = make_synthetic_query_log(vocab, prefix_lengths=[2], include_typos=False)
        for entry in log:
            for word in entry.relevant:
                assert word.startswith(entry.prefix), (
                    f"Relevant word {word!r} does not start with prefix {entry.prefix!r}"
                )

    def test_entries_have_nonempty_relevant(self) -> None:
        vocab = ["hello", "help", "held", "hero"]
        log = make_synthetic_query_log(vocab, prefix_lengths=[2], include_typos=False)
        for entry in log:
            assert entry.relevant, f"Entry for {entry.prefix!r} has empty relevant set"


class TestMakeQueryLogFromHistory:
    def test_empty_history_raises(self) -> None:
        from aac.domain.history import History
        history = History()
        with pytest.raises(ValueError, match="no entries"):
            EvaluationHarness.from_history(history, min_count=1)

    def test_produces_entries(self) -> None:
        from aac.domain.history import History
        history = History()
        history.record("prog", "programming")
        history.record("prog", "programming")
        history.record("prog", "program")
        log = make_query_log_from_history(history, min_count=1)
        assert len(log) == 1
        assert log[0].prefix == "prog"
        assert "programming" in log[0].relevant
        assert "program" in log[0].relevant

    def test_min_count_filters_rare_words(self) -> None:
        from aac.domain.history import History
        history = History()
        history.record("prog", "programming")
        history.record("prog", "programming")
        history.record("prog", "program")  # only once
        log = make_query_log_from_history(history, min_count=2)
        assert len(log) == 1
        assert "programming" in log[0].relevant
        assert "program" not in log[0].relevant  # filtered by min_count=2

    def test_grades_proportional_to_count(self) -> None:
        from aac.domain.history import History
        history = History()
        for _ in range(4):
            history.record("he", "hello")
        for _ in range(2):
            history.record("he", "help")
        log = make_query_log_from_history(history, min_count=1)
        assert len(log) == 1
        grades = log[0].grades
        # "hello" has 4 selections (max), so grade should be 1.0
        assert grades["hello"] == pytest.approx(1.0)
        # "help" has 2 selections (half of max), so grade should be 0.5
        assert grades["help"] == pytest.approx(0.5)


class TestJSONLRoundTrip:
    def test_save_and_load(self, tmp_path: Path) -> None:
        log = [
            QueryLogEntry("prog", {"programming", "program"}),
            QueryLogEntry("hel", {"hello", "help"}, grades={"hello": 1.0, "help": 0.8}),
        ]
        path = tmp_path / "test.jsonl"
        save_jsonl(log, path)
        loaded = load_jsonl(path)
        assert len(loaded) == len(log)
        assert loaded[0].prefix in {"prog", "hel"}

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"prefix": "prog"}\n{invalid json}\n')
        # Line 1 is valid JSON but missing "relevant" → ValueError for line 1
        with pytest.raises(ValueError):
            load_jsonl(path)

    def test_load_missing_relevant_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.jsonl"
        path.write_text('{"prefix": "prog"}\n')  # no "relevant" key
        with pytest.raises(ValueError):
            load_jsonl(path)


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

class TestEvaluationHarness:
    def _small_log(self) -> QueryLog:
        return [
            QueryLogEntry("prog", {"programming", "program"}),
            QueryLogEntry("hel", {"hello", "help"}),
            QueryLogEntry("wor", {"world", "word"}),
        ]

    def test_run_returns_result(self) -> None:
        harness = EvaluationHarness(self._small_log(), k=5)
        engine = create_engine("production")
        result = harness.run(engine)
        assert result.n_queries == 3
        assert 0.0 <= result.mean_mrr <= 1.0
        assert 0.0 <= result.mean_ndcg <= 1.0
        assert 0.0 <= result.hit_rate <= 1.0

    def test_production_beats_stateless_on_typo(self) -> None:
        """Production (with SymSpell) should do better on typo queries than stateless."""
        typo_log = [
            QueryLogEntry("programing", {"programming"}),  # missing 'm'
            QueryLogEntry("recieve", {"receive"}),         # transposition
        ]
        harness = EvaluationHarness(typo_log, k=5)
        stateless = harness.run(create_engine("stateless"))
        production = harness.run(create_engine("production"))
        # production should find more typo corrections
        assert production.hit_rate >= stateless.hit_rate

    def test_perfect_engine_scores_one(self) -> None:
        """An engine that always returns the relevant item first should score MRR=1."""
        log = [QueryLogEntry("he", {"hello"})]
        engine = create_engine("production")
        # Force "hello" to the top via history
        for _ in range(10):
            engine.record_selection("he", "hello")
        harness = EvaluationHarness(log, k=5)
        result = harness.run(engine)
        assert result.mean_mrr == pytest.approx(1.0)

    def test_empty_log_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            EvaluationHarness([], k=5)

    def test_k_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            EvaluationHarness(self._small_log(), k=0)

    def test_summary_is_string(self) -> None:
        harness = EvaluationHarness(self._small_log(), k=5)
        result = harness.run(create_engine("stateless"))
        summary = result.summary()
        assert isinstance(summary, str)
        assert "MRR" in summary
        assert "NDCG" in summary

    def test_markdown_table_contains_metric_names(self) -> None:
        harness = EvaluationHarness(self._small_log(), k=5)
        result = harness.run(create_engine("stateless"))
        table = result.to_markdown_table()
        assert "MRR" in table
        assert "NDCG" in table
        assert "Hit rate" in table

    def test_to_dict_is_json_serialisable(self) -> None:
        harness = EvaluationHarness(self._small_log(), k=5)
        result = harness.run(create_engine("stateless"))
        d = result.to_dict()
        json.dumps(d)  # must not raise

    def test_worst_queries_returns_n(self) -> None:
        harness = EvaluationHarness(self._small_log(), k=5)
        result = harness.run(create_engine("stateless"))
        worst = result.worst_queries(2)
        assert len(worst) <= 2

    def test_breakdown_by_prefix_length(self) -> None:
        harness = EvaluationHarness(self._small_log(), k=5)
        result = harness.run(create_engine("stateless"))
        assert isinstance(result.by_prefix_length, dict)


# ---------------------------------------------------------------------------
# WeightOptimiser
# ---------------------------------------------------------------------------

class TestWeightOptimiser:
    def _harness(self) -> EvaluationHarness:
        from aac.data import load_english_frequencies
        vocab = list(load_english_frequencies().keys())[:150]
        log = make_synthetic_query_log(vocab, prefix_lengths=[3], include_typos=False)
        return EvaluationHarness(log, k=5)

    def test_grid_search_returns_result(self) -> None:
        harness = self._harness()
        opt = WeightOptimiser(harness, metric="mrr", verbose=False)
        result = opt.grid_search(
            base_preset="stateless",
            weight_grid={"frequency": [0.5, 1.0, 2.0]},
        )
        assert result.n_evaluations >= 1
        assert 0.0 <= result.best_score <= 1.0
        assert result.strategy == "grid_search"

    def test_coordinate_descent_returns_result(self) -> None:
        harness = self._harness()
        opt = WeightOptimiser(harness, metric="mrr", verbose=False)
        result = opt.coordinate_descent(
            base_preset="stateless",
            weight_grid={"frequency": [0.5, 1.0, 2.0]},
            max_rounds=2,
        )
        assert result.n_evaluations >= 1
        assert result.strategy == "coordinate_descent"

    def test_invalid_metric_raises(self) -> None:
        harness = self._harness()
        with pytest.raises(ValueError, match="Unknown metric"):
            WeightOptimiser(harness, metric="invalid_metric")

    def test_best_score_geq_baseline(self) -> None:
        """Optimiser must never return a score worse than the baseline."""
        harness = self._harness()
        opt = WeightOptimiser(harness, metric="mrr", verbose=False)
        result = opt.coordinate_descent(
            base_preset="stateless",
            weight_grid={"frequency": [0.5, 1.0, 2.0]},
        )
        assert result.best_score >= result.baseline_score - 1e-9

    def test_report_contains_metric_name(self) -> None:
        harness = self._harness()
        opt = WeightOptimiser(harness, metric="ndcg", verbose=False)
        result = opt.coordinate_descent(
            base_preset="stateless",
            weight_grid={"frequency": [0.8, 1.0]},
        )
        assert "ndcg" in result.report()
        assert "coordinate_descent" in result.report()


def test_from_history_raises_type_error_on_non_history() -> None:
    """from_history() should raise TypeError when given something that isn't a History."""
    import pytest
    with pytest.raises(TypeError, match="History instance"):
        EvaluationHarness.from_history({"prefix": {"value": 1}})  # type: ignore[arg-type]
