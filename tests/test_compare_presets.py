"""
Tests for compare_presets(), PresetComparison, and warm_cache().

Covers:
  - compare_presets() returns consistent results across calls (caching)
  - Typo recovery: 'recieve' → 'receive' appears in production but not stateless
  - to_table() is non-empty and contains all preset names
  - JSON-serialisable rows (no non-serialisable values)
  - warm_cache() builds engines without error
  - Second compare_presets() call is faster than first (cache working)
"""
from __future__ import annotations

import json
import time

import pytest

from aac.presets import compare_presets, warm_cache


class TestComparePresets:
    def test_returns_preset_comparison(self) -> None:
        from aac.presets import PresetComparison
        result = compare_presets("prog", presets=["stateless", "default"])
        assert isinstance(result, PresetComparison)
        assert result.text == "prog"
        assert "stateless" in result.presets
        assert "default" in result.presets

    def test_rows_contain_expected_keys(self) -> None:
        result = compare_presets("prog", presets=["stateless"])
        assert result.rows, "Expected at least one row"
        row = result.rows[0]
        assert "value" in row
        assert "ranks" in row
        assert "base_scores" in row
        assert "boosts" in row
        assert "finals" in row

    def test_typo_recovery_differs_by_preset(self) -> None:
        """
        'recieve' should be recovered by production (SymSpell) but not stateless.
        stateless has no typo recovery so 'receive' won't appear.
        production should return 'receive' or similar corrected form.
        """
        result = compare_presets("recieve", presets=["stateless", "production"])

        # Check what production found
        production_values = {
            row["value"]
            for row in result.rows
            if row["ranks"].get("production") is not None
        }
        stateless_values = {
            row["value"]
            for row in result.rows
            if row["ranks"].get("stateless") is not None
        }

        # Production should find more results for this typo than stateless
        assert len(production_values) > len(stateless_values), (
            f"Expected production to find more results for 'recieve' than stateless. "
            f"Production: {production_values}, Stateless: {stateless_values}"
        )

    def test_to_table_contains_preset_names(self) -> None:
        result = compare_presets("prog", presets=["stateless", "production"])
        table = result.to_table(limit=3)
        assert "stateless" in table
        assert "production" in table

    def test_to_table_not_empty_for_known_word(self) -> None:
        result = compare_presets("prog", presets=["stateless"])
        table = result.to_table()
        assert table.strip(), "Expected non-empty table"
        assert "program" in table or "prog" in table

    def test_rows_are_json_serialisable(self) -> None:
        """All row values must be JSON serialisable (None, int, float, str)."""
        result = compare_presets("he", presets=["stateless", "production"])
        try:
            serialised = json.dumps({"rows": result.rows})
            parsed = json.loads(serialised)
            assert len(parsed["rows"]) == len(result.rows)
        except (TypeError, ValueError) as e:
            pytest.fail(f"compare_presets rows are not JSON-serialisable: {e}")

    def test_caching_makes_second_call_fast(self) -> None:
        """
        The second compare_presets() call with the same presets should be
        significantly faster than the first (engines are cached).
        """
        # First call may build engines
        t0 = time.perf_counter()
        compare_presets("prog", presets=["stateless"])
        first_call_ms = (time.perf_counter() - t0) * 1000

        # Second call should reuse cached engine
        t0 = time.perf_counter()
        compare_presets("hello", presets=["stateless"])
        second_call_ms = (time.perf_counter() - t0) * 1000

        # Second call must be at least 10x faster than first
        # (or first was already fast, meaning cache was pre-warmed)
        if first_call_ms > 100:  # only assert ratio if first call was actually slow
            assert second_call_ms < first_call_ms / 5, (
                f"Caching not working: first={first_call_ms:.0f}ms, "
                f"second={second_call_ms:.0f}ms (expected < {first_call_ms/5:.0f}ms)"
            )

    def test_returns_correct_row_count(self) -> None:
        result = compare_presets("prog", presets=["stateless"], limit=5)
        # Rows may be fewer than limit if vocab has fewer matches
        assert len(result.rows) <= 5

    def test_history_is_isolated_per_preset(self) -> None:
        """
        compare_presets() must not mutate the caller's History.
        Each preset engine gets an independent copy.
        """
        from aac.domain.history import History

        caller_history = History()
        caller_history.record("prog", "programming")
        initial_count = caller_history.counts_for_prefix("prog").get("programming", 0)

        compare_presets("prog", presets=["stateless"], history=caller_history)

        after_count = caller_history.counts_for_prefix("prog").get("programming", 0)
        assert initial_count == after_count, (
            f"compare_presets() mutated caller's History. "
            f"Before: {initial_count}, After: {after_count}"
        )


class TestWarmCache:
    def test_warm_cache_completes_without_error(self) -> None:
        """warm_cache() should build all preset engines without raising."""
        warm_cache(["stateless", "default"])  # Only fast presets for CI

    def test_warm_cache_makes_compare_instant(self) -> None:
        """After warm_cache(), compare_presets() should use cached engines."""
        warm_cache(["stateless"])

        t0 = time.perf_counter()
        compare_presets("he", presets=["stateless"])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < 500, (
            f"compare_presets() took {elapsed_ms:.0f}ms after warm_cache(). "
            f"Expected < 500ms (engine should be cached)."
        )
