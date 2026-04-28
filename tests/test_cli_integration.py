"""
Integration tests for the CLI round-trip.

These tests invoke the CLI as a subprocess so they exercise the full
stack: argument parsing, engine construction, persistence, and output.
They are intentionally slow (subprocess overhead).

Run only unit tests (fast):
    pytest -m "not integration"

Run everything including integration tests:
    pytest
"""
from __future__ import annotations

import json as _json
import subprocess
from pathlib import Path

import pytest

_POETRY_RUN = ["poetry", "run", "aac"]


def _aac(*args: str, history_path: Path) -> subprocess.CompletedProcess[str]:
    """Run `aac` via poetry with a scoped history file."""
    return subprocess.run(
        [*_POETRY_RUN, "--history-path", str(history_path), *args],
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.mark.integration
class TestCliRoundTrip:
    """Verify that selections written in one process are visible in the next."""

    def test_record_creates_history_file(self, tmp_path: Path) -> None:
        """aac record writes a history file to disk."""
        history = tmp_path / "history.json"
        assert not history.exists()

        _aac("record", "he", "hello", history_path=history)

        assert history.exists(), "History file was not created after aac record"

    def test_history_file_is_valid_json_with_correct_schema(self, tmp_path: Path) -> None:
        """aac record writes valid JSON in the expected v2 schema."""
        history = tmp_path / "history.json"
        _aac("record", "he", "hello", history_path=history)

        data = _json.loads(history.read_text())
        assert data.get("version") == 2
        assert isinstance(data.get("entries"), list)
        assert len(data["entries"]) == 1
        entry = data["entries"][0]
        assert entry["prefix"] == "he"
        assert entry["value"] == "hello"
        assert "timestamp" in entry

    def test_recording_raises_equal_frequency_word_above_unrecorded_peer(
        self, tmp_path: Path
    ) -> None:
        """One recording of 'zealous' causes it to rank above 'zed' (same corpus freq=10).

        Both words have identical corpus frequency so only the history signal
        separates them.  This verifies learning reaches the engine without
        depending on overcoming a large frequency gap.
        """
        history = tmp_path / "history.json"

        _aac("record", "ze", "zealous", history_path=history)

        result = _aac("suggest", "ze", "--limit", "50", history_path=history)
        suggestions = result.stdout.strip().splitlines()

        assert "zealous" in suggestions, (
            f"Expected 'zealous' in suggestions:\n{result.stdout}"
        )
        assert "zed" in suggestions, (
            f"Expected 'zed' in suggestions:\n{result.stdout}"
        )
        assert suggestions.index("zealous") < suggestions.index("zed"), (
            f"Expected 'zealous' (recorded) to rank above 'zed' (not recorded, same freq).\n"
            f"Got: zealous at {suggestions.index('zealous')}, "
            f"zed at {suggestions.index('zed')}"
        )

    def test_suggest_uses_history_written_by_previous_process(self, tmp_path: Path) -> None:
        """suggest in a second process reads history written by a first process.

        We write the history file manually (bypassing aac record) to isolate
        the load path from the record path, and give 'zealous' enough entries
        to definitively rank above 'zed' (same corpus freq=10).
        """
        history = tmp_path / "history.json"

        # Write history directly: 5 recordings of 'zealous' under 'ze'
        history.write_text(
            _json.dumps({
                "version": 2,
                "entries": [
                    {
                        "prefix": "ze",
                        "value": "zealous",
                        "timestamp": "2024-01-01T00:00:00+00:00",
                    }
                ] * 5,
            }),
            encoding="utf-8",
        )

        result = _aac("suggest", "ze", "--limit", "50", history_path=history)
        suggestions = result.stdout.strip().splitlines()

        assert "zealous" in suggestions, (
            f"Expected 'zealous' in suggest output from pre-written history:\n{result.stdout}"
        )
        assert "zed" in suggestions, (
            f"Expected 'zed' in suggest output:\n{result.stdout}"
        )
        assert suggestions.index("zealous") < suggestions.index("zed"), (
            "Expected 'zealous' to rank above 'zed' when loaded from persisted history"
        )

    def test_presets_command_text(self) -> None:
        """presets subcommand outputs all preset names."""
        from aac.presets import available_presets
        result = subprocess.run(
            [*_POETRY_RUN, "presets"],
            capture_output=True, text=True, check=True,
        )
        for name in available_presets():
            assert name in result.stdout, f"Preset '{name}' missing from presets output"

    def test_presets_command_json(self) -> None:
        """presets --json outputs valid JSON with all preset names."""
        from aac.presets import available_presets
        result = subprocess.run(
            [*_POETRY_RUN, "presets", "--json"],
            capture_output=True, text=True, check=True,
        )
        data = _json.loads(result.stdout)
        names = {p["name"] for p in data}
        assert set(available_presets()) == names

    def test_typo_recovery(self, tmp_path: Path) -> None:
        """Production preset recovers from a single-character typo."""
        result = _aac(
            "--preset", "production", "suggest", "programing",
            history_path=tmp_path / "history.json",
        )
        assert "programming" in result.stdout, (
            f"Expected 'programming' in typo-recovery output:\n{result.stdout}"
        )
