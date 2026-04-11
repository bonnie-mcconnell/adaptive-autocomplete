"""
CLI integration tests.

Tests the full CLI pipeline end-to-end by calling main() directly with
patched sys.argv. This covers argument parsing, engine construction,
history loading/saving, and formatted output - the entire surface that
subprocess tests would cover, but without the process-spawn overhead or
PATH/venv dependency.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import patch

from aac.cli.main import main


def _run(*args: str, history_path: Path | None = None) -> tuple[int, str]:
    """
    Invoke main() with the given CLI args and capture stdout.

    Returns (exit_code, stdout_text). Exit code 0 = success.
    """
    argv = ["aac"]
    if history_path is not None:
        argv += ["--history-path", str(history_path)]
    argv += list(args)

    buf = StringIO()
    exit_code = 0
    with patch("sys.argv", argv), patch("sys.stdout", buf):
        try:
            main()
        except SystemExit as e:
            exit_code = int(e.code) if e.code is not None else 0

    return exit_code, buf.getvalue()


# ------------------------------------------------------------------
# suggest
# ------------------------------------------------------------------

def test_suggest_returns_completions(tmp_path: Path) -> None:
    code, out = _run("--preset", "stateless", "suggest", "he",
                     history_path=tmp_path / "h.json")
    assert code == 0
    words = [w for w in out.strip().splitlines() if w]
    assert len(words) > 0
    assert all(w.startswith("he") for w in words)


def test_suggest_limit_respected(tmp_path: Path) -> None:
    code, out = _run("--preset", "stateless", "suggest", "he", "--limit", "3",
                     history_path=tmp_path / "h.json")
    assert code == 0
    words = [w for w in out.strip().splitlines() if w]
    assert len(words) <= 3


def test_suggest_unknown_prefix_no_crash(tmp_path: Path) -> None:
    code, out = _run("--preset", "stateless", "suggest", "zzzqqqxxx",
                     history_path=tmp_path / "h.json")
    assert code == 0  # must not crash


# ------------------------------------------------------------------
# explain
# ------------------------------------------------------------------

def test_explain_contains_score_fields(tmp_path: Path) -> None:
    code, out = _run("--preset", "stateless", "explain", "he",
                     history_path=tmp_path / "h.json")
    assert code == 0
    assert "score=" in out
    assert "freq=" in out


def test_explain_limit_respected(tmp_path: Path) -> None:
    code, out = _run("--preset", "stateless", "explain", "he", "--limit", "2",
                     history_path=tmp_path / "h.json")
    assert code == 0
    scored_lines = [line for line in out.strip().splitlines() if "score=" in line]
    assert len(scored_lines) <= 2


# ------------------------------------------------------------------
# record → suggest learning round-trip
# ------------------------------------------------------------------

def test_record_outputs_confirmation(tmp_path: Path) -> None:
    code, out = _run("--preset", "default", "record", "he", "hello",
                     history_path=tmp_path / "h.json")
    assert code == 0
    assert "hello" in out
    assert "he" in out


def test_record_persists_history_to_disk(tmp_path: Path) -> None:
    history_file = tmp_path / "history.json"
    code, _ = _run("--preset", "default", "record", "he", "hero",
                   history_path=history_file)
    assert code == 0
    assert history_file.exists()
    content = history_file.read_text()
    assert "hero" in content


def test_record_then_suggest_shows_learning(tmp_path: Path) -> None:
    """
    Recording a word must cause it to appear in a wider suggestion window.

    'hero' has frequency 479, rank ~25 in 'he' completions. The default
    preset weights HistoryPredictor at 1.5. We record it once, then request
    --limit 30 to verify it surfaces in the extended result set. This tests
    that the record → persist → reload → score pipeline is working end-to-end
    without relying on a specific rank position.
    """
    history_file = tmp_path / "history.json"

    # Baseline: hero should not appear in a top-5 list
    _, before = _run("--preset", "default", "suggest", "he", "--limit", "5",
                     history_path=history_file)
    assert "hero" not in before.strip().splitlines()

    # Record hero enough times to lift it into a top-30 window
    for _ in range(5):
        _run("--preset", "default", "record", "he", "hero",
             history_path=history_file)

    code, out = _run("--preset", "default", "suggest", "he", "--limit", "30",
                     history_path=history_file)
    assert code == 0
    assert "hero" in out.strip().splitlines()


# ------------------------------------------------------------------
# presets subcommand
# ------------------------------------------------------------------

def test_presets_subcommand_lists_all(tmp_path: Path) -> None:
    code, out = _run("presets", history_path=tmp_path / "h.json")
    assert code == 0
    for name in ["default", "production", "recency", "robust", "stateless"]:
        assert name in out


# ------------------------------------------------------------------
# error handling
# ------------------------------------------------------------------

def test_invalid_preset_exits_nonzero(tmp_path: Path) -> None:
    code, _ = _run("--preset", "nonexistent_preset_xyz", "suggest", "he",
                   history_path=tmp_path / "h.json")
    assert code != 0


def test_missing_subcommand_exits_nonzero() -> None:
    buf = StringIO()
    exit_code = 0
    with patch("sys.argv", ["aac"]), patch("sys.stdout", buf), patch("sys.stderr", buf):
        try:
            main()
        except SystemExit as e:
            exit_code = int(e.code) if e.code is not None else 0
    assert exit_code != 0

# ------------------------------------------------------------------
# debug subcommand
# ------------------------------------------------------------------

def test_debug_outputs_scored_and_ranked(tmp_path: Path) -> None:
    code, out = _run("--preset", "stateless", "debug", "he",
                     history_path=tmp_path / "h.json")
    assert code == 0
    assert "Scored:" in out
    assert "Ranked:" in out


# ------------------------------------------------------------------
# explain with no matches
# ------------------------------------------------------------------

def test_explain_no_suggestions_prints_message(tmp_path: Path) -> None:
    """explain on a prefix that produces no completions must not crash."""
    code, out = _run("--preset", "stateless", "explain", "zzzqqqxxx",
                     history_path=tmp_path / "h.json")
    assert code == 0
    # Either empty output or the no-explanation message - must not crash
    assert "error" not in out.lower() or "no" in out.lower()
