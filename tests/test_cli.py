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
    assert "base=" in out


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
    from aac.presets import available_presets
    code, out = _run("presets", history_path=tmp_path / "h.json")
    assert code == 0
    for name in available_presets():
        assert name in out
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


# ------------------------------------------------------------------
# presets --json flag
# ------------------------------------------------------------------

def test_presets_json_flag_outputs_valid_json() -> None:
    """presets --json outputs a valid JSON array with all preset names."""
    import json as _json

    code, out = _run("presets", "--json")
    assert code == 0, f"Expected exit 0, got {code}"
    data = _json.loads(out)
    assert isinstance(data, list)
    from aac.presets import available_presets
    names = {p["name"] for p in data}
    assert set(available_presets()) == names
    for preset in data:
        assert "description" in preset
        assert "predictors" in preset
        assert "ranking" in preset
        assert "learning" in preset


# ------------------------------------------------------------------
# BrokenPipeError handling
# ------------------------------------------------------------------

def test_broken_pipe_exits_cleanly(tmp_path: Path) -> None:
    """BrokenPipeError during output is caught and main() exits with code 0.

    Regression test for the bug where `aac suggest he | head -5` printed a
    Python traceback to stderr.  We simulate a broken pipe by patching the
    suggest.run() call to raise BrokenPipeError directly, which is what
    happens when print() writes to a pipe whose consumer has closed.
    """
    from unittest.mock import patch as _patch

    exit_code = 0
    with _patch("sys.argv", ["aac", "--preset", "stateless",
                              "--history-path", str(tmp_path / "h.json"),
                              "suggest", "he"]):
        with _patch("aac.cli.suggest.run", side_effect=BrokenPipeError):
            try:
                main()
            except SystemExit as e:
                exit_code = int(e.code) if e.code is not None else 0
            except BrokenPipeError:
                exit_code = 1  # BrokenPipeError was not caught — test fails

    # Must exit 0 — BrokenPipeError must not propagate as unhandled exception
    assert exit_code == 0, (
        f"BrokenPipeError was not caught: main() exited with code {exit_code}"
    )


# ------------------------------------------------------------------
# explain output format
# ------------------------------------------------------------------

def test_explain_recency_column_always_shows_sign(tmp_path: Path) -> None:
    """recency column uses signed format (+0.00) even when boost is zero.

    Regression test for the bug where zero boost was displayed as ' 0.00'
    (space prefix) instead of '+0.00', inconsistent with non-zero boosts
    and with demo.py output.
    """
    code, out = _run("--preset", "stateless", "explain", "he",
                     history_path=tmp_path / "h.json")
    assert code == 0
    lines = [ln for ln in out.strip().splitlines() if "score=" in ln]
    assert lines, "Expected at least one scored line"
    for line in lines:
        assert "boost=+0.00" in line or "boost=+" in line, (
            f"Expected signed boost format (boost=+N.NN) in line:\n  {line}"
        )


class TestVocabPathFlag:
    """CLI --vocab-path and --vocab-format flags."""

    def test_wordlist_vocab_replaces_english(self, tmp_path: Path) -> None:
        vocab_file = tmp_path / "words.txt"
        vocab_file.write_text("zork\nzeppelin\nzigzag\n")
        code, out = _run(
            "--vocab-path", str(vocab_file),
            "suggest", "z",
            history_path=tmp_path / "h.json",
        )
        assert code == 0
        lines = [line for line in out.strip().splitlines() if line]
        assert set(lines) == {"zork", "zeppelin", "zigzag"}

    def test_text_format_counts_frequency(self, tmp_path: Path) -> None:
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("programming programming programming python python")
        code, out = _run(
            "--vocab-path", str(corpus_file),
            "--vocab-format", "text",
            "suggest", "prog",
            history_path=tmp_path / "h.json",
        )
        assert code == 0
        assert "programming" in out

    def test_missing_vocab_file_exits_with_error(self, tmp_path: Path) -> None:
        code, out = _run(
            "--vocab-path", str(tmp_path / "nonexistent.txt"),
            "suggest", "he",
            history_path=tmp_path / "h.json",
        )
        assert code != 0

    def test_vocab_path_explain_works(self, tmp_path: Path) -> None:
        vocab_file = tmp_path / "words.txt"
        vocab_file.write_text("hello\nhelp\nhero\n")
        code, out = _run(
            "--vocab-path", str(vocab_file),
            "explain", "he",
            history_path=tmp_path / "h.json",
        )
        assert code == 0
        assert "score=" in out
