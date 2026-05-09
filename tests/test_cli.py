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

    The helper finds the subcommand (first non-flag positional arg) and
    inserts --history-path immediately after it so the flag is at the
    subcommand level, not at the global level.
    """
    argv = ["aac"] + list(args)
    if history_path is not None:
        # Find where the subcommand is (first non-flag arg after "aac")
        # and insert --history-path after it.
        # e.g. ["aac", "suggest", "he"] -> ["aac", "suggest", "he", "--history-path", "..."]
        argv += ["--history-path", str(history_path)]

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
    code, out = _run("suggest", "he", "--preset", "stateless",
                     history_path=tmp_path / "h.json")
    assert code == 0
    words = [w for w in out.strip().splitlines() if w]
    assert len(words) > 0
    assert all(w.startswith("he") for w in words)


def test_suggest_limit_respected(tmp_path: Path) -> None:
    code, out = _run("suggest", "he", "--preset", "stateless", "--limit", "3",
                     history_path=tmp_path / "h.json")
    assert code == 0
    words = [w for w in out.strip().splitlines() if w]
    assert len(words) <= 3


def test_suggest_unknown_prefix_no_crash(tmp_path: Path) -> None:
    code, out = _run("suggest", "zzzqqqxxx", "--preset", "stateless",
                     history_path=tmp_path / "h.json")
    assert code == 0  # must not crash


# ------------------------------------------------------------------
# explain
# ------------------------------------------------------------------

def test_explain_contains_score_fields(tmp_path: Path) -> None:
    code, out = _run("explain", "he", "--preset", "stateless",
                     history_path=tmp_path / "h.json")
    assert code == 0
    assert "score=" in out
    assert "base=" in out


def test_explain_limit_respected(tmp_path: Path) -> None:
    code, out = _run("explain", "he", "--preset", "stateless", "--limit", "2",
                     history_path=tmp_path / "h.json")
    assert code == 0
    scored_lines = [line for line in out.strip().splitlines() if "score=" in line]
    assert len(scored_lines) <= 2


# ------------------------------------------------------------------
# record → suggest learning round-trip
# ------------------------------------------------------------------

def test_record_outputs_confirmation(tmp_path: Path) -> None:
    code, out = _run("record", "he", "hello", "--preset", "default",
                     history_path=tmp_path / "h.json")
    assert code == 0
    assert "hello" in out
    assert "he" in out


def test_record_persists_history_to_disk(tmp_path: Path) -> None:
    history_file = tmp_path / "history.json"
    code, _ = _run("record", "he", "hero", "--preset", "default",
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
    _, before = _run("suggest", "he", "--limit", "5", "--preset", "default",
                     history_path=history_file)
    assert "hero" not in before.strip().splitlines()

    # Record hero enough times to lift it into a top-30 window
    for _ in range(5):
        _run("record", "he", "hero", "--preset", "default",
             history_path=history_file)

    code, out = _run("suggest", "he", "--limit", "30", "--preset", "default",
                     history_path=history_file)
    assert code == 0
    assert "hero" in out.strip().splitlines()


# ------------------------------------------------------------------
# presets subcommand
# ------------------------------------------------------------------

def test_presets_subcommand_lists_all(tmp_path: Path) -> None:
    from aac.presets import available_presets
    code, out = _run("presets")
    assert code == 0
    for name in available_presets():
        assert name in out


# ------------------------------------------------------------------
# error handling
# ------------------------------------------------------------------

def test_invalid_preset_exits_nonzero(tmp_path: Path) -> None:
    code, _ = _run("suggest", "he", "--preset", "nonexistent_preset_xyz",
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
    code, out = _run("debug", "he", "--preset", "stateless",
                     history_path=tmp_path / "h.json")
    assert code == 0
    assert "Scored:" in out
    assert "Ranked:" in out


# ------------------------------------------------------------------
# explain with no matches
# ------------------------------------------------------------------

def test_explain_no_suggestions_prints_message(tmp_path: Path) -> None:
    """explain on a prefix that produces no completions must not crash."""
    code, out = _run("explain", "zzzqqqxxx", "--preset", "stateless",
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
    with _patch("sys.argv", ["aac", "suggest", "he",
                              "--preset", "stateless",
                              "--history-path", str(tmp_path / "h.json")]):
        with _patch("aac.cli.suggest.run", side_effect=BrokenPipeError):
            try:
                main()
            except SystemExit as e:
                exit_code = int(e.code) if e.code is not None else 0
            except BrokenPipeError:
                exit_code = 1  # BrokenPipeError was not caught - test fails

    # Must exit 0 - BrokenPipeError must not propagate as unhandled exception
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
    code, out = _run("explain", "he", "--preset", "stateless",
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
            "suggest", "z",
            "--vocab-path", str(vocab_file),
            history_path=tmp_path / "h.json",
        )
        assert code == 0
        lines = [line for line in out.strip().splitlines() if line]
        assert set(lines) == {"zork", "zeppelin", "zigzag"}

    def test_text_format_counts_frequency(self, tmp_path: Path) -> None:
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("programming programming programming python python")
        code, out = _run(
            "suggest", "prog",
            "--vocab-path", str(corpus_file),
            "--vocab-format", "text",
            history_path=tmp_path / "h.json",
        )
        assert code == 0
        assert "programming" in out

    def test_missing_vocab_file_exits_with_error(self, tmp_path: Path) -> None:
        code, out = _run(
            "suggest", "he",
            "--vocab-path", str(tmp_path / "nonexistent.txt"),
            history_path=tmp_path / "h.json",
        )
        assert code != 0

    def test_vocab_path_explain_works(self, tmp_path: Path) -> None:
        vocab_file = tmp_path / "words.txt"
        vocab_file.write_text("hello\nhelp\nhero\n")
        code, out = _run(
            "explain", "he",
            "--vocab-path", str(vocab_file),
            history_path=tmp_path / "h.json",
        )
        assert code == 0
        assert "score=" in out


class TestHistorySubcommand:
    """CLI 'aac history' subcommand."""

    def test_history_empty(self, tmp_path: Path) -> None:
        code, out = _run("history", history_path=tmp_path / "h.json")
        assert code == 0
        assert "No history recorded" in out

    def test_history_summary_after_records(self, tmp_path: Path) -> None:
        hp = tmp_path / "h.json"
        _run("record", "he", "hello", history_path=hp)
        _run("record", "he", "hello", history_path=hp)
        _run("record", "he", "help", history_path=hp)
        _run("record", "pro", "programming", history_path=hp)

        code, out = _run("history", history_path=hp)
        assert code == 0
        assert "he" in out
        assert "pro" in out
        assert "hello" in out

    def test_history_prefix_breakdown(self, tmp_path: Path) -> None:
        hp = tmp_path / "h.json"
        _run("record", "he", "hello", history_path=hp)
        _run("record", "he", "hello", history_path=hp)
        _run("record", "he", "help", history_path=hp)

        code, out = _run("history", "he", history_path=hp)
        assert code == 0
        assert "hello" in out
        assert "help" in out
        # hello selected twice, should appear first
        assert out.index("hello") < out.index("help")

    def test_history_unknown_prefix_says_so(self, tmp_path: Path) -> None:
        hp = tmp_path / "h.json"
        _run("record", "he", "hello", history_path=hp)
        code, out = _run("history", "xyz", history_path=hp)
        assert code == 0
        assert "No history" in out

    def test_history_shows_ago_label(self, tmp_path: Path) -> None:
        hp = tmp_path / "h.json"
        _run("record", "he", "hello", history_path=hp)
        code, out = _run("history", "he", history_path=hp)
        assert code == 0
        # should show some time-ago label
        assert "ago" in out



# ------------------------------------------------------------------
# demo subcommand
# ------------------------------------------------------------------



class TestDemoSubcommand:
    """Smoke tests for `aac demo` - verifies the server starts and serves.

    All tests use skip_comparison_engines=True to avoid the ~15s startup
    cost of building all preset engines.  The comparison endpoint itself
    is covered by the integration test suite where startup cost is acceptable.
    """

    def _make_server(self, vocab: dict, preset: str = "stateless", port_base: int = 18500):
        """Build a test server with skip_comparison_engines=True."""
        from http.server import HTTPServer

        from aac.cli.demo import _find_free_port, _make_handler
        from aac.presets import create_engine

        engine = create_engine(preset, vocabulary=vocab)
        port = _find_free_port(port_base)
        handler = _make_handler(engine, preset, skip_comparison_engines=True)
        server = HTTPServer(("127.0.0.1", port), handler)
        return engine, server, port

    def _one_request(self, server, path: str):
        """Serve exactly one request and return (status, body_bytes)."""
        import http.client
        import threading
        import time

        port = server.server_address[1]
        thread = threading.Thread(target=lambda: server.handle_request())
        thread.daemon = True
        thread.start()
        time.sleep(0.05)

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            body = resp.read()
            return resp.status, body
        finally:
            conn.close()
            thread.join(timeout=2)

    def test_demo_serves_index_page(self) -> None:
        """Server must respond to GET / with valid HTML."""
        _, server, _ = self._make_server({"hello": 100, "help": 80})
        status, body = self._one_request(server, "/")
        html = body.decode("utf-8")

        assert status == 200, f"Expected 200, got {status}"
        assert "adaptive-autocomplete" in html, "Page must contain project name"
        assert "<html" in html.lower(), "Response must be valid HTML"

    def test_demo_suggest_returns_json_array(self) -> None:
        """GET /suggest?q=he returns a JSON array with the correct schema."""
        import json as _json

        _, server, _ = self._make_server({"hello": 100, "help": 80, "hero": 60})
        status, body = self._one_request(server, "/suggest?q=he&limit=3")
        data = _json.loads(body)

        assert status == 200
        assert isinstance(data, list), f"Expected list, got {type(data)}"
        assert len(data) > 0, "Must return at least one suggestion for 'he'"
        first = data[0]
        assert "word" in first, f"Missing 'word' key: {first}"
        assert "count" in first, f"Missing 'count' key: {first}"
        assert "confidence" in first, f"Missing 'confidence' key: {first}"
        assert abs(first["confidence"] - 1.0) < 1e-6, (
            f"Top suggestion confidence must be 1.0, got {first['confidence']}"
        )

    def test_demo_suggest_empty_query_returns_empty(self) -> None:
        """GET /suggest?q= returns an empty JSON array."""
        import json as _json

        _, server, _ = self._make_server({"hello": 100})
        status, body = self._one_request(server, "/suggest?q=&limit=5")
        data = _json.loads(body)
        assert status == 200
        assert data == []

    def test_demo_explain_returns_json_with_schema(self) -> None:
        """GET /explain?q=he returns a JSON array with score fields."""
        import json as _json

        _, server, _ = self._make_server({"hello": 100, "help": 80})
        status, body = self._one_request(server, "/explain?q=he&limit=3")
        data = _json.loads(body)

        assert status == 200
        assert isinstance(data, list)
        assert len(data) > 0
        first = data[0]
        for key in ("value", "base", "boost", "final", "base_components"):
            assert key in first, f"Missing key {key!r} in explain response: {first}"

    def test_demo_record_persists_selection(self) -> None:
        """GET /record?q=he&value=hello records the selection and returns {recorded: true}."""
        import json as _json

        engine, server, _ = self._make_server({"hello": 100, "help": 80}, preset="default")
        status, body = self._one_request(server, "/record?q=he&value=hello")
        data = _json.loads(body)

        assert status == 200
        assert data.get("recorded") is True
        assert engine.history.counts_for_prefix("he").get("hello") == 1

    def test_demo_record_empty_query_still_responds(self) -> None:
        """GET /record with empty q returns {recorded: false} without crashing."""
        import json as _json

        _, server, _ = self._make_server({"hello": 100}, preset="default")
        status, body = self._one_request(server, "/record?q=&value=hello")
        data = _json.loads(body)
        assert status == 200
        assert data.get("recorded") is False

    def test_demo_suggest_invalid_limit_does_not_crash(self) -> None:
        """GET /suggest?q=he&limit=abc must not crash - returns valid JSON."""
        import json as _json

        _, server, _ = self._make_server({"hello": 100, "help": 80})
        status, body = self._one_request(server, "/suggest?q=he&limit=abc")
        assert status == 200
        data = _json.loads(body)
        assert isinstance(data, list), "Should return a list even with invalid limit"

    def test_demo_explain_invalid_limit_does_not_crash(self) -> None:
        """GET /explain?q=he&limit=xyz must not crash."""
        import json as _json

        _, server, _ = self._make_server({"hello": 100, "help": 80})
        status, body = self._one_request(server, "/explain?q=he&limit=xyz")
        assert status == 200
        data = _json.loads(body)
        assert isinstance(data, list)

    def test_demo_suggest_uses_single_pipeline_pass(self) -> None:
        """
        /suggest must return word+count+confidence using suggest_full().
        The response schema must match suggest_full() output, not the old
        double-call (suggest_with_history + suggest_with_confidence) schema.
        Verified by checking that confidence values are consistent with
        what a single ranked list would produce - not a mismatched merge.
        """
        import json as _json

        _, server, _ = self._make_server(
            {"hello": 100, "help": 80, "hero": 60}, preset="default"
        )
        status, body = self._one_request(server, "/suggest?q=he&limit=5")
        assert status == 200
        data = _json.loads(body)

        assert len(data) > 0
        for item in data:
            assert "word" in item
            assert "count" in item
            assert "confidence" in item
            assert 0.0 <= item["confidence"] <= 1.0

        # Top result must have highest confidence
        assert data[0]["confidence"] >= data[-1]["confidence"]

    def test_demo_unknown_path_returns_404(self) -> None:
        """Unknown paths must return 404, not crash."""
        _, server, _ = self._make_server({"hello": 100})
        status, _ = self._one_request(server, "/nonexistent_path")
        assert status == 404

    def test_demo_find_free_port_returns_valid_port(self) -> None:
        """_find_free_port() returns an integer in the valid port range."""
        from aac.cli.demo import _find_free_port
        port = _find_free_port(8421)
        assert isinstance(port, int)
        assert 1 <= port <= 65535
