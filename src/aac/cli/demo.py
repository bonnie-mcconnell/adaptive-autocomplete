"""
DEMO COMMAND

Starts a minimal local HTTP server and opens an interactive browser demo
showing suggest(), explain(), and compare_presets() live.

No external dependencies - uses only the stdlib HTTP server and the
installed adaptive-autocomplete package.

Usage:
    aac demo
    aac demo --port 8765
    aac demo --preset default
    aac demo --vocab-path commands.txt
"""
from __future__ import annotations

import json
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

from aac.cli._demo_html import _DEMO_HTML

if TYPE_CHECKING:
    from aac.engine.engine import AutocompleteEngine
    from aac.ranking.explanation import RankingExplanation


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

def _make_handler(
    engine: AutocompleteEngine,
    preset: str,
    *,
    skip_comparison_engines: bool = False,
) -> type:
    from aac.presets import available_presets, create_engine

    # Build comparison engines once at startup.  Each preset engine takes
    # 1-2s to construct (SymSpell + trigram indexes).  Building them per
    # request on the /compare endpoint would block the single-threaded server
    # for 10-15s per request.  Caching at startup keeps /compare fast.
    #
    # skip_comparison_engines=True skips this build - used in tests to avoid
    # the 15s startup cost when testing other endpoints.
    _comparison_engines: dict[str, AutocompleteEngine] = {}
    if not skip_comparison_engines:
        import time as _time
        print("  building comparison engines (first run takes ~15s)...", flush=True)
        _build_start = _time.monotonic()
        for name in available_presets():
            _t = _time.monotonic()
            _comparison_engines[name] = create_engine(name)
            print(f"    {name:<12} {_time.monotonic() - _t:.1f}s", flush=True)
        print(f"  ready in {_time.monotonic() - _build_start:.1f}s", flush=True)

    class DemoHandler(BaseHTTPRequestHandler):
        _engine = engine
        _preset = preset
        _cmp_engines = _comparison_engines

        def log_message(self, fmt: str, *args: object) -> None:
            pass  # suppress per-request logs; the demo prints its own banner

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)

            if path == "/" or path == "/index.html":
                self._send_html(_DEMO_HTML.replace("window._PRESET || \"production\"", f'"{self._preset}"'))
            elif path == "/suggest":
                q = (qs.get("q", [""])[0]).strip()
                try:
                    limit = min(max(int(qs.get("limit", ["15"])[0]), 1), 100)
                except (ValueError, TypeError):
                    limit = 15
                if not q:
                    self._send_json([])
                    return
                # Single pipeline pass: value + count + confidence in one call.
                # Previously called suggest_with_history() then suggest_with_confidence()
                # separately, which ran the full score→rank pipeline twice per keystroke.
                data = self._engine.suggest_full(q, limit=limit)
                self._send_json(data)
            elif path == "/explain":
                q = (qs.get("q", [""])[0]).strip()
                try:
                    limit = min(max(int(qs.get("limit", ["15"])[0]), 1), 100)
                except (ValueError, TypeError):
                    limit = 15
                if not q:
                    self._send_json([])
                    return
                exps = self._engine.explain(q)[:limit]
                data = [
                    {
                        "value": e.value,
                        "base": round(e.base_score, 5),
                        "boost": round(e.history_boost, 5),
                        "final": round(e.final_score, 5),
                        "base_components": {k: round(v, 5) for k, v in e.base_components.items()},
                        "history_components": {k: round(v, 5) for k, v in e.history_components.items()},
                        "contribution_pct": {k: round(v, 4) for k, v in e.contribution_pct.items()},
                    }
                    for e in exps
                ]
                self._send_json(data)
            elif path == "/compare":
                q = (qs.get("q", [""])[0]).strip()
                try:
                    limit = min(max(int(qs.get("limit", ["8"])[0]), 1), 50)
                except (ValueError, TypeError):
                    limit = 8
                if not q:
                    self._send_json({"presets": [], "rows": []})
                    return

                preset_names = list(self._cmp_engines.keys())

                # Run explain() on each cached engine - fast because engines
                # are already built.
                explanations_by_preset: dict[str, list[RankingExplanation]] = {
                    name: eng.explain(q)[:limit]
                    for name, eng in self._cmp_engines.items()
                }

                seen: dict[str, None] = {}
                for name in preset_names:
                    for exp in explanations_by_preset[name]:
                        seen.setdefault(exp.value, None)
                all_values = list(seen)

                lookup: dict[str, dict[str, tuple[int, RankingExplanation]]] = {}
                for name in preset_names:
                    lookup[name] = {
                        exp.value: (i + 1, exp)
                        for i, exp in enumerate(explanations_by_preset[name])
                    }

                rows: list[dict[str, object]] = []
                for value in all_values:
                    ranks: dict[str, int | None] = {}
                    bases: dict[str, float | None] = {}
                    boosts: dict[str, float | None] = {}
                    finals: dict[str, float | None] = {}
                    for name in preset_names:
                        entry = lookup[name].get(value)
                        if entry is not None:
                            rank, exp = entry
                            ranks[name] = rank
                            bases[name] = round(exp.base_score, 5)
                            boosts[name] = round(exp.history_boost, 5)
                            finals[name] = round(exp.final_score, 5)
                        else:
                            ranks[name] = None
                            bases[name] = None
                            boosts[name] = None
                            finals[name] = None
                    rows.append({
                        "value": value,
                        "ranks": ranks,
                        "base_scores": bases,
                        "boosts": boosts,
                        "finals": finals,
                    })

                self._send_json({"presets": preset_names, "rows": rows})
            elif path == "/health":
                # Docker healthcheck endpoint.
                self._send_json({"status": "ok", "preset": self._preset})
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self) -> None:
            """
            POST /record  - preferred over GET /record for semantic correctness.

            HTTP GET must be idempotent and safe (no side effects). Recording a
            selection is a state mutation; it belongs on POST. Browser prefetch,
            link scanners, and crawlers will not automatically issue POST requests,
            so using POST prevents accidental history pollution.

            Body: application/x-www-form-urlencoded or query string on the URL.
            Returns: {"recorded": true|false}
            """
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)

            if path == "/record":
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length > 0:
                    body_bytes = self.rfile.read(content_length)
                    from urllib.parse import parse_qs as _pqs
                    body_qs = _pqs(body_bytes.decode("utf-8", errors="replace"))
                    qs = {**qs, **body_qs}

                q = (qs.get("q", [""])[0]).strip()
                value = (qs.get("value", [""])[0]).strip()
                if q and value:
                    self._engine.record_selection(q, value)
                self._send_json({"recorded": bool(q and value)})
            else:
                self.send_response(405)
                self.send_header("Allow", "GET")
                self.end_headers()

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, data: object) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

    return DemoHandler


def _find_free_port(preferred: int, host: str = "127.0.0.1") -> int:
    """Return preferred port if free on host, otherwise any free port on host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, preferred))
            return preferred
        except OSError:
            s.bind((host, 0))
            return int(s.getsockname()[1])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    *,
    engine: AutocompleteEngine,
    host: str = "127.0.0.1",
    port: int = 8421,
    preset: str = "production",
    no_browser: bool = False,
) -> None:
    """
    Start the interactive demo server and open it in the default browser.

    Parameters:
        engine:     The engine to use for suggestions and explanations.
        host:       Host interface to bind to.  Default ``"127.0.0.1"`` binds
                    to localhost only (safe for local use).  Pass
                    ``"0.0.0.0"`` to listen on all interfaces - required
                    when running inside Docker so the container port is
                    reachable from the host machine.
        port:       Preferred local port.  If occupied, a free port is used.
        preset:     Name of the active preset (display only).
        no_browser: If True, print the URL but do not open the browser.
    """
    port = _find_free_port(port, host)
    # When bound to all interfaces, show 127.0.0.1 in the browser URL so
    # clicking it from the local machine actually works.
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{display_host}:{port}"

    handler_class = _make_handler(engine, preset)
    server = HTTPServer((host, port), handler_class)

    print("\nadaptive-autocomplete demo")
    print(f"  preset:  {preset}")
    print(f"  host:    {host}")
    print(f"  url:     {url}")
    print("\npress Ctrl+C to stop\n")

    if not no_browser:
        # Open after a short delay so the server is listening first.
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("\ndemo server stopped")
