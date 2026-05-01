"""
DEMO COMMAND

Starts a minimal local HTTP server and opens an interactive browser demo
showing suggest(), explain(), and compare_presets() live.

No external dependencies - uses only the stdlib HTTP server and the
installed adaptive-autocomplete package.

Usage:
    aac demo
    aac demo --port 8765
    aac --preset default demo
    aac --vocab-path commands.txt demo
"""
from __future__ import annotations

import json
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    from aac.engine.engine import AutocompleteEngine

# ---------------------------------------------------------------------------
# HTML - embedded so the demo requires zero extra files
# ---------------------------------------------------------------------------

_DEMO_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>adaptive-autocomplete · live demo</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --accent: #7c6af7;
    --accent-dim: #4a3fa0;
    --text: #e2e4ed;
    --text-dim: #7b7f94;
    --green: #4ade80;
    --amber: #fbbf24;
    --red: #f87171;
    --mono: "JetBrains Mono", "Fira Code", "Cascadia Code", ui-monospace, monospace;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { font-size: 15px; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    line-height: 1.6;
    min-height: 100vh;
    padding: 2rem 1rem;
  }
  .container { max-width: 900px; margin: 0 auto; }

  /* Header */
  .header { margin-bottom: 2.5rem; }
  .header h1 {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text);
  }
  .header h1 span { color: var(--accent); }
  .header p { color: var(--text-dim); margin-top: 0.4rem; font-size: 0.9rem; }
  .preset-badge {
    display: inline-block;
    background: var(--accent-dim);
    color: var(--text);
    font-size: 0.75rem;
    font-family: var(--mono);
    padding: 0.15rem 0.55rem;
    border-radius: 4px;
    margin-left: 0.5rem;
    vertical-align: middle;
  }

  /* Search box */
  .search-row {
    display: flex;
    gap: 0.6rem;
    margin-bottom: 1.75rem;
    align-items: center;
  }
  .search-input {
    flex: 1;
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 1.1rem;
    font-family: var(--mono);
    color: var(--text);
    outline: none;
    transition: border-color 0.15s;
  }
  .search-input:focus { border-color: var(--accent); }
  .search-input::placeholder { color: var(--text-dim); }
  .record-btn {
    background: var(--accent-dim);
    border: none;
    border-radius: 8px;
    color: var(--text);
    cursor: pointer;
    font-size: 0.85rem;
    padding: 0.75rem 1.1rem;
    white-space: nowrap;
    transition: background 0.15s;
  }
  .record-btn:hover { background: var(--accent); }
  .record-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Layout */
  .grid { display: grid; grid-template-columns: 1fr 1.4fr; gap: 1.25rem; }
  @media (max-width: 680px) { .grid { grid-template-columns: 1fr; } }

  /* Panels */
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }
  .panel-header {
    padding: 0.65rem 1rem;
    background: rgba(124,106,247,0.08);
    border-bottom: 1px solid var(--border);
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-dim);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .panel-header .count { color: var(--accent); font-family: var(--mono); }

  /* Suggestion list */
  .suggestion-list { padding: 0.4rem 0; }
  .suggestion-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    cursor: pointer;
    transition: background 0.1s;
    border-left: 2.5px solid transparent;
  }
  .suggestion-item:hover { background: rgba(124,106,247,0.1); border-left-color: var(--accent); }
  .suggestion-item.selected { background: rgba(124,106,247,0.15); border-left-color: var(--accent); }
  .rank {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-dim);
    min-width: 1.5rem;
    text-align: right;
  }
  .word {
    font-family: var(--mono);
    font-size: 0.95rem;
    flex: 1;
  }
  .conf-bar-wrap { width: 60px; }
  .conf-bar {
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent-dim));
  }
  .conf-num {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--text-dim);
    min-width: 3rem;
    text-align: right;
  }
  .hist-badge {
    font-family: var(--mono);
    font-size: 0.7rem;
    background: rgba(74,222,128,0.15);
    color: var(--green);
    border-radius: 3px;
    padding: 0.05rem 0.35rem;
    min-width: 1.6rem;
    text-align: center;
  }

  /* Explanation panel */
  .explain-panel { padding: 0; }
  .explain-empty {
    padding: 2rem 1rem;
    text-align: center;
    color: var(--text-dim);
    font-size: 0.85rem;
  }
  .explain-word {
    padding: 0.75rem 1rem 0.5rem;
    font-family: var(--mono);
    font-size: 1rem;
    color: var(--text);
    border-bottom: 1px solid var(--border);
  }
  .explain-word .ex-label { font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.06em; display: block; margin-bottom: 0.2rem; }
  .score-row {
    display: flex;
    align-items: center;
    padding: 0.55rem 1rem;
    gap: 0.75rem;
    border-bottom: 1px solid rgba(42,45,58,0.5);
    font-size: 0.84rem;
  }
  .score-row:last-child { border-bottom: none; }
  .score-label { color: var(--text-dim); min-width: 5.5rem; font-size: 0.8rem; }
  .score-value { font-family: var(--mono); font-size: 0.85rem; color: var(--text); min-width: 3.5rem; }
  .score-bar-wrap { flex: 1; height: 6px; background: rgba(42,45,58,0.8); border-radius: 3px; overflow: hidden; }
  .score-bar { height: 100%; border-radius: 3px; transition: width 0.3s ease; }
  .bar-base { background: var(--accent); }
  .bar-boost { background: var(--green); }
  .bar-final { background: linear-gradient(90deg, var(--accent), var(--green)); }
  .score-pct { font-family: var(--mono); font-size: 0.72rem; color: var(--text-dim); min-width: 2.8rem; text-align: right; }

  .components {
    padding: 0.4rem 1rem 0.75rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
  }
  .component-chip {
    font-family: var(--mono);
    font-size: 0.7rem;
    background: rgba(42,45,58,0.9);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    color: var(--text-dim);
  }
  .component-chip span { color: var(--text); }

  /* Status / recording */
  .status-row {
    margin-bottom: 1.25rem;
    height: 1.4rem;
    font-size: 0.82rem;
    color: var(--text-dim);
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; flex-shrink: 0; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
  .flash { animation: flash 0.8s ease; }
  @keyframes flash { 0%{color:var(--green)} 100%{color:var(--text-dim)} }

  /* Compare section */
  .compare-section { margin-top: 1.25rem; }
  .compare-toggle {
    background: none;
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-dim);
    cursor: pointer;
    font-size: 0.8rem;
    padding: 0.4rem 0.9rem;
    transition: border-color 0.15s, color 0.15s;
    margin-bottom: 0.75rem;
  }
  .compare-toggle:hover, .compare-toggle.active { border-color: var(--accent); color: var(--text); }
  .compare-table-wrap { overflow-x: auto; }
  table.compare {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
    font-family: var(--mono);
  }
  table.compare th {
    text-align: left;
    padding: 0.4rem 0.6rem;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    font-weight: 600;
    white-space: nowrap;
  }
  table.compare td {
    padding: 0.35rem 0.6rem;
    border-bottom: 1px solid rgba(42,45,58,0.4);
    vertical-align: middle;
  }
  table.compare tr:last-child td { border-bottom: none; }
  .rank-cell { color: var(--accent); }
  .dash { color: var(--text-dim); }
  .missing { opacity: 0.35; }
  .col-word { color: var(--text); font-weight: 500; }
  .col-num { text-align: right; }
  .col-boost-pos { color: var(--green); }
  .col-boost-zero { color: var(--text-dim); }

  /* Empty / loading */
  .loading { padding: 1.5rem 1rem; text-align: center; color: var(--text-dim); font-size: 0.85rem; }
  .empty { padding: 1.5rem 1rem; text-align: center; color: var(--text-dim); font-size: 0.85rem; }
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>adaptive-autocomplete <span class="preset-badge" id="preset-badge">production</span></h1>
    <p>Type a prefix to see live suggestions, confidence scores, and score breakdowns. Click a suggestion to record it - watch the rankings shift.</p>
  </div>

  <div class="search-row">
    <input
      id="query"
      class="search-input"
      type="text"
      placeholder="start typing…  try  prog,  recieve,  helo"
      autofocus
      autocomplete="off"
      spellcheck="false"
    >
    <button class="record-btn" id="record-btn" disabled>record selection</button>
  </div>

  <div class="status-row">
    <div class="dot"></div>
    <span id="status">ready</span>
  </div>

  <div class="grid">
    <!-- Left: suggestions -->
    <div class="panel">
      <div class="panel-header">
        suggestions
        <span class="count" id="sug-count"></span>
      </div>
      <div id="suggestions" class="suggestion-list">
        <div class="empty">type something to see completions</div>
      </div>
    </div>

    <!-- Right: explain -->
    <div class="panel">
      <div class="panel-header">
        score breakdown
        <span id="explain-word-label" style="font-size:0.75rem;color:var(--accent);font-family:var(--mono)"></span>
      </div>
      <div id="explain-panel" class="explain-panel">
        <div class="explain-empty">click a suggestion to see its score breakdown</div>
      </div>
    </div>
  </div>

  <!-- Compare presets -->
  <div class="compare-section">
    <button class="compare-toggle" id="compare-toggle">compare presets ↓</button>
    <div id="compare-wrap" style="display:none">
      <div class="panel">
        <div class="panel-header">
          preset comparison - same query, all presets side-by-side
          <span class="count" id="compare-count"></span>
        </div>
        <div class="compare-table-wrap">
          <div id="compare-table" class="loading">loading…</div>
        </div>
      </div>
    </div>
  </div>

</div>

<script>
const API = "";  // same origin
let debounceTimer = null;
let lastQuery = "";
let selectedWord = null;
let compareOpen = false;
let lastExplainData = [];

// --- API calls ---

async function apiFetch(path) {
  try {
    const r = await fetch(API + path);
    if (!r.ok) throw new Error(r.statusText);
    return await r.json();
  } catch (e) {
    setStatus("error: " + e.message, true);
    return null;
  }
}

// --- Main query handler ---

async function onQuery(q) {
  if (!q.trim()) {
    renderSuggestions([]);
    renderExplain(null);
    setStatus("ready");
    return;
  }

  setStatus("fetching…");
  const [suggestions, explanations] = await Promise.all([
    apiFetch(`/suggest?q=${encodeURIComponent(q)}&limit=15`),
    apiFetch(`/explain?q=${encodeURIComponent(q)}&limit=15`),
  ]);

  if (!suggestions || !explanations) return;

  lastExplainData = explanations;
  renderSuggestions(suggestions, explanations);

  if (selectedWord && suggestions.some(s => s.word === selectedWord)) {
    const exp = explanations.find(e => e.value === selectedWord);
    if (exp) renderExplain(exp, explanations);
  } else {
    selectedWord = null;
    renderExplain(null);
    document.getElementById("record-btn").disabled = true;
  }

  setStatus(`${suggestions.length} suggestions for "${q}"`);

  if (compareOpen) {
    loadCompare(q);
  }
}

// --- Suggestions renderer ---

function renderSuggestions(suggestions, explanations) {
  const container = document.getElementById("suggestions");
  document.getElementById("sug-count").textContent = suggestions.length || "";

  if (!suggestions.length) {
    container.innerHTML = `<div class="empty">no suggestions - try a different prefix</div>`;
    return;
  }

  const maxFinal = Math.max(...(explanations || []).map(e => e.final), 1e-9);

  container.innerHTML = suggestions.map((s, i) => {
    const conf = s.confidence ?? (explanations ? (explanations.find(e=>e.value===s.word)?.final ?? 0) / maxFinal : 0);
    const count = s.count ?? 0;
    const pct = Math.round(conf * 100);
    const barW = Math.max(pct, 2);
    const isSelected = s.word === selectedWord;
    return `
      <div class="suggestion-item ${isSelected ? "selected" : ""}"
           data-word="${esc(s.word)}"
           onclick="selectWord('${esc(s.word)}')">
        <span class="rank">${i+1}</span>
        <span class="word">${esc(s.word)}</span>
        ${count > 0 ? `<span class="hist-badge">${count}</span>` : ""}
        <div class="conf-bar-wrap"><div class="conf-bar" style="width:${barW}%"></div></div>
        <span class="conf-num">${pct}%</span>
      </div>`;
  }).join("");
}

// --- Explain renderer ---

function renderExplain(exp, allExps) {
  const panel = document.getElementById("explain-panel");
  const label = document.getElementById("explain-word-label");

  if (!exp) {
    panel.innerHTML = `<div class="explain-empty">click a suggestion to see its score breakdown</div>`;
    label.textContent = "";
    return;
  }

  label.textContent = exp.value;

  const maxFinal = allExps ? Math.max(...allExps.map(e => e.final), 1e-9) : exp.final || 1;
  const basePct  = Math.round((exp.base  / maxFinal) * 100);
  const boostPct = Math.round((exp.boost / maxFinal) * 100);
  const finalPct = Math.round((exp.final / maxFinal) * 100);

  const boostColor = exp.boost > 0.001 ? "var(--green)" : "var(--text-dim)";
  const boostSign  = exp.boost >= 0 ? "+" : "";

  // Base components chips
  const baseChips = Object.entries(exp.base_components || {}).map(([k, v]) =>
    `<div class="component-chip">${esc(k)} <span>${v.toFixed(3)}</span></div>`
  ).join("");
  const boostChips = Object.entries(exp.history_components || {}).map(([k, v]) =>
    `<div class="component-chip">${esc(k)} <span style="color:var(--green)">${v >= 0 ? "+" : ""}${v.toFixed(3)}</span></div>`
  ).join("");

  panel.innerHTML = `
    <div class="explain-word">
      <span class="ex-label">suggestion</span>
      ${esc(exp.value)}
    </div>
    <div class="score-row">
      <span class="score-label">base score</span>
      <span class="score-value">${exp.base.toFixed(4)}</span>
      <div class="score-bar-wrap"><div class="score-bar bar-base" style="width:${Math.max(basePct,1)}%"></div></div>
      <span class="score-pct">${basePct}%</span>
    </div>
    <div class="score-row">
      <span class="score-label">history boost</span>
      <span class="score-value" style="color:${boostColor}">${boostSign}${exp.boost.toFixed(4)}</span>
      <div class="score-bar-wrap"><div class="score-bar bar-boost" style="width:${Math.max(boostPct,0)}%;background:${boostColor}"></div></div>
      <span class="score-pct" style="color:${boostColor}">${boostPct > 0 ? "+" : ""}${boostPct}%</span>
    </div>
    <div class="score-row">
      <span class="score-label">final score</span>
      <span class="score-value">${exp.final.toFixed(4)}</span>
      <div class="score-bar-wrap"><div class="score-bar bar-final" style="width:${Math.max(finalPct,1)}%"></div></div>
      <span class="score-pct">${finalPct}%</span>
    </div>
    ${baseChips || boostChips ? `
    <div class="components">
      ${baseChips}
      ${boostChips}
    </div>` : ""}
  `;
}

// --- Selection recording ---

async function selectWord(word) {
  selectedWord = word;
  document.getElementById("record-btn").disabled = false;
  // Update selected style
  document.querySelectorAll(".suggestion-item").forEach(el => {
    el.classList.toggle("selected", el.dataset.word === word);
  });
  // Show explain for this word
  const exp = lastExplainData.find(e => e.value === word);
  if (exp) renderExplain(exp, lastExplainData);
}

document.getElementById("record-btn").addEventListener("click", async () => {
  const q = document.getElementById("query").value.trim();
  if (!q || !selectedWord) return;

  const btn = document.getElementById("record-btn");
  btn.disabled = true;
  btn.textContent = "recording…";

  await apiFetch(`/record?q=${encodeURIComponent(q)}&value=${encodeURIComponent(selectedWord)}`);

  btn.textContent = "recorded ✓";
  setStatus(`recorded "${selectedWord}" for "${q}"`, false, true);

  // Re-query immediately so the boost is visible
  await onQuery(q);
  btn.textContent = "record selection";
  btn.disabled = !selectedWord;
});

// --- Compare presets ---

document.getElementById("compare-toggle").addEventListener("click", () => {
  compareOpen = !compareOpen;
  const wrap = document.getElementById("compare-wrap");
  const btn = document.getElementById("compare-toggle");
  wrap.style.display = compareOpen ? "block" : "none";
  btn.classList.toggle("active", compareOpen);
  btn.textContent = compareOpen ? "compare presets ↑" : "compare presets ↓";
  if (compareOpen && lastQuery) loadCompare(lastQuery);
});

async function loadCompare(q) {
  if (!q.trim()) return;
  const el = document.getElementById("compare-table");
  el.className = "loading";
  el.textContent = "building preset engines…";

  const data = await apiFetch(`/compare?q=${encodeURIComponent(q)}&limit=8`);
  if (!data) return;

  document.getElementById("compare-count").textContent = `${data.presets.length} presets`;
  el.className = "";

  if (!data.rows.length) {
    el.innerHTML = `<div class="empty">no results</div>`;
    return;
  }

  const presets = data.presets;
  let html = `<table class="compare"><thead><tr>
    <th>suggestion</th>`;
  for (const p of presets) {
    html += `<th colspan="3" style="text-align:center;border-left:1px solid var(--border)">${esc(p)}</th>`;
  }
  html += `</tr><tr><th></th>`;
  for (const _ of presets) {
    html += `<th class="col-num" style="border-left:1px solid var(--border)">rank</th><th class="col-num">base</th><th class="col-num">boost</th>`;
  }
  html += `</tr></thead><tbody>`;

  for (const row of data.rows) {
    html += `<tr><td class="col-word">${esc(row.value)}</td>`;
    for (const p of presets) {
      const rank  = row.ranks[p];
      const base  = row.base_scores[p];
      const boost = row.boosts[p];
      if (rank === null) {
        html += `<td class="col-num dash missing" style="border-left:1px solid var(--border)">-</td><td class="col-num dash missing">-</td><td class="col-num dash missing">-</td>`;
      } else {
        const boostClass = (boost > 0.0005) ? "col-boost-pos" : "col-boost-zero";
        const boostStr = boost > 0.0005 ? `+${boost.toFixed(3)}` : boost.toFixed(3);
        html += `<td class="col-num rank-cell" style="border-left:1px solid var(--border)">#${rank}</td>`;
        html += `<td class="col-num">${base.toFixed(3)}</td>`;
        html += `<td class="col-num ${boostClass}">${boostStr}</td>`;
      }
    }
    html += `</tr>`;
  }

  html += `</tbody></table>`;
  el.innerHTML = html;
}

// --- Utilities ---

function esc(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function setStatus(msg, isError, flash) {
  const el = document.getElementById("status");
  el.textContent = msg;
  if (isError) el.style.color = "var(--red)";
  else if (flash) { el.style.color = "var(--green)"; setTimeout(() => el.style.color = "", 1200); }
  else el.style.color = "";
}

// --- Input handler ---

document.getElementById("query").addEventListener("input", e => {
  const q = e.target.value.trim();
  lastQuery = q;
  selectedWord = null;
  document.getElementById("record-btn").disabled = true;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => onQuery(q), 120);
});

// --- Bootstrap ---

document.getElementById("preset-badge").textContent = window._PRESET || "production";
</script>
</body>
</html>
"""


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
                limit = int(qs.get("limit", ["15"])[0])
                if not q:
                    self._send_json([])
                    return
                results = self._engine.suggest_with_history(q, limit=limit)
                with_conf = self._engine.suggest_with_confidence(q, limit=limit)
                conf_map = {w: c for w, c in with_conf}
                data = [
                    {"word": word, "count": count, "confidence": conf_map.get(word, 0.0)}
                    for word, count in results
                ]
                self._send_json(data)
            elif path == "/explain":
                q = (qs.get("q", [""])[0]).strip()
                limit = int(qs.get("limit", ["15"])[0])
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
            elif path == "/record":
                q = (qs.get("q", [""])[0]).strip()
                value = (qs.get("value", [""])[0]).strip()
                if q and value:
                    self._engine.record_selection(q, value)
                self._send_json({"recorded": bool(q and value)})
            elif path == "/compare":
                q = (qs.get("q", [""])[0]).strip()
                limit = int(qs.get("limit", ["8"])[0])
                if not q:
                    self._send_json({"presets": [], "rows": []})
                    return

                preset_names = list(self._cmp_engines.keys())

                # Run explain() on each cached engine - fast because engines
                # are already built.
                explanations_by_preset: dict[str, list[object]] = {
                    name: eng.explain(q)[:limit]
                    for name, eng in self._cmp_engines.items()
                }

                seen: dict[str, None] = {}
                for name in preset_names:
                    for exp in explanations_by_preset[name]:
                        seen.setdefault(exp.value, None)
                all_values = list(seen)

                lookup: dict[str, dict[str, tuple[int, object]]] = {}
                for name in preset_names:
                    lookup[name] = {
                        exp.value: (i + 1, exp)
                        for i, exp in enumerate(explanations_by_preset[name])
                    }

                rows = []
                for value in all_values:
                    ranks: dict[str, object] = {}
                    bases: dict[str, object] = {}
                    boosts: dict[str, object] = {}
                    finals: dict[str, object] = {}
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
            else:
                self.send_response(404)
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


def _find_free_port(preferred: int) -> int:
    """Return preferred port if free, otherwise any free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    *,
    engine: AutocompleteEngine,
    port: int = 8421,
    preset: str = "production",
    no_browser: bool = False,
) -> None:
    """
    Start the interactive demo server and open it in the default browser.

    Parameters:
        engine:     The engine to use for suggestions and explanations.
        port:       Preferred local port.  If occupied, a free port is used.
        preset:     Name of the active preset (display only).
        no_browser: If True, print the URL but do not open the browser.
    """
    port = _find_free_port(port)
    url = f"http://127.0.0.1:{port}"

    handler_class = _make_handler(engine, preset)
    server = HTTPServer(("127.0.0.1", port), handler_class)

    print("\nadaptive-autocomplete demo")
    print(f"  preset:  {preset}")
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
