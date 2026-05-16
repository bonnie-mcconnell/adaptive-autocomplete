"""
FastAPI autocomplete endpoint using adaptive-autocomplete.

This example shows how to wire the engine into a real async web service:
persistent history across restarts, thread-safe recording, and async
suggest/explain endpoints that don't block the event loop.

Install extras:
    pip install adaptive-autocomplete fastapi uvicorn

Run:
    uvicorn examples.fastapi_app:app --reload

Endpoints:
    GET /suggest?q=prog&limit=10     → ["programming", "program", ...]
    GET /explain?q=prog&limit=5      → [{value, base, boost, final}, ...]
    GET /batch?q=prog&q=hel&q=wor   → {"prog": [...], "hel": [...], "wor": [...]}
    POST /record?q=prog&value=programming  → {"recorded": true}
    GET /health                       → {"status": "ok", "history_entries": N}

The engine loads persisted history on startup and saves it on shutdown.
Selections recorded via POST /record are visible immediately in subsequent
GET /suggest calls within the same process, and survive restarts.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

try:
    from fastapi import FastAPI, Query
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "FastAPI is required for this example. "
        "Install it with: pip install fastapi uvicorn"
    ) from e

from aac import ThreadSafeHistory
from aac import __version__ as _aac_version
from aac.engine.engine import AutocompleteEngine
from aac.presets import create_engine
from aac.storage.json_store import JsonHistoryStore

# ---------------------------------------------------------------------------
# Configuration - override with environment variables
# ---------------------------------------------------------------------------

HISTORY_PATH = Path(os.environ.get("AAC_HISTORY_PATH", "~/.aac_history.json")).expanduser()
PRESET = os.environ.get("AAC_PRESET", "production")
DEFAULT_LIMIT = int(os.environ.get("AAC_DEFAULT_LIMIT", "10"))
VOCAB_PATH = os.environ.get("AAC_VOCAB_PATH")          # None → bundled 48k English vocab
VOCAB_FORMAT = os.environ.get("AAC_VOCAB_FORMAT", "wordlist")  # "wordlist" or "text"

# ---------------------------------------------------------------------------
# Application state - initialised in lifespan
# ---------------------------------------------------------------------------

_store: JsonHistoryStore
_engine_instance: AutocompleteEngine | None = None


def get_engine() -> AutocompleteEngine:
    """
    Dependency-style accessor for the shared engine instance.

    Raises RuntimeError if called before the lifespan context has initialised
    the engine - i.e. if called outside a running FastAPI application.
    """
    if _engine_instance is None:  # pragma: no cover
        raise RuntimeError("Engine not initialised - lifespan not running")
    return _engine_instance


# ---------------------------------------------------------------------------
# Lifespan: load history on startup, save on shutdown
# ---------------------------------------------------------------------------

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _store, _engine_instance

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _store = JsonHistoryStore(HISTORY_PATH)

    # Load custom vocabulary if AAC_VOCAB_PATH is set, otherwise use bundled English vocab
    vocabulary = None
    if VOCAB_PATH:
        from aac.vocabulary import vocabulary_from_file
        vocabulary = vocabulary_from_file(VOCAB_PATH, fmt=VOCAB_FORMAT)
        print(f"adaptive-autocomplete: loaded {len(vocabulary)} words from {VOCAB_PATH}")

    # thread_safe=True wraps the loaded History in ThreadSafeHistory so that
    # concurrent suggest() and record_selection() calls from the async thread
    # pool are safe without external locking.
    _engine_instance = create_engine(
        PRESET,
        history=_store.load(),
        vocabulary=vocabulary,
        thread_safe=True,
    )
    ts_history = _engine_instance.history  # ThreadSafeHistory instance

    print(
        f"adaptive-autocomplete: {PRESET} preset loaded, "
        f"{len(ts_history)} history entries from {HISTORY_PATH}"
    )

    yield  # application runs here

    # Save on shutdown
    assert isinstance(ts_history, ThreadSafeHistory)
    _store.save(ts_history.snapshot_history())
    print(f"adaptive-autocomplete: history saved to {HISTORY_PATH}")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="adaptive-autocomplete",
    description="Autocomplete API with learning and explainability",
    version=_aac_version,
    lifespan=lifespan,
)


@app.get("/batch")
async def batch(
    q: list[str] = Query(..., description="List of prefixes to complete"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=100, description="Max suggestions per prefix"),
) -> dict[str, list[str]]:
    """
    Return suggestions for multiple prefixes in one request.

    Runs all queries concurrently via the async thread pool so total
    latency is the max of individual query latencies, not the sum.

    Example: GET /batch?q=prog&q=hel&q=wor&limit=5
    → {"prog": ["programming", ...], "hel": ["help", ...], "wor": ["word", ...]}
    """
    engine = get_engine()
    return await engine.batch_suggest_async(q, limit=limit)


@app.get("/suggest", response_model=list[str])
async def suggest(
    q: str = Query(..., min_length=1, description="Input prefix to complete"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=100, description="Max suggestions"),
) -> list[str]:
    """
    Return ranked completion suggestions for a prefix.

    Suggestions are ranked by a combination of corpus frequency, personal
    selection history, and typo recovery (for the production and robust presets).

    Example: GET /suggest?q=prog&limit=5
    → ["programming", "program", "progress", "project", "programs"]
    """
    engine = get_engine()
    return await engine.suggest_async(q, limit=limit)


@app.get("/explain")
async def explain(
    q: str = Query(..., min_length=1, description="Input prefix to explain"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=100, description="Max suggestions"),
) -> list[dict[str, object]]:
    """
    Return per-suggestion score breakdowns.

    Shows how base frequency score and recency/history boost combine
    into the final ranking score for each suggestion, with per-predictor
    component breakdown and percentage contribution.

    Returns the full schema from ``engine.explain_as_dicts()``:
    ``value``, ``base_score``, ``history_boost``, ``final_score``,
    ``source``, ``sources``, ``base_components``, ``history_components``,
    ``contribution_pct``.
    """
    engine = get_engine()
    # explain_as_dicts() is synchronous. Run in a thread to avoid blocking
    # the event loop, consistent with the engine's async API pattern.
    results = await asyncio.to_thread(engine.explain_as_dicts, q)
    return results[:limit]


@app.post("/record")
async def record(
    q: str = Query(..., min_length=1, description="Input prefix the user was completing"),
    value: str = Query(..., min_length=1, description="Completion the user selected"),
) -> dict[str, bool]:
    """
    Record a user selection. The engine learns immediately.

    Selections are held in memory and persisted to disk on process shutdown.
    To persist immediately (e.g. in a serverless environment), call
    POST /save after recording.
    """
    engine = get_engine()
    await engine.record_selection_async(q, value)
    return {"recorded": True}


@app.post("/save")
async def save() -> dict[str, str]:
    """
    Flush history to disk immediately.

    Normally history is saved on process shutdown. Call this endpoint
    to persist mid-session - useful before scaling down or deploying.
    """
    engine = get_engine()
    history = engine.history
    if isinstance(history, ThreadSafeHistory):
        _store.save(history.snapshot_history())
    else:
        _store.save(history)
    return {"saved": str(HISTORY_PATH)}


@app.get("/health")
async def health() -> dict[str, object]:
    """Health check with basic engine statistics."""
    engine = get_engine()
    state = engine.describe()
    return {
        "status": "ok",
        "preset": PRESET,
        "history_entries": state["history_entries"],
        "predictors": [p["name"] for p in state["predictors"]],
    }


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("examples.fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
