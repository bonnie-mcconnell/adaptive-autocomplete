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
    POST /record?q=prog&value=programming  → {"recorded": true}
    GET /health                       → {"status": "ok", "history_entries": N}

The engine loads persisted history on startup and saves it on shutdown.
Selections recorded via POST /record are visible immediately in subsequent
GET /suggest calls within the same process, and survive restarts.
"""
from __future__ import annotations

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
from aac.engine.engine import AutocompleteEngine
from aac.presets import create_engine
from aac.storage.json_store import JsonHistoryStore

# ---------------------------------------------------------------------------
# Configuration — override with environment variables
# ---------------------------------------------------------------------------

HISTORY_PATH = Path(os.environ.get("AAC_HISTORY_PATH", "~/.aac_history.json")).expanduser()
PRESET = os.environ.get("AAC_PRESET", "production")
DEFAULT_LIMIT = int(os.environ.get("AAC_DEFAULT_LIMIT", "10"))

# ---------------------------------------------------------------------------
# Application state — initialised in lifespan
# ---------------------------------------------------------------------------

_store: JsonHistoryStore
_engine_instance: AutocompleteEngine | None = None


def get_engine() -> AutocompleteEngine:
    """
    Dependency-style accessor for the shared engine instance.

    Raises RuntimeError if called before the lifespan context has initialised
    the engine — i.e. if called outside a running FastAPI application.
    """
    if _engine_instance is None:  # pragma: no cover
        raise RuntimeError("Engine not initialised — lifespan not running")
    return _engine_instance


# ---------------------------------------------------------------------------
# Lifespan: load history on startup, save on shutdown
# ---------------------------------------------------------------------------

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _store, _engine_instance

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _store = JsonHistoryStore(HISTORY_PATH)

    # ThreadSafeHistory wraps a loaded History for concurrent record() calls
    ts_history = ThreadSafeHistory(_store.load())
    _engine_instance = create_engine(PRESET, history=ts_history)

    print(
        f"adaptive-autocomplete: {PRESET} preset loaded, "
        f"{len(list(ts_history.entries()))} history entries from {HISTORY_PATH}"
    )

    yield  # application runs here

    # Save on shutdown
    _store.save(ts_history.snapshot_history())
    print(f"adaptive-autocomplete: history saved to {HISTORY_PATH}")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="adaptive-autocomplete",
    description="Autocomplete API with learning and explainability",
    version="1.0.0",
    lifespan=lifespan,
)


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
) -> list[dict[str, float | str]]:
    """
    Return per-suggestion score breakdowns.

    Shows how base frequency score and recency/history boost combine
    into the final ranking score for each suggestion.
    """
    engine = get_engine()
    explanations = await engine.explain_async(q)
    return [
        {
            "value": e.value,
            "base": round(e.base_score, 4),
            "boost": round(e.history_boost, 4),
            "final": round(e.final_score, 4),
        }
        for e in explanations[:limit]
    ]


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
    to persist mid-session — useful before scaling down or deploying.
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
