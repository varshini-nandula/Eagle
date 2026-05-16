"""
apps/backend/main.py — Eagle FastAPI backend entry point.

Phase 5 scaffold: health check + active tracks endpoint.

Endpoints
---------
GET /health
    Returns server + Redis connectivity status.
    Responds with {"status": "ok", "redis": "connected"} when healthy,
    or {"status": "degraded", "redis": "<error>"} when Redis is unreachable.

GET /tracks?camera_id=cam_01
    Returns active track IDs for the given camera by scanning Redis keys
    matching the pattern ``track:{camera_id}:*``.

GET /metrics
    Prometheus metrics scrape endpoint.
"""
from __future__ import annotations

import json
import logging
import os

import redis as redis_sync
from fastapi import FastAPI, Query
from fastapi.responses import Response
from prometheus_client import generate_latest

from apps.backend.routes.cameras import identity_router, router as cameras_router
from apps.backend.routes.feedback import router as feedback_router
from libs.observability.metrics import frames_processed_total

logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Eagle Surveillance API",
    description="Real-time semantic surveillance — detection, tracking, and reasoning.",
    version="0.1.0",
)

# ── Redis (sync client for simple health / track-list queries) ────────────────

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

_redis: redis_sync.Redis | None = None


def _get_redis() -> redis_sync.Redis | None:
    """
    Return a lazily-initialised sync Redis client, or None if the connection
    has never succeeded.  Errors are logged but never raised — callers must
    handle a None return.
    """
    global _redis
    if _redis is None:
        try:
            client = redis_sync.from_url(REDIS_URL, socket_connect_timeout=2)
            client.ping()
            _redis = client
            logger.info("Redis connected at %s", REDIS_URL)
        except Exception as exc:
            logger.warning("Redis unavailable: %s", exc)
    return _redis


# Attempt connection at startup (non-fatal if Redis is down).
_get_redis()

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(cameras_router)
app.include_router(identity_router)
app.include_router(feedback_router)


# ── Health endpoint ───────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health() -> dict:
    """
    Return server and Redis health.

    Responses
    ---------
    Healthy::

        {"status": "ok", "redis": "connected"}

    Degraded (Redis unreachable)::

        {"status": "degraded", "redis": "<error message>"}
    """
    r = _get_redis()
    if r is None:
        return {"status": "degraded", "redis": "unavailable"}

    try:
        r.ping()
        return {"status": "ok", "redis": "connected"}
    except Exception as exc:
        # Redis was reachable at startup but is now down.
        global _redis
        _redis = None          # force reconnect attempt on next call
        return {"status": "degraded", "redis": str(exc)}


# ── Tracks endpoint ───────────────────────────────────────────────────────────

@app.get("/tracks", tags=["tracks"])
async def list_active_tracks(
    camera_id: str = Query(default="cam_01", description="Camera identifier"),
) -> dict:
    """
    Return active track IDs for a camera.

    Scans Redis for keys matching ``track:{camera_id}:*`` and returns the
    integer track IDs whose stored state is ``ACTIVE``.

    Query Parameters
    ----------------
    camera_id : str
        Camera identifier (default: ``cam_01``).

    Responses
    ---------
    Redis healthy::

        {"camera_id": "cam_01", "track_ids": [1, 3, 7]}

    Redis unavailable::

        {"camera_id": "cam_01", "track_ids": [], "error": "Redis unavailable"}
    """
    r = _get_redis()
    if r is None:
        return {"camera_id": camera_id, "track_ids": [], "error": "Redis unavailable"}

    try:
        pattern = f"track:{camera_id}:*"
        keys: list[bytes] = r.keys(pattern)

        active_ids: list[int] = []
        for key in keys:
            raw = r.get(key)
            if raw is None:
                continue
            try:
                record = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue
            if record.get("state") == "ACTIVE":
                active_ids.append(int(record["track_id"]))

        return {"camera_id": camera_id, "track_ids": sorted(active_ids)}

    except Exception as exc:
        logger.error("Failed to list tracks for %s: %s", camera_id, exc)
        return {"camera_id": camera_id, "track_ids": [], "error": str(exc)}


# ── Metrics endpoint ──────────────────────────────────────────────────────────

@app.get("/metrics", tags=["ops"], include_in_schema=False)
async def metrics() -> Response:
    """Prometheus metrics scrape endpoint."""
    return Response(generate_latest(), media_type="text/plain")
