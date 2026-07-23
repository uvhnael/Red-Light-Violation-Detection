"""Local FastAPI Control Plane for the Edge Node.

Endpoints allow the Central Server to monitor the health of this node
and trigger administrative actions without SSH access.

Run standalone::

    uvicorn edge_node.api.server:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from edge_node.settings import get_settings
from edge_node.core.contracts import Point, CrossingDirection
from edge_node.core.config import TripwireConfig, set_active_tripwire


LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="Edge Node Control Plane",
    description="Health checks and administrative actions for the Edge Node.",
    version="1.0.0",
)

_START_TIME = time.monotonic()


def _check_redis() -> Dict[str, Any]:
    """Ping the local Redis instance."""
    try:
        import redis as redis_lib

        settings = get_settings()
        client = redis_lib.from_url(settings.redis_url, socket_timeout=2)
        client.ping()
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def _check_celery() -> Dict[str, Any]:
    """Check whether at least one Celery worker is responding."""
    try:
        from edge_node.worker.celery_app import app as celery_app

        inspector = celery_app.control.inspect(timeout=2.0)
        pings = inspector.ping()
        if pings:
            return {"status": "ok", "workers": len(pings)}
        return {"status": "warning", "detail": "no workers responded"}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def _check_camera() -> Dict[str, Any]:
    """Best-effort camera availability check."""
    try:
        import cv2

        settings = get_settings()
        # For RTSP/file input this is a quick open-and-release test.
        # If the input is a device index, convert accordingly.
        cap = cv2.VideoCapture(0)
        opened = cap.isOpened()
        cap.release()
        return {"status": "ok" if opened else "error", "opened": opened}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@app.get("/health", summary="Health check")
def health() -> JSONResponse:
    """Return the health status of the Node, Redis, and Celery."""
    redis_status = _check_redis()
    celery_status = _check_celery()
    camera_status = _check_camera()

    overall = "ok"
    for component in (redis_status, celery_status, camera_status):
        if component["status"] == "error":
            overall = "degraded"
            break
        if component["status"] == "warning":
            overall = "degraded"

    settings = get_settings()
    uptime_seconds = round(time.monotonic() - _START_TIME, 1)

    return JSONResponse(
        status_code=200 if overall == "ok" else 503,
        content={
            "status": overall,
            "node_id": settings.node_id,
            "uptime_seconds": uptime_seconds,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "redis": redis_status,
                "celery": celery_status,
                "camera": camera_status,
            },
        },
    )


@app.post("/action/restart", summary="Restart edge-node services")
def restart_services() -> JSONResponse:
    """Force-restart the pipeline, Celery worker, and Redis services.

    This endpoint is invoked by the Central Server when the node
    appears unresponsive.  It uses ``supervisorctl`` when available,
    falling back to ``systemctl``.
    """
    results: Dict[str, Any] = {}

    # Restart Celery worker
    try:
        subprocess.run(
            ["supervisorctl", "restart", "celery-worker"],
            capture_output=True,
            timeout=15,
            check=False,
        )
        results["celery"] = "restart requested"
    except FileNotFoundError:
        try:
            subprocess.run(
                ["systemctl", "restart", "edge-celery"],
                capture_output=True,
                timeout=15,
                check=False,
            )
            results["celery"] = "restart requested (systemctl)"
        except Exception as exc:
            results["celery"] = f"failed: {exc}"

    # Restart pipeline process
    try:
        subprocess.run(
            ["supervisorctl", "restart", "edge-pipeline"],
            capture_output=True,
            timeout=15,
            check=False,
        )
        results["pipeline"] = "restart requested"
    except FileNotFoundError:
        try:
            subprocess.run(
                ["systemctl", "restart", "edge-pipeline"],
                capture_output=True,
                timeout=15,
                check=False,
            )
            results["pipeline"] = "restart requested (systemctl)"
        except Exception as exc:
            results["pipeline"] = f"failed: {exc}"

    LOGGER.warning("Service restart requested: %s", results)

    return JSONResponse(
        status_code=200,
        content={
            "message": "Restart commands issued",
            "results": results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


class TripwireUpdateReq(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    direction: str = "any"

@app.post("/action/stop-line", summary="Update stop line")
def update_stop_line(req: TripwireUpdateReq) -> JSONResponse:
    """Dynamically update the stop line for the pipeline."""
    try:
        direction = CrossingDirection(req.direction)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Invalid direction"})

    new_tw = TripwireConfig(
        start=Point(req.x1, req.y1),
        end=Point(req.x2, req.y2),
        direction=direction
    )
    set_active_tripwire(new_tw)
    LOGGER.info("Stop line updated via API: %s", new_tw)

    return JSONResponse(
        status_code=200,
        content={
            "message": "Stop line updated",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

