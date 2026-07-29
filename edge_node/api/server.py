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
from datetime import datetime, timezone, timedelta

# Vietnam timezone (UTC+7)
TZ_VIETNAM = timezone(timedelta(hours=7))
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from edge_node.settings import get_settings
from edge_node.core.contracts import Point, CrossingDirection
from edge_node.core.config import TripwireConfig, set_active_tripwire, get_active_tripwire


LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="Edge Node Control Plane",
    description="Health checks and administrative actions for the Edge Node.",
    version="1.0.0",
)

# CORS middleware for web frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
            "timestamp": datetime.now(TZ_VIETNAM).isoformat(),
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
            "timestamp": datetime.now(TZ_VIETNAM).isoformat(),
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
            "timestamp": datetime.now(TZ_VIETNAM).isoformat(),
        }
    )


# ------------------------------------------------------------------ #
# Stop-line config get                                                 #
# ------------------------------------------------------------------ #
@app.get("/action/stop-line", summary="Get current stop line")
def get_stop_line() -> JSONResponse:
    """Return the currently active tripwire configuration."""
    tw = get_active_tripwire()
    if tw is None:
        return JSONResponse(status_code=404, content={"error": "No stop line configured"})
    return JSONResponse(content={
        "start": {"x": tw.start.x, "y": tw.start.y},
        "end": {"x": tw.end.x, "y": tw.end.y},
        "direction": tw.direction.value,
    })


# ------------------------------------------------------------------ #
# Light ROI config                                                     #
# ------------------------------------------------------------------ #
_light_roi: dict = {"x": 0, "y": 0, "w": 0, "h": 0}


class LightROIReq(BaseModel):
    x: int
    y: int
    w: int
    h: int


@app.get("/api/light-roi", summary="Get light ROI")
def get_light_roi() -> JSONResponse:
    return JSONResponse(content=_light_roi)


@app.post("/action/light-roi", summary="Set light ROI")
def set_light_roi(req: LightROIReq) -> JSONResponse:
    global _light_roi
    if req.w <= 0 or req.h <= 0:
        return JSONResponse(status_code=400, content={"error": "ROI w/h must be positive"})
    _light_roi = {"x": req.x, "y": req.y, "w": req.w, "h": req.h}
    LOGGER.info("Light ROI updated: %s", _light_roi)
    return JSONResponse(
        content={
            "message": "Light ROI updated",
            "timestamp": datetime.now(TZ_VIETNAM).isoformat(),
        }
    )


# ------------------------------------------------------------------ #
# Camera endpoints                                                     #
# ------------------------------------------------------------------ #
@app.get("/api/cameras", summary="List available cameras")
def list_cameras() -> JSONResponse:
    """Return a list of cameras available on this edge node."""
    settings = get_settings()
    hls_dir = Path(settings.fake_camera_hls_dir)
    playlist = hls_dir / "stream.m3u8"

    cameras = [
        {
            "id": "fake-cam-1",
            "name": "Fake Camera 1 (aziz1.MP4)",
            "status": "active" if playlist.exists() else "starting",
            "resolution": "1920x1080",
            "location": "Intersection Demo",
            "stream_url": f"/edge-api/cameras/fake-cam-1/stream",
            "snapshot_url": f"/edge-api/cameras/fake-cam-1/snapshot",
        }
    ]
    return JSONResponse(content=cameras)


@app.get("/api/cameras/{camera_id}/stream", summary="HLS stream playlist")
def camera_stream(camera_id: str):
    """Serve the HLS m3u8 playlist file."""
    settings = get_settings()
    hls_dir = Path(settings.fake_camera_hls_dir)
    playlist = hls_dir / "stream.m3u8"

    if not playlist.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Stream not ready for camera {camera_id}.",
        )

    content = playlist.read_text()
    return Response(
        content=content,
        media_type="application/vnd.apple.mpegurl",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get(
    "/api/cameras/{camera_id}/stream/{filename}",
    summary="HLS segment file (with /stream/ prefix)",
)
def camera_stream_segment(camera_id: str, filename: str):
    return _serve_segment(camera_id, filename)


@app.get(
    "/api/cameras/{camera_id}/{filename}",
    summary="HLS segment file (relative to stream URL, no /stream/)",
)
def camera_segment_relative(camera_id: str, filename: str):
    """Serve HLS segment when resolved relative to playlist URL (RFC 3986)."""
    if not filename.endswith((".ts", ".m3u8", ".tmp")):
        raise HTTPException(status_code=404, detail="Not a segment file")
    return _serve_segment(camera_id, filename)


def _serve_segment(camera_id: str, filename: str):
    settings = get_settings()
    hls_dir = Path(settings.fake_camera_hls_dir)
    segment = hls_dir / filename

    if not segment.exists():
        raise HTTPException(status_code=404, detail=f"Segment not found: {filename}")

    if filename.endswith(".ts"):
        media_type = "video/mp2t"
    elif filename.endswith(".m3u8"):
        media_type = "application/vnd.apple.mpegurl"
    else:
        media_type = "application/octet-stream"

    return FileResponse(
        path=str(segment),
        media_type=media_type,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/api/cameras/{camera_id}/snapshot", summary="Camera snapshot")
def camera_snapshot(camera_id: str):
    """Capture a frame from the video source as JPEG."""
    settings = get_settings()
    video_path = settings.fake_camera_video

    if not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    try:
        import cv2
    except ImportError:
        raise HTTPException(status_code=500, detail="OpenCV not available")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Cannot open video file")

    try:
        ok, frame = cap.read()
        if not ok:
            raise HTTPException(status_code=500, detail="Cannot read frame")
        _, jpeg = cv2.imencode(".jpg", frame)
        return Response(
            content=jpeg.tobytes(),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": "*",
            },
        )
    finally:
        cap.release()

