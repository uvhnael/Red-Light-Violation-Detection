"""Control-plane FastAPI cục bộ cho Edge Node.

Các endpoint cho phép Central Server theo dõi sức khoẻ node và kích hoạt
các tác vụ quản trị mà không cần truy cập SSH.

Chạy độc lập::

    uvicorn edge_node.api.server:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import hmac
import logging
import os
import signal
import subprocess
import time
from datetime import datetime, timezone, timedelta

# Múi giờ Việt Nam (UTC+7)
TZ_VIETNAM = timezone(timedelta(hours=7))
from pathlib import Path
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from edge_node.settings import get_settings
from edge_node.core.contracts import Point, CrossingDirection
from edge_node.core.config import (
    TripwireConfig,
    set_active_tripwire,
    get_active_tripwire,
    set_active_light_roi,
    get_active_light_roi,
)


LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="Edge Node Control Plane",
    description="Health checks and administrative actions for the Edge Node.",
    version="1.0.0",
)

# CORS middleware for web frontend access.
# NOTE: allow_origins=["*"] KHÔNG được phép đi cùng allow_credentials=True
# (browser từ chối theo spec CORS). Web dashboard gọi các endpoint đọc
# (camera, calibration, light-state) không cần cookie nên tắt credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_START_TIME = time.monotonic()

# ------------------------------------------------------------------ #
# Admin token auth (bảo vệ các endpoint /action/*)                      #
# ------------------------------------------------------------------ #
# Shared secret đọc từ EDGE_API_TOKEN. Client gửi qua header
# ``X-Edge-Token`` hoặc ``Authorization: Bearer <token>``. Khi token chưa
# được cấu hình (rỗng) các endpoint vẫn mở — chế độ dev — nhưng server
# log cảnh báo một lần lúc import.
_ADMIN_TOKEN = get_settings().api_token
_TOKEN_WARNED = False


def require_admin_token(request: Request) -> None:
    """Dependency: chặn các endpoint quản trị nếu thiếu/sai token.

    So sánh constant-time để tránh timing attack.
    """
    global _TOKEN_WARNED
    if not _ADMIN_TOKEN:
        if not _TOKEN_WARNED:
            LOGGER.warning(
                "EDGE_API_TOKEN chưa được cấu hình — các endpoint /action/* "
                "đang MỞ không xác thực. Đặt EDGE_API_TOKEN trong production."
            )
            _TOKEN_WARNED = True
        return

    provided = request.headers.get("X-Edge-Token", "")
    if not provided:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()

    if not provided or not hmac.compare_digest(provided, _ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing admin token")

# ------------------------------------------------------------------ #
# Camera probe cache                                                    #
# ------------------------------------------------------------------ #
# Mở cv2.VideoCapture cho RTSP tốn 1-5 giây (connect + thương lượng
# codec) và block worker của FastAPI. Health bị central poll thường
# xuyên nên KHÔNG được mở capture mỗi request. Cache kết quả probe
# với TTL ngắn; file local thì check tồn tại là đủ (rẻ).
_CAMERA_CACHE_TTL_SECONDS = 30.0
_camera_cache: Dict[str, Any] = {"checked_at": 0.0, "result": None}
_resolution_cache: Dict[str, Any] = {"checked_at": 0.0, "value": "unknown"}


def _check_camera() -> Dict[str, Any]:
    """Best-effort camera availability check (video file / RTSP / device).

    Kết quả được cache ``_CAMERA_CACHE_TTL_SECONDS`` giây để các health
    poll liên tiếp không phải mở lại nguồn video (đặc biệt RTSP).
    """
    settings = get_settings()
    source = settings.video_input
    if not source:
        return {"status": "error", "detail": "No video source configured"}

    now = time.monotonic()
    cached = _camera_cache["result"]
    if cached is not None and (now - _camera_cache["checked_at"]) < _CAMERA_CACHE_TTL_SECONDS:
        return cached

    # Nguồn là file local: chỉ cần check tồn tại (không mở capture).
    if not (str(source).startswith(("rtsp://", "http://", "https://")) or str(source).isdigit()):
        exists = Path(source).exists()
        result = {
            "status": "ok" if exists else "error",
            "opened": exists,
            "detail": None if exists else f"Video file not found: {source}",
        }
        _camera_cache.update({"checked_at": now, "result": result})
        return result

    # RTSP / HTTP / device index: mở capture thật (đắt) nhưng có cache.
    try:
        import cv2

        if str(source).isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
        opened = cap.isOpened()
        cap.release()
        result = {"status": "ok" if opened else "error", "opened": opened}
    except Exception as exc:
        result = {"status": "error", "detail": str(exc)}

    _camera_cache.update({"checked_at": now, "result": result})
    return result


@app.get("/health", summary="Health check")
def health() -> JSONResponse:
    """Return the health status of the node and its camera source.

    Redis/Celery are intentionally NOT part of this check: violations are
    delivered through the durable outbox, so those services are no longer
    required for a healthy edge node.
    """
    camera_status = _check_camera()

    overall = "ok" if camera_status["status"] == "ok" else "degraded"

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
                "camera": camera_status,
            },
        },
    )


@app.post(
    "/action/restart",
    summary="Restart edge-node services",
    dependencies=[Depends(require_admin_token)],
)
def restart_services() -> JSONResponse:
    """Force-restart the edge pipeline.

    This endpoint is invoked by the Central Server when the node
    appears unresponsive.

    Docker-aware: khi chạy trong container (phát hiện qua ``/.dockerenv``),
    edge-pipeline là tiến trình chính của container nên cách restart đúng
    là gửi SIGTERM cho chính mình — Docker restart policy sẽ kéo container
    lên lại. ``supervisorctl``/``systemctl`` KHÔNG tồn tại trong container
    nên chỉ dùng làm fallback cho deployment bare-metal.
    """
    results: Dict[str, Any] = {}

    in_docker = Path("/.dockerenv").exists()
    if in_docker:
        # Gửi SIGTERM cho tiến trình hiện tại (PID 1 trong container).
        # Docker restart policy (restart: unless-stopped trong compose)
        # sẽ tự khởi động lại container. Trả lời trước khi signal tới.
        results["pipeline"] = "SIGTERM sent to self — Docker will restart the container"
        LOGGER.warning("Service restart requested (Docker): %s", results)
        response = JSONResponse(
            status_code=200,
            content={
                "message": "Restart initiated (Docker self-SIGTERM)",
                "results": results,
                "timestamp": datetime.now(TZ_VIETNAM).isoformat(),
            },
        )
        # Trì hoãn signal một chút để response kịp gửi đi.
        import threading

        def _deferred_sigterm() -> None:
            time.sleep(0.5)
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=_deferred_sigterm, daemon=True).start()
        return response

    # Bare-metal fallback: supervisorctl -> systemctl
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

@app.post(
    "/action/stop-line",
    summary="Update stop line",
    dependencies=[Depends(require_admin_token)],
)
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
class LightROIReq(BaseModel):
    x: int
    y: int
    w: int
    h: int


@app.get("/api/light-roi", summary="Get light ROI")
def get_light_roi() -> JSONResponse:
    roi = get_active_light_roi()
    if roi is None:
        return JSONResponse(content={"x": 0, "y": 0, "w": 0, "h": 0, "set": False})
    x, y, w, h = roi
    return JSONResponse(content={"x": x, "y": y, "w": w, "h": h, "set": True})


@app.post(
    "/action/light-roi",
    summary="Set light ROI",
    dependencies=[Depends(require_admin_token)],
)
def set_light_roi(req: LightROIReq) -> JSONResponse:
    if req.w <= 0 or req.h <= 0:
        return JSONResponse(status_code=400, content={"error": "ROI w/h must be positive"})
    set_active_light_roi((req.x, req.y, req.w, req.h))
    LOGGER.info("Light ROI updated: x=%s y=%s w=%s h=%s", req.x, req.y, req.w, req.h)
    return JSONResponse(
        content={
            "message": "Light ROI updated",
            "timestamp": datetime.now(TZ_VIETNAM).isoformat(),
        }
    )


# ------------------------------------------------------------------ #
# Calibration frame source (used by the snapshot endpoint so the web   #
# UI has an image to draw the stop line / light ROI on)                #
# ------------------------------------------------------------------ #
def _calibration_video_path() -> str:
    """Pick the video source used for calibration frames."""
    settings = get_settings()
    return settings.video_input


@app.get("/api/calibration", summary="Get current calibration state")
def get_calibration() -> JSONResponse:
    """Return the active stop line + light ROI so the web UI can draw them."""
    tw = get_active_tripwire()
    roi = get_active_light_roi()
    return JSONResponse(content={
        "stop_line": (
            {
                "start": {"x": tw.start.x, "y": tw.start.y},
                "end": {"x": tw.end.x, "y": tw.end.y},
                "direction": tw.direction.value,
            }
            if tw else None
        ),
        "light_roi": (
            {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]}
            if roi else None
        ),
    })


@app.get("/api/light-state", summary="Get live traffic-light state")
def get_light_state() -> JSONResponse:
    """Return the debounced traffic-light state published by the pipeline.

    The pipeline updates this every processed frame, so the web dashboard
    can poll it to show the live signal (red/yellow/green/unknown).
    """
    from edge_node.core.config import get_light_state as _get_light_state

    snap = _get_light_state()
    if snap is None:
        return JSONResponse(content={
            "state": "unknown",
            "confidence": 0.0,
            "stable": False,
            "frame_index": None,
            "timestamp_ms": None,
            "source": None,
            "updated": False,
        })
    return JSONResponse(content={
        "state": snap.state.value,
        "confidence": snap.confidence,
        "stable": snap.stable,
        "frame_index": snap.frame_index,
        "timestamp_ms": snap.timestamp_ms,
        "source": snap.source,
        "updated": True,
    })


@app.get("/api/calibration/snapshot", summary="Calibration frame as JPEG")
def calibration_snapshot():
    """JPEG of a frame from the configured video source.

    The web UI draws the stop line / light ROI overlays on this image so the
    pixel coordinates drawn by the operator match 1:1.
    """
    from edge_node.core.calibration import grab_calibration_frame

    try:
        import cv2
    except ImportError:
        raise HTTPException(status_code=500, detail="OpenCV not available")

    video_path = _calibration_video_path()
    if not video_path:
        raise HTTPException(status_code=400, detail="No video source configured")

    frame = grab_calibration_frame(video_path)
    if frame is None:
        raise HTTPException(status_code=500, detail="Cannot read calibration frame")

    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(
        content=jpeg.tobytes(),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ------------------------------------------------------------------ #
# Camera endpoints                                                     #
# ------------------------------------------------------------------ #
def _probe_resolution(video_path: str) -> str:
    """Best-effort WxH probe of the camera video file (cached).

    Mở VideoCapture chỉ để đọc metadata là tốn kém với RTSP, nên cache
    kết quả theo đường dẫn nguồn với TTL dài (resolution hiếm khi đổi).
    """
    now = time.monotonic()
    if (
        _resolution_cache["value"] != "unknown"
        and (now - _resolution_cache["checked_at"]) < 300.0
    ):
        return _resolution_cache["value"]

    value = "unknown"
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w > 0 and h > 0:
                value = f"{w}x{h}"
    except Exception:
        pass

    _resolution_cache.update({"checked_at": now, "value": value})
    return value


@app.get("/api/cameras", summary="List available cameras")
def list_cameras() -> JSONResponse:
    """Return the list of cameras served by this edge node.

    The edge node exposes its configured video input as a live HLS camera
    feed. The web dashboard fetches this list directly from the edge node
    (not via the central server).
    """
    settings = get_settings()
    video_path = settings.video_input

    if not video_path or not Path(video_path).exists():
        return JSONResponse(content=[])

    hls_dir = Path(settings.camera_stream_hls_dir)
    playlist = hls_dir / "stream.m3u8"

    camera_id = settings.camera_id
    name = settings.camera_name or f"Camera {settings.node_id} ({Path(video_path).name})"
    location = settings.camera_location or settings.node_id

    cameras = [
        {
            "id": camera_id,
            "name": name,
            "status": "active" if playlist.exists() else "starting",
            "resolution": _probe_resolution(video_path),
            "location": location,
            "stream_url": f"/edge-api/cameras/{camera_id}/stream",
            "snapshot_url": f"/edge-api/cameras/{camera_id}/snapshot",
        }
    ]
    return JSONResponse(content=cameras)


@app.get("/api/cameras/{camera_id}/stream", summary="HLS stream playlist")
def camera_stream(camera_id: str):
    """Serve the HLS m3u8 playlist file."""
    settings = get_settings()
    if camera_id != settings.camera_id:
        raise HTTPException(status_code=404, detail=f"Unknown camera: {camera_id}")

    hls_dir = Path(settings.camera_stream_hls_dir)
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
    if camera_id != settings.camera_id:
        raise HTTPException(status_code=404, detail=f"Unknown camera: {camera_id}")

    hls_dir = Path(settings.camera_stream_hls_dir)
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
    """Capture a frame from the camera video source as JPEG."""
    settings = get_settings()
    if camera_id != settings.camera_id:
        raise HTTPException(status_code=404, detail=f"Unknown camera: {camera_id}")

    video_path = settings.video_input

    if not video_path or not Path(video_path).exists():
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

