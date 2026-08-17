"""Fake violation sender – periodically POSTs violation data + uploads frame capture.

Usage::

    from edge_node.fake_violation_sender import start_fake_violation_sender, stop_fake_violation_sender
    start_fake_violation_sender(settings)
    stop_fake_violation_sender()
"""

from __future__ import annotations

import io
import logging
import random
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

LOGGER = logging.getLogger(__name__)

_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()

# Vietnamese license plate prefixes
_PLATE_PREFIXES = [
    "29", "30", "31", "32", "33", "34", "35", "36", "37", "38",
    "43", "48", "49", "50", "51", "52", "53", "54", "55", "56",
    "57", "58", "59", "60", "61", "62", "63", "65", "66", "67",
    "68", "69", "70", "71", "72", "73", "74", "75", "76", "77",
    "79", "80", "81", "82", "83", "84", "85", "86", "88", "89",
    "90", "92", "93", "94", "95", "97", "98", "99",
]
_PLATE_LETTERS = "ABCDEFGHKLMNPSTUVXYZ"


def _random_plate() -> str:
    prefix = random.choice(_PLATE_PREFIXES)
    letter = random.choice(_PLATE_LETTERS)
    num1 = random.randint(100, 999)
    num2 = random.randint(10, 99)
    return f"{prefix}{letter}-{num1}.{num2}"


def _generate_violation(frame_counter: int) -> dict:
    """Generate a realistic-looking fake violation payload."""
    cx = random.uniform(200, 1700)
    cy = random.uniform(300, 900)
    w = random.uniform(60, 200)
    h = random.uniform(80, 250)

    return {
        "event_id": str(uuid.uuid4()),
        "track_id": random.randint(1, 100),
        "frame_index": frame_counter,
        "timestamp_ms": frame_counter * 33.33,
        "crossing_point": {
            "x": round(cx, 1),
            "y": round(cy + h / 2, 1),
        },
        "previous_point": {
            "x": round(cx + random.uniform(-20, 20), 1),
            "y": round(cy + h / 2 - random.uniform(30, 80), 1),
        },
        "bbox_xyxy": [
            round(cx - w / 2, 1),
            round(cy - h / 2, 1),
            round(cx + w / 2, 1),
            round(cy + h / 2, 1),
        ],
        "light_state": "red",
        "light_confidence": round(random.uniform(0.85, 0.99), 3),
        "previous_side": -1,
        "current_side": 1,
        "plate": {
            "text": _random_plate(),
            "confidence": round(random.uniform(0.70, 0.95), 3),
        },
        "metadata": {
            "source": "fake_camera",
            "camera_id": "fake-cam-1",
        },
    }


def _capture_frame_from_video(video_path: str) -> Optional[bytes]:
    """Capture a single frame from the fake camera video as JPEG bytes."""
    try:
        import cv2
    except ImportError:
        LOGGER.warning("OpenCV not available — cannot capture frame image")
        return None

    video = Path(video_path)
    if not video.exists():
        LOGGER.warning("Video file not found for frame capture: %s", video_path)
        return None

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        LOGGER.warning("Cannot open video for frame capture: %s", video_path)
        return None

    try:
        # Seek to a random position in the video for variety
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            target_frame = random.randint(0, max(0, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        ok, frame = cap.read()
        if not ok:
            # Fallback: read first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()

        if not ok:
            return None

        # Draw bounding box on frame for visual context
        h, w = frame.shape[:2]
        x1 = random.randint(int(w * 0.1), int(w * 0.3))
        y1 = random.randint(int(h * 0.2), int(h * 0.4))
        x2 = random.randint(int(w * 0.4), int(w * 0.6))
        y2 = random.randint(int(h * 0.5), int(h * 0.7))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"RED LIGHT - {_random_plate()}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes()
    except Exception as exc:
        LOGGER.warning("Frame capture failed: %s", exc)
        return None
    finally:
        cap.release()


def _upload_media(central_url: str, event_id: str, image_bytes: bytes, timeout: int) -> bool:
    """Upload the captured frame as violation media to central server."""
    upload_url = central_url.replace("/api/violations", "/api/v1/violations") + f"/{event_id}/media"
    # More robust replacement
    if "/api/violations" in central_url:
        upload_url = central_url.rstrip("/").replace("/api/violations", "") + f"/api/v1/violations/{event_id}/media"
    else:
        # Assume base URL pattern: http://host:port/api/violations → strip suffix
        base = central_url.rstrip("/").rsplit("/", 1)[0]  # remove /violations
        upload_url = f"{base}/v1/violations/{event_id}/media"

    try:
        response = requests.post(
            upload_url,
            files={"file": ("violation.jpg", io.BytesIO(image_bytes), "image/jpeg")},
            timeout=timeout,
        )
        response.raise_for_status()
        LOGGER.info("Media uploaded for event_id=%s: HTTP %d", event_id, response.status_code)
        return True
    except requests.exceptions.RequestException as exc:
        LOGGER.warning("Failed to upload media for event_id=%s: %s", event_id, exc)
        return False


def _sender_loop(settings) -> None:
    """Background loop: send violations + capture and upload frame image."""
    url = settings.central_server_url
    node_id = settings.node_id
    timeout = settings.push_timeout_seconds
    interval_min = settings.fake_violation_interval_min
    interval_max = settings.fake_violation_interval_max
    video_path = settings.fake_camera_video

    frame_counter = 0
    violation_count = 0

    LOGGER.info(
        "Fake violation sender started: url=%s, node_id=%s, interval=%d-%ds, video=%s",
        url, node_id, interval_min, interval_max, video_path,
    )

    while not _stop_event.is_set():
        wait_seconds = random.randint(interval_min, interval_max)
        if _stop_event.wait(wait_seconds):
            break

        frame_counter += random.randint(100, 500)
        payload = _generate_violation(frame_counter)
        event_id = payload["event_id"]

        # 1. POST violation JSON
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=timeout,
                headers={
                    "Content-Type": "application/json",
                    "X-Node-ID": node_id,
                },
            )
            response.raise_for_status()
            violation_count += 1
            LOGGER.info(
                "Fake violation #%d sent: event_id=%s, plate=%s (HTTP %d)",
                violation_count,
                event_id,
                payload["plate"]["text"],
                response.status_code,
            )

            # 2. Upload media (frame capture)
            image_bytes = _capture_frame_from_video(video_path)
            if image_bytes:
                _upload_media(url, event_id, image_bytes, timeout)

        except requests.exceptions.RequestException as exc:
            LOGGER.warning(
                "Failed to send fake violation %s: %s",
                event_id, exc,
            )


def start_fake_violation_sender(settings) -> None:
    """Start the fake violation sender in a background thread."""
    global _thread
    _stop_event.clear()
    _thread = threading.Thread(
        target=_sender_loop,
        args=(settings,),
        daemon=True,
        name="fake-violation-sender",
    )
    _thread.start()


def stop_fake_violation_sender() -> None:
    """Stop the fake violation sender."""
    global _thread
    _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=5)
        _thread = None
    LOGGER.info("Fake violation sender stopped")