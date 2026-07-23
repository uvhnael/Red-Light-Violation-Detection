"""Centralised configuration for the Edge Node.

All tunables are read from environment variables with sensible defaults
for local development.  In production the corresponding env-vars are
set via ``docker-compose.yml`` or systemd unit files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class EdgeNodeSettings:
    """Immutable snapshot of the current configuration."""

    # ---- Redis / Celery ----
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0"))
    celery_result_backend: str = field(default_factory=lambda: _env("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"))

    # ---- Central server ----
    central_server_url: str = field(
        default_factory=lambda: _env("CENTRAL_SERVER_URL", "http://central-server:8000/api/violations")
    )
    push_timeout_seconds: int = field(default_factory=lambda: _env_int("PUSH_TIMEOUT", 10))
    push_max_retries: int = field(default_factory=lambda: _env_int("PUSH_MAX_RETRIES", 5))
    push_retry_delay: int = field(default_factory=lambda: _env_int("PUSH_RETRY_DELAY", 30))

    # ---- Video input ----
    video_input: str = field(default_factory=lambda: _env("VIDEO_INPUT", ""))
    video_loop: bool = field(default_factory=lambda: _env_bool("VIDEO_LOOP", False))

    # ---- YOLO / Detection ----
    yolo_model_path: str = field(default_factory=lambda: _env("YOLO_MODEL_PATH", "edge_node/models/yolov8s.pt"))
    yolo_confidence: float = field(default_factory=lambda: _env_float("YOLO_CONFIDENCE", 0.35))
    yolo_img_size: int = field(default_factory=lambda: _env_int("YOLO_IMG_SIZE", 640))
    yolo_device: str = field(default_factory=lambda: _env("YOLO_DEVICE", ""))

    # ---- Tracker ----
    tracker_activation_threshold: float = field(default_factory=lambda: _env_float("TRACKER_ACTIVATION_THRESHOLD", 0.25))
    tracker_lost_buffer: int = field(default_factory=lambda: _env_int("TRACKER_LOST_BUFFER", 30))
    tracker_match_threshold: float = field(default_factory=lambda: _env_float("TRACKER_MATCH_THRESHOLD", 0.80))
    tracker_frame_rate: int = field(default_factory=lambda: _env_int("TRACKER_FRAME_RATE", 30))

    # ---- Pipeline ----
    enable_queue: bool = field(default_factory=lambda: _env_bool("ENABLE_QUEUE", True))
    enable_ocr: bool = field(default_factory=lambda: _env_bool("ENABLE_OCR", False))
    max_frames: int = field(default_factory=lambda: _env_int("MAX_FRAMES", 0))  # 0 = unlimited

    # ---- API ----
    api_host: str = field(default_factory=lambda: _env("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: _env_int("API_PORT", 8080))

    # ---- Node identity ----
    node_id: str = field(default_factory=lambda: _env("NODE_ID", "edge-node-01"))


def get_settings() -> EdgeNodeSettings:
    """Build settings from the current environment."""
    return EdgeNodeSettings()
