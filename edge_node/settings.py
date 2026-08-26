"""Cấu hình tập trung của Edge Node.

Mọi tham số đều đọc từ biến môi trường với giá trị mặc định hợp lý cho
dev local. Trong production, env-var tương ứng được set qua
``docker-compose.yml`` hoặc file unit systemd.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).lower() in ("1", "true", "yes")


def _resolve_optional_bool(key: str) -> Optional[bool]:
    """Parse a tri-state env var: unset/empty -> None (auto), else True/False."""
    raw = os.environ.get(key, "").strip().lower()
    if raw in ("", "auto"):
        return None
    return raw in ("1", "true", "yes")


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
    node_register_url: str = field(
        default_factory=lambda: _env("NODE_REGISTER_URL", "http://central-server:8000/api/v1/edge-nodes/register")
    )
    push_timeout_seconds: int = field(default_factory=lambda: _env_int("PUSH_TIMEOUT", 10))
    push_max_retries: int = field(default_factory=lambda: _env_int("PUSH_MAX_RETRIES", 5))
    push_retry_delay: int = field(default_factory=lambda: _env_int("PUSH_RETRY_DELAY", 30))

    # ---- Violation outbox (durable SQLite queue + batch sender) ----
    outbox_db_path: str = field(
        default_factory=lambda: _env("OUTBOX_DB_PATH", "edge_node/data/outbox/violations.db")
    )
    outbox_batch_size: int = field(default_factory=lambda: _env_int("OUTBOX_BATCH_SIZE", 20))
    outbox_flush_interval: int = field(default_factory=lambda: _env_int("OUTBOX_FLUSH_INTERVAL", 5))
    outbox_media_upload: bool = field(default_factory=lambda: _env_bool("OUTBOX_MEDIA_UPLOAD", True))
    # Max width of the evidence JPEG (0 = keep original resolution)
    evidence_image_max_width: int = field(default_factory=lambda: _env_int("EVIDENCE_IMAGE_MAX_WIDTH", 1280))
    evidence_image_quality: int = field(default_factory=lambda: _env_int("EVIDENCE_IMAGE_QUALITY", 80))

    # ---- Video input ----
    video_input: str = field(default_factory=lambda: _env("VIDEO_INPUT", ""))
    video_loop: bool = field(default_factory=lambda: _env_bool("VIDEO_LOOP", False))

    # ---- YOLO / Detection ----
    yolo_model_path: str = field(default_factory=lambda: _env("YOLO_MODEL_PATH", "edge_node/models/yolo26m_vehicle.pt"))
    yolo_confidence: float = field(default_factory=lambda: _env_float("YOLO_CONFIDENCE", 0.35))
    yolo_img_size: int = field(default_factory=lambda: _env_int("YOLO_IMG_SIZE", 640))
    # Empty = auto-detect (CUDA if available, else CPU). Set "cpu" to force CPU.
    yolo_device: str = field(default_factory=lambda: _env("YOLO_DEVICE", ""))
    # Empty = auto (FP16 only when running on CUDA). Set "0"/"1" to force on/off.
    yolo_fp16: Optional[bool] = field(
        default_factory=lambda: _resolve_optional_bool("YOLO_FP16")
    )

    # ---- Traffic-light colour classifier (YOLO26n-cls) ----
    # Empty = use the default edge_node/models/traffic_light_cls.pt; when the
    # weights are missing the pipeline falls back to the OpenCV HSV classifier.
    traffic_light_model_path: str = field(
        default_factory=lambda: _env("TRAFFIC_LIGHT_MODEL_PATH", "edge_node/models/traffic_light_cls.pt")
    )
    traffic_light_img_size: int = field(default_factory=lambda: _env_int("TRAFFIC_LIGHT_IMG_SIZE", 64))
    # Empty = auto-detect (CUDA if available, else CPU). Set "cpu" to force CPU.
    traffic_light_device: str = field(default_factory=lambda: _env("TRAFFIC_LIGHT_DEVICE", ""))
    # Cross-check the YOLO colour vote against HSV colour evidence + lamp
    # vertical position (domain-independent physical cues).  Recommended ON:
    # the LISA-trained model suffers domain shift on Vietnamese lights.
    traffic_light_fusion_enabled: bool = field(
        default_factory=lambda: _env_bool("TRAFFIC_LIGHT_FUSION", True)
    )

    # ---- Traffic-light state stabilizer (anti-flicker debounce) ----
    # Frames of the same confident colour needed to lock in the first state.
    red_stable_frames: int = field(default_factory=lambda: _env_int("RED_STABLE_FRAMES", 3))
    # Frames needed to SWITCH between two stable states (hysteresis). Raise
    # this if the signal still flickers between red/green.
    red_switch_frames: int = field(default_factory=lambda: _env_int("RED_SWITCH_FRAMES", 7))
    # Classifier confidence below which a reading is treated as unknown.
    # Small/distant lights often report 0.5-0.6 confidence on the true colour,
    # so this must stay low enough to let real transitions through.
    red_min_confidence: float = field(default_factory=lambda: _env_float("RED_MIN_CONFIDENCE", 0.55))

    # ---- Tracker ----
    tracker_activation_threshold: float = field(default_factory=lambda: _env_float("TRACKER_ACTIVATION_THRESHOLD", 0.25))
    tracker_lost_buffer: int = field(default_factory=lambda: _env_int("TRACKER_LOST_BUFFER", 30))
    tracker_match_threshold: float = field(default_factory=lambda: _env_float("TRACKER_MATCH_THRESHOLD", 0.80))
    tracker_frame_rate: int = field(default_factory=lambda: _env_int("TRACKER_FRAME_RATE", 30))

    # ---- Pipeline ----
    enable_queue: bool = field(default_factory=lambda: _env_bool("ENABLE_QUEUE", True))
    enable_ocr: bool = field(default_factory=lambda: _env_bool("ENABLE_OCR", True))
    max_frames: int = field(default_factory=lambda: _env_int("MAX_FRAMES", 0))  # 0 = unlimited

    # ---- License-plate OCR (fast-plate-ocr) ----
    # Empty device = auto (CUDA if available, else CPU). Set "cuda" or "cpu" to force.
    ocr_model_name: str = field(
        default_factory=lambda: _env("OCR_MODEL_NAME", "global-plates-mobile-vit-v2-model")
    )
    ocr_device: str = field(default_factory=lambda: _env("OCR_DEVICE", "auto"))
    ocr_pad_to: int = field(default_factory=lambda: _env_int("OCR_PAD_TO", 8))

    # ---- API ----
    api_host: str = field(default_factory=lambda: _env("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: _env_int("API_PORT", 8080))

    # ---- Node identity ----
    node_id: str = field(default_factory=lambda: _env("NODE_ID", "edge-node-01"))
    node_name: str = field(default_factory=lambda: _env("NODE_NAME", ""))
    node_ip_address: str = field(default_factory=lambda: _env("NODE_IP_ADDRESS", ""))
    node_status: str = field(default_factory=lambda: _env("NODE_STATUS", "online"))
    node_heartbeat_interval_seconds: int = field(default_factory=lambda: _env_int("NODE_HEARTBEAT_INTERVAL", 60))

    # ---- Camera stream (serve the video input as a live HLS camera) ----
    camera_stream_enabled: bool = field(default_factory=lambda: _env_bool("CAMERA_STREAM_ENABLED", True))
    camera_stream_hls_dir: str = field(default_factory=lambda: _env("CAMERA_STREAM_HLS_DIR", "edge_node/data/hls/cam-1"))
    camera_id: str = field(
        default_factory=lambda: _env("CAMERA_ID", "") or f"{_env('NODE_ID', 'edge-node-01')}-cam-1"
    )
    camera_name: str = field(default_factory=lambda: _env("CAMERA_NAME", ""))
    camera_location: str = field(default_factory=lambda: _env("CAMERA_LOCATION", ""))


def get_settings() -> EdgeNodeSettings:
    """Build settings from the current environment."""
    return EdgeNodeSettings()
