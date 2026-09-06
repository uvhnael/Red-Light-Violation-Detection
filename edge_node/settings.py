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


def _env_optional_int(key: str) -> int | None:
    """Parse env var dạng int: unset/rỗng -> None, else int."""
    raw = os.environ.get(key, "").strip()
    if raw == "":
        return None
    return int(raw)


def _env_optional_float(key: str) -> float | None:
    """Parse env var dạng float: unset/rỗng -> None, else float."""
    raw = os.environ.get(key, "").strip()
    if raw == "":
        return None
    return float(raw)


def _resolve_optional_bool(key: str) -> Optional[bool]:
    """Parse a tri-state env var: unset/empty -> None (auto), else True/False."""
    raw = os.environ.get(key, "").strip().lower()
    if raw in ("", "auto"):
        return None
    return raw in ("1", "true", "yes")


@dataclass(frozen=True)
class EdgeNodeSettings:
    """Immutable snapshot of the current configuration."""

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
        default_factory=lambda: _env("OUTBOX_DB_PATH", "data/outbox/violations.db")
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
    # Realtime drop-frame cho stream RTSP/HTTP: bỏ frame cũ để bám thời gian
    # thực khi pipeline xử lý chậm hơn tốc độ camera. Mặc định TẮT (đọc tuần
    # tự) để giữ nguyên hành vi cho file video.
    video_realtime: bool = field(default_factory=lambda: _env_bool("VIDEO_REALTIME", False))
    # Ngưỡng trễ tối đa (ms) cho phép trước khi bắt đầu bỏ frame ở chế độ realtime.
    video_max_lag_ms: int = field(default_factory=lambda: _env_int("VIDEO_MAX_LAG_MS", 1000))

    # ---- YOLO / Detection ----
    yolo_model_path: str = field(default_factory=lambda: _env("YOLO_MODEL_PATH", "models/yolo26m_vehicle.pt"))
    # 0.30 (hạ từ 0.35): xe máy nhỏ/bị che thường conf 0.30-0.35 — ngưỡng cũ
    # làm lọt detection xuống tracker quá thưa, track xe máy đứt liên tục.
    # Tracker tự lọc noise bằng activation threshold riêng.
    yolo_confidence: float = field(default_factory=lambda: _env_float("YOLO_CONFIDENCE", 0.30))
    yolo_img_size: int = field(default_factory=lambda: _env_int("YOLO_IMG_SIZE", 640))
    # Empty = auto-detect (CUDA if available, else CPU). Set "cpu" to force CPU.
    yolo_device: str = field(default_factory=lambda: _env("YOLO_DEVICE", ""))
    # Empty = auto (FP16 only when running on CUDA). Set "0"/"1" to force on/off.
    yolo_fp16: Optional[bool] = field(
        default_factory=lambda: _resolve_optional_bool("YOLO_FP16")
    )

    # ---- Traffic-light colour classifier (YOLO26n-cls) ----
    # Empty = use the default models/traffic_light_cls.pt; when the
    # weights are missing the pipeline falls back to the OpenCV HSV classifier.
    traffic_light_model_path: str = field(
        default_factory=lambda: _env("TRAFFIC_LIGHT_MODEL_PATH", "models/traffic_light_cls.pt")
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
    # Mốc thời gian tính theo GIÂY để hành vi đồng nhất ở mọi FPS nguồn
    # (video thực tế 3-10 fps: tham số frame cứng khiến camera 3 fps trễ
    # ~2.3s khi chuyển đèn). Các giá trị frame legacy vẫn override nếu set
    # (dành cho test + fine-tune theo từng camera nếu cần).
    # Lock trạng thái đầu từ UNKNOWN (giây).
    red_lock_seconds: float = field(default_factory=lambda: _env_float("RED_LOCK_SECONDS", 0.4))
    # Hysteresis chuyển giữa 2 trạng thái đã stable (giây).
    red_switch_seconds: float = field(default_factory=lambda: _env_float("RED_SWITCH_SECONDS", 0.8))
    # Dung sai đọc UNKNOWN liên tục trước khi reset signal về UNKNOWN (giây).
    red_unknown_tolerance_seconds: float = field(
        default_factory=lambda: _env_float("RED_UNKNOWN_TOLERANCE_SECONDS", 0.8)
    )
    # Legacy override theo FRAME (ưu tiên cao hơn giá trị giây nếu set).
    # RED_STABLE_FRAMES / RED_SWITCH_FRAMES cũ vẫn hoạt động để không phá
    # cấu hình đã deploy; set RED_USE_SECONDS=false để ép dùng frame.
    red_stable_frames: int | None = field(default_factory=lambda: _env_optional_int("RED_STABLE_FRAMES"))
    red_switch_frames: int | None = field(default_factory=lambda: _env_optional_int("RED_SWITCH_FRAMES"))
    red_use_seconds: bool = field(default_factory=lambda: _env_bool("RED_USE_SECONDS", True))
    # Classifier confidence below which a reading is treated as unknown.
    # Small/distant lights often report 0.5-0.6 confidence on the true colour,
    # so this must stay low enough to let real transitions through.
    red_min_confidence: float = field(default_factory=lambda: _env_float("RED_MIN_CONFIDENCE", 0.55))

    # ---- Tracker ----
    # FPS nguồn: None = tự probe từ metadata video (khuyến nghị — các mốc
    # thời gian của tracker hoá theo giây đúng ở mọi camera). Set số cụ thể
    # khi metadata sai hoặc muốn ép.
    tracker_frame_rate: float | None = field(
        default_factory=lambda: _env_optional_float("TRACKER_FRAME_RATE")
    )

    # ---- Pipeline ----
    # Đường gửi vi phạm duy nhất hiện nay là durable outbox + batch sender.
    # Bật tắt bằng OUTBOX_ENABLED. (Đường Celery/Redis cũ đã bị loại bỏ hoàn
    # toàn — không còn field cấu hình nào liên quan.)
    outbox_enabled: bool = field(default_factory=lambda: _env_bool("OUTBOX_ENABLED", True))
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
    # Shared secret bảo vệ các endpoint quản trị /action/* (restart, đổi vạch
    # dừng, đổi ROI). Client gửi qua header ``X-Edge-Token`` hoặc
    # ``Authorization: Bearer <token>``. Rỗng = không bắt buộc (chế độ dev,
    # server sẽ log cảnh báo một lần). Đặt giá trị trong production.
    api_token: str = field(default_factory=lambda: _env("EDGE_API_TOKEN", ""))

    # --- Bảo mật control-plane ---------------------------------------- #
    # Danh sách origin được phép gọi API (CORS) — phân tách bằng dấu phẩy.
    # Rỗng = "*" (mọi origin, chỉ dành cho dev).
    allowed_origins: str = field(
        default_factory=lambda: _env("EDGE_ALLOWED_ORIGINS", "")
    )
    # Rate limit (cửa sổ trượt) cho endpoint ghi /action/* — chống brute
    # force token và spam. Đơn vị: số request/phút/IP.
    rate_limit_per_minute: int = field(
        default_factory=lambda: _env_int("EDGE_RATE_LIMIT_PER_MINUTE", 30)
    )
    # Bật buộc token trong production: khi true, token rỗng → từ chối mọi
    # endpoint ghi thay vì cảnh báo rồi cho qua.
    require_token: bool = field(
        default_factory=lambda: _env_bool("EDGE_REQUIRE_TOKEN", False)
    )
    # Token xác thực node biên khi đẩy dữ liệu lên Central (header
    # X-Ingest-Token). Phải trùng INGEST_TOKEN đặt ở Central Server.
    ingest_token: str = field(
        default_factory=lambda: _env("INGEST_TOKEN", "")
    )

    # ---- Node identity ----
    node_id: str = field(default_factory=lambda: _env("NODE_ID", "edge-node-01"))
    node_name: str = field(default_factory=lambda: _env("NODE_NAME", ""))
    node_ip_address: str = field(default_factory=lambda: _env("NODE_IP_ADDRESS", ""))
    node_status: str = field(default_factory=lambda: _env("NODE_STATUS", "online"))
    node_heartbeat_interval_seconds: int = field(default_factory=lambda: _env_int("NODE_HEARTBEAT_INTERVAL", 60))

    # ---- Camera stream (serve the video input as a live HLS camera) ----
    camera_stream_enabled: bool = field(default_factory=lambda: _env_bool("CAMERA_STREAM_ENABLED", True))
    camera_stream_hls_dir: str = field(default_factory=lambda: _env("CAMERA_STREAM_HLS_DIR", "data/hls/cam-1"))
    camera_id: str = field(
        default_factory=lambda: _env("CAMERA_ID", "") or f"{_env('NODE_ID', 'edge-node-01')}-cam-1"
    )
    camera_name: str = field(default_factory=lambda: _env("CAMERA_NAME", ""))
    camera_location: str = field(default_factory=lambda: _env("CAMERA_LOCATION", ""))


def get_settings() -> EdgeNodeSettings:
    """Build settings from the current environment."""
    return EdgeNodeSettings()
