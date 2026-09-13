#!/usr/bin/env python3
"""Điểm vào chính của Edge Node.

Vi phạm được gửi qua durable SQLite outbox + batch sender (không còn
Redis/Celery). Vạch dừng + vùng đèn do operator kẻ trên web UI sau khi
camera đăng ký với central server; chưa kẻ vạch thì vi phạm bị TẮT.

Cách dùng thông thường::

    # Chạy pipeline đầy đủ (outbox + sender tự khởi động)
    python -m edge_node.main --input rtsp://camera:554/stream

    # Chạy với vạch dừng truyền tay (bỏ qua web UI)
    python -m edge_node.main --input video.mp4 \
        --stop-line 100,400,800,400 --direction negative_to_positive

    # Export model trước lần chạy đầu
    python -m edge_node.main --export-onnx models/yolo26m_vehicle.pt
    python -m edge_node.main --export-tensorrt models/yolo26m_vehicle.pt
"""

from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path

from edge_node.settings import get_settings

LOGGER = logging.getLogger("edge_node")


def _parse_stop_line(value: str) -> tuple:
    from edge_node.core.contracts import Point

    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Expected stop line as x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = (float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Stop-line coordinates must be numeric"
        ) from exc
    return Point(x1, y1), Point(x2, y2)


def _parse_light_roi(value: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Expected light ROI as x,y,width,height"
        )
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Light ROI coordinates must be integers"
        ) from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(
            "Light ROI width and height must be positive"
        )
    return x, y, w, h


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Edge Node – Red-light violation detection pipeline",
    )
    p.add_argument("--input", type=str, help="Video path or RTSP URL")
    p.add_argument(
        "--stop-line",
        type=_parse_stop_line,
        help="Calibrated tripwire as x1,y1,x2,y2",
    )
    p.add_argument(
        "--direction",
        choices=["any", "positive_to_negative", "negative_to_positive"],
        default="any",
        help="Crossing direction relative to the stop line",
    )
    p.add_argument(
        "--light-roi",
        type=_parse_light_roi,
        help="Traffic-light ROI as x,y,width,height",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame processing limit",
    )
    p.add_argument(
        "--no-outbox",
        action="store_true",
        help="Disable the durable outbox delivery (offline / stats-only mode)",
    )
    p.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable license-plate OCR (overrides ENABLE_OCR=true)",
    )
    p.add_argument(
        "--loop",
        action="store_true",
        help="Loop the video file (useful when testing without camera stream)",
    )
    p.add_argument(
        "--start-api",
        action="store_true",
        help="Start the FastAPI control-plane server in a background thread",
    )
    p.add_argument(
        "--export-onnx",
        type=Path,
        metavar="MODEL",
        help="Export a .pt model to ONNX and exit",
    )
    p.add_argument(
        "--export-tensorrt",
        type=Path,
        metavar="MODEL",
        help="Export a .pt model to TensorRT engine and exit",
    )
    p.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return p


def _start_api_background() -> None:
    """Launch the FastAPI control-plane in a daemon thread."""
    import uvicorn
    from edge_node.api.server import app as api_app

    settings = get_settings()
    LOGGER.info(
        "Starting control-plane API on %s:%s",
        settings.api_host,
        settings.api_port,
    )
    thread = threading.Thread(
        target=uvicorn.run,
        kwargs={
            "app": api_app,
            "host": settings.api_host,
            "port": settings.api_port,
            "log_level": "warning",
        },
        daemon=True,
    )
    thread.start()


def run_pipeline(args) -> int:
    """Build and execute the vision pipeline."""
    from edge_node.core.contracts import CrossingDirection, Point
    from edge_node.core.config import (
        RedStabilizerConfig,
        TripwireConfig,
        ViolationConfig,
        scaled_stabilizer_config,
    )
    from edge_node.core.byte_tracker import ByteTrackerConfig, SupervisionByteTracker
    from edge_node.core.detector import YoloDetector
    from edge_node.core.traffic_light_yolo import FusionConfig, create_light_classifier
    from edge_node.core.violation_logic import RedLightStabilizer, ViolationDetector
    from edge_node.core.pipeline import RedLightViolationPipeline
    from edge_node.core.video_io import OpenCVFrameSource

    settings = get_settings()

    if args.input is None:
        args.input = settings.video_input
    if not args.input:
        LOGGER.error("--input or VIDEO_INPUT env var is required")
        return 1

    # --- FPS nguồn: tracker (lost buffer theo giây) + stabilizer (quy đổi
    # giây -> frame) đều cần FPS thật để mốc thời gian đúng ở mọi camera.
    # Video thực tế 3-10 fps; hard-code 30 khiến camera 3fps trễ ~2.3s
    # khi chuyển đèn và giữ zombie track ~10s.
    from edge_node.core.video_io import probe_fps

    source_fps = settings.tracker_frame_rate or probe_fps(args.input)
    LOGGER.info("Video source FPS (đ dùng cho tracker + stabilizer): %.2f", source_fps)

    # --- Camera stream: expose the video input as a live HLS camera feed
    # so the web dashboard can watch it directly from this edge node. ---
    if settings.camera_stream_enabled and Path(args.input).exists():
        from edge_node.camera_stream import start_camera_stream

        try:
            start_camera_stream(args.input, settings)
        except Exception as exc:
            LOGGER.warning("Camera stream failed to start: %s", exc)

    # --- Stop line + traffic light ROI: no auto-detection. The operator
    # draws them on the web UI (CalibrationEditor) after the camera
    # registers with the central server. Until a stop line exists the
    # violation detector stays disabled (no red-light events). Explicit CLI
    # flags still win when provided. ---
    light_roi = args.light_roi
    if args.stop_line is not None:
        start, end = args.stop_line
        LOGGER.info(
            "Using explicit stop line: (%s,%s)->(%s,%s)",
            start.x, start.y, end.x, end.y,
        )
        tripwire_config = TripwireConfig(
            start=start,
            end=end,
            direction=CrossingDirection(args.direction),
        )
    else:
        tripwire_config = None
        LOGGER.info(
            "No stop line configured yet — violations DISABLED until the "
            "operator draws one via the web UI"
        )
    if light_roi is not None:
        LOGGER.info("Using explicit light ROI: x=%d y=%d w=%d h=%d", *light_roi)
    else:
        from edge_node.core.config import get_active_light_roi

        # Giữ lại ROI đã kẻ qua web UI từ phiên trước (sống sót qua restart).
        light_roi = get_active_light_roi()

    from edge_node.core.config import set_active_tripwire

    # None = not calibrated yet; the violation gate treats it as disabled.
    set_active_tripwire(tripwire_config)

    # --- Components ---
    detector = YoloDetector(
        model_path=settings.yolo_model_path,
        confidence=settings.yolo_confidence,
        img_size=settings.yolo_img_size,
        device=settings.yolo_device or None,
    )
    # Tracker: mặc định class mới đã tối ưu xe máy (activation 0.10,
    # match 0.85, lost buffer ~1s theo FPS thật) + tầng track quality:
    # quỹ đạo/vận tốc, EMA confidence, vote nhãn, log ID switch.
    # frame_rate = FPS nguồn thật để các mốc thời gian đúng theo giây.
    tracker = SupervisionByteTracker(
        ByteTrackerConfig(
            frame_rate=source_fps,
            trajectory_max_samples=settings.tracker_trajectory_samples,
            confidence_ema_alpha=settings.tracker_confidence_ema_alpha,
            label_vote_window=settings.tracker_label_vote_window,
            gate_enabled=settings.tracker_gate_enabled,
            gate_distance_factor=settings.tracker_gate_distance_factor,
            gate_direction_cosine=settings.tracker_gate_direction_cosine,
            debug_associations=settings.tracker_debug_associations,
            id_switch_log_interval=settings.tracker_id_switch_log_interval,
        )
    )
    if settings.tracker_gate_enabled:
        LOGGER.warning(
            "Tracker gating ENABLED (distance factor %.2f, direction cosine "
            "%.2f) — association behaviour differs from the measured baseline",
            settings.tracker_gate_distance_factor,
            settings.tracker_gate_direction_cosine,
        )
    classifier = create_light_classifier(
        roi=light_roi,
        model_path=settings.traffic_light_model_path,
        img_size=settings.traffic_light_img_size,
        device=settings.traffic_light_device or None,
        fusion=FusionConfig(enabled=settings.traffic_light_fusion_enabled),
    )
    # Stabilizer: mốc thời gian theo GIÂY (quy đổi frame theo FPS nguồn)
    # — chuyển đèn mượt ở camera 3fps như ở 30fps. Override frame legacy
    # (RED_STABLE_FRAMES / RED_SWITCH_FRAMES) vẫn thắng khi được set.
    if (
        settings.red_use_seconds
        and settings.red_stable_frames is None
        and settings.red_switch_frames is None
    ):
        stabilizer = RedLightStabilizer(scaled_stabilizer_config(
            seconds=(
                settings.red_lock_seconds,
                settings.red_switch_seconds,
                settings.red_unknown_tolerance_seconds,
            ),
            fps=source_fps,
            min_confidence=settings.red_min_confidence,
        ))
    else:
        # Chế độ legacy: dùng tham số frame trực tiếp (giữ hành vi cũ cho
        # cấu hình đã deploy + test đơn vị).
        stabilizer = RedLightStabilizer(RedStabilizerConfig(
            required_consecutive_frames=settings.red_stable_frames or 3,
            switch_consecutive_frames=settings.red_switch_frames or 7,
            min_confidence=settings.red_min_confidence,
        ))
    violation_detector = ViolationDetector(
        ViolationConfig(tripwire=tripwire_config)
    )

    enable_outbox = settings.outbox_enabled and not args.no_outbox

    # --- Durable outbox + background batch sender ---
    # Violation được ghi vào SQLite ngay lúc phát hiện;
    # the sender thread batches them to the central server whenever the
    # network allows (survives outages and restarts).
    outbox = None
    sender = None
    if enable_outbox:
        from edge_node.outbox import ViolationOutbox
        from edge_node.violation_sender import ViolationSender

        outbox = ViolationOutbox(settings.outbox_db_path)
        sender = ViolationSender(outbox, settings)
        sender.start()
        pending = outbox.pending_count()
        if pending:
            LOGGER.info("Outbox has %d pending violation(s) from previous runs", pending)

    # --- License-plate OCR (fast-plate-ocr) ---
    ocr = None
    if settings.enable_ocr and not args.no_ocr:
        from edge_node.core.ocr_recognizer import FastPlateOCR

        ocr = FastPlateOCR(
            model_name=settings.ocr_model_name,
            device=settings.ocr_device,
            pad_to=settings.ocr_pad_to,
        )
        LOGGER.info(
            "OCR enabled (model=%s, device=%s)",
            settings.ocr_model_name, settings.ocr_device,
        )

    pipeline = RedLightViolationPipeline(
        detector=detector,
        tracker=tracker,
        light_classifier=classifier,
        stabilizer=stabilizer,
        violation_detector=violation_detector,
        ocr=ocr,
        logger=LOGGER,
        outbox=outbox,
        evidence_max_width=settings.evidence_image_max_width,
        evidence_quality=settings.evidence_image_quality,
    )

    loop = args.loop or settings.video_loop
    source = OpenCVFrameSource(
        args.input,
        max_frames=args.max_frames,
        loop=loop,
        realtime=settings.video_realtime,
        max_lag_ms=float(settings.video_max_lag_ms),
    )

    LOGGER.info(
        "Starting pipeline (outbox=%s, ocr=%s, loop=%s)",
        enable_outbox, ocr is not None, loop,
    )
    result = pipeline.process(source, max_frames=args.max_frames)
    LOGGER.info(
        "Done: %s frames, %s violation(s)",
        result.frames_processed,
        len(result.violations),
    )

    # Cho sender cơ hội rút hết outbox trước khi thoát
    if sender is not None and outbox is not None:
        import time

        deadline = time.monotonic() + settings.outbox_flush_interval + 5
        while outbox.pending_count() > 0 and time.monotonic() < deadline:
            time.sleep(0.5)
        remaining = outbox.pending_count()
        if remaining:
            LOGGER.warning(
                "%d violation(s) still pending in outbox — will be sent on next run",
                remaining,
            )
        else:
            LOGGER.info("Outbox drained — all violations delivered")
        sender.stop()
        outbox.close()
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # Xử lý các lệnh export model
    if args.export_onnx:
        from edge_node.core.detector import YoloDetector

        YoloDetector.export_onnx(args.export_onnx)
        return 0

    if args.export_tensorrt:
        from edge_node.core.detector import YoloDetector

        YoloDetector.export_tensorrt(args.export_tensorrt)
        return 0

    settings = get_settings()

    from edge_node.central_client import register_node, start_registration_heartbeat

    register_node(settings)
    start_registration_heartbeat(settings)

    # Tuỳ chọn khởi động control-plane API
    if args.start_api:
        _start_api_background()

    try:
        return run_pipeline(args)
    except Exception:
        LOGGER.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
