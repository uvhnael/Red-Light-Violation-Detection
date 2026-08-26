#!/usr/bin/env python3
"""Điểm vào chính của Edge Node.

Cách dùng thông thường::

    # Chạy pipeline đầy đủ với dispatch qua queue
    python -m edge_node.main --input rtsp://camera:554/stream \
        --stop-line 100,400,800,400 --direction negative_to_positive

    # Export model trước lần chạy đầu
    python -m edge_node.main --export-onnx models/yolov8s.pt
    python -m edge_node.main --export-tensorrt models/yolov8s.pt
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
        "--no-queue",
        action="store_true",
        help="Disable Celery queue dispatch (offline / debug mode)",
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
    tracker = SupervisionByteTracker(
        ByteTrackerConfig(
            track_activation_threshold=settings.tracker_activation_threshold,
            lost_track_buffer=settings.tracker_lost_buffer,
            minimum_matching_threshold=settings.tracker_match_threshold,
            frame_rate=settings.tracker_frame_rate,
        )
    )
    classifier = create_light_classifier(
        roi=light_roi,
        model_path=settings.traffic_light_model_path,
        img_size=settings.traffic_light_img_size,
        device=settings.traffic_light_device or None,
        fusion=FusionConfig(enabled=settings.traffic_light_fusion_enabled),
    )
    stabilizer = RedLightStabilizer(RedStabilizerConfig(
        required_consecutive_frames=settings.red_stable_frames,
        switch_consecutive_frames=settings.red_switch_frames,
        min_confidence=settings.red_min_confidence,
    ))
    violation_detector = ViolationDetector(
        ViolationConfig(tripwire=tripwire_config)
    )

    enable_queue = settings.enable_queue and not args.no_queue

    # --- Durable outbox + background batch sender ---
    # Violation được ghi vào SQLite ngay lúc phát hiện;
    # the sender thread batches them to the central server whenever the
    # network allows (survives outages and restarts).
    outbox = None
    sender = None
    if enable_queue:
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
        enable_queue=False,  # legacy Celery path replaced by the outbox
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
    )

    LOGGER.info(
        "Starting pipeline (queue=%s, ocr=%s, loop=%s)",
        enable_queue, ocr is not None, loop,
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
