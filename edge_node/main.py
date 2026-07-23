#!/usr/bin/env python3
"""Edge Node entry-point.

Typical usage::

    # Run the full pipeline with queue dispatch
    python -m edge_node.main --input rtsp://camera:554/stream \
        --stop-line 100,400,800,400 --direction negative_to_positive

    # Export model before first run
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
    from edge_node.core.traffic_light_cv import OpenCVTrafficLightClassifier
    from edge_node.core.violation_logic import RedLightStabilizer, ViolationDetector
    from edge_node.core.pipeline import RedLightViolationPipeline
    from edge_node.core.video_io import OpenCVFrameSource

    settings = get_settings()

    if args.input is None:
        args.input = settings.video_input
    if not args.input:
        LOGGER.error("--input or VIDEO_INPUT env var is required")
        return 1
    if args.stop_line is None:
        LOGGER.error("--stop-line is required")
        return 1

    start, end = args.stop_line
    tripwire_config = TripwireConfig(
        start=start,
        end=end,
        direction=CrossingDirection(args.direction),
    )
    from edge_node.core.config import set_active_tripwire
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
    classifier = OpenCVTrafficLightClassifier(roi=args.light_roi)
    stabilizer = RedLightStabilizer(RedStabilizerConfig())
    violation_detector = ViolationDetector(
        ViolationConfig(tripwire=tripwire_config)
    )

    enable_queue = settings.enable_queue and not args.no_queue

    pipeline = RedLightViolationPipeline(
        detector=detector,
        tracker=tracker,
        light_classifier=classifier,
        stabilizer=stabilizer,
        violation_detector=violation_detector,
        enable_queue=enable_queue,
        logger=LOGGER,
    )

    loop = args.loop or settings.video_loop
    source = OpenCVFrameSource(
        args.input,
        max_frames=args.max_frames,
        loop=loop,
    )

    LOGGER.info("Starting pipeline (queue=%s, loop=%s)", enable_queue, loop)
    result = pipeline.process(source, max_frames=args.max_frames)
    LOGGER.info(
        "Done: %s frames, %s violation(s)",
        result.frames_processed,
        len(result.violations),
    )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # Handle export commands
    if args.export_onnx:
        from edge_node.core.detector import YoloDetector

        YoloDetector.export_onnx(args.export_onnx)
        return 0

    if args.export_tensorrt:
        from edge_node.core.detector import YoloDetector

        YoloDetector.export_tensorrt(args.export_tensorrt)
        return 0

    # Optionally start control-plane API
    if args.start_api:
        _start_api_background()

    try:
        return run_pipeline(args)
    except Exception:
        LOGGER.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
