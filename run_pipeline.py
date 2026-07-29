#!/usr/bin/env python3
"""Standalone runner for the edge node detection pipeline — no Docker required.

Usage
-----
    # Quick start with default video and stop-line — LIVE VIEW
    python run_pipeline.py

    # Custom video + stop-line
    python run_pipeline.py edge_node/data/videos/tr.mp4 --stop-line 100,400,800,400

    # Headless mode (no window) — for servers or batch runs
    python run_pipeline.py --no-window

    # Record output video
    python run_pipeline.py --record output.mp4

    # Loop continuously + live view
    python run_pipeline.py --loop

    # Max frames limit, disable Celery queue
    python run_pipeline.py tr.mp4 --max-frames 300 --no-queue

    # Different model
    python run_pipeline.py --model edge_node/models/yolov8x.pt

Interactive controls (live window):
    q / ESC  — quit
    Space    — pause / resume

Dependencies
------------
Install with pip (preferably in a venv):

    python3 -m venv .venv
    source .venv/bin/activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    pip install ultralytics opencv-python supervision numpy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _setup_logging(level: str = "INFO") -> None:
    import logging
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s  [%(name)-20s] %(message)s",
        datefmt="%H:%M:%S",
    )
    for lib in ("ultralytics", "matplotlib", "PIL"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def _parse_stop_line(value: str):
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Expected x1,y1,x2,y2 (e.g. 100,400,800,400)")
    try:
        x1, y1, x2, y2 = (float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("Stop-line coordinates must be numeric")
    from edge_node.core.contracts import Point
    return Point(x1, y1), Point(x2, y2)


def _parse_light_roi(value: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Expected x,y,w,h (e.g. 100,50,60,100)")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("Light ROI coordinates must be integers")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("Light ROI width/height must be positive")
    return x, y, w, h


def check_deps() -> list[str]:
    missing = []
    for mod, pkg in [
        ("ultralytics", "ultralytics"),
        ("cv2", "opencv-python"),
        ("supervision", "supervision"),
        ("torch", "torch"),
        ("numpy", "numpy"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    return missing


def find_default_video() -> Optional[Path]:
    candidates = [
        PROJECT_ROOT / "edge_node/data/videos/aziz1.MP4",
        PROJECT_ROOT / "edge_node/data/videos/tr.mp4",
        PROJECT_ROOT / "edge_node/data/videos/16h30.25.9.22.mp4",
        PROJECT_ROOT / "edge_node/data/videos/16h15.25.9.22.mp4",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Red-Light Violation Detection Pipeline — live view runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Controls: q/ESC=quit  Space=pause/resume",
    )

    p.add_argument(
        "video", nargs="?", default=None,
        help="Input video path (auto-detected if omitted)",
    )
    p.add_argument("--max-frames", "-n", type=int, default=None)
    p.add_argument("--loop", "-l", action="store_true")
    p.add_argument(
        "--stop-line", "-s", type=_parse_stop_line, metavar="x1,y1,x2,y2",
        default=None,
    )
    p.add_argument(
        "--direction", "-d",
        choices=["any", "positive_to_negative", "negative_to_positive"],
        default="negative_to_positive",
    )
    p.add_argument("--light-roi", "-r", type=_parse_light_roi, metavar="x,y,w,h", default=None)
    p.add_argument(
        "--model", "-m", type=str,
        default=str(PROJECT_ROOT / "edge_node/models/yolov8s.pt"),
    )
    p.add_argument("--confidence", "-c", type=float, default=0.35)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-queue", action="store_true")
    p.add_argument("--save-events", "-o", type=str, default=None)
    p.add_argument("--record", type=str, default=None, help="Save annotated output as MP4")
    p.add_argument(
        "--no-window", action="store_true",
        help="Headless mode — no GUI window (use with --record)",
    )
    p.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO",
    )
    p.add_argument("--check-deps", action="store_true")
    return p


# ────────────────────────────────────────────────────────────────────
# Visual pipeline: frame-by-frame with overlay
# ────────────────────────────────────────────────────────────────────

def run_pipeline_interactive(args) -> int:
    """Process video frame-by-frame, showing live annotated output."""
    import logging
    import cv2

    logger = logging.getLogger("pipeline")

    # ── Dependencies ──
    missing = check_deps()
    if missing:
        logger.error("Missing: %s", ", ".join(missing))
        return 1

    # ── Heavy imports ──
    from edge_node.core.contracts import CrossingDirection, Point, LightState, ViolationEvent
    from edge_node.core.config import (
        RedStabilizerConfig, TripwireConfig, ViolationConfig,
        set_active_tripwire,
    )
    from edge_node.core.byte_tracker import ByteTrackerConfig, SupervisionByteTracker
    from edge_node.core.detector import YoloDetector
    from edge_node.core.traffic_light_cv import OpenCVTrafficLightClassifier
    from edge_node.core.violation_logic import RedLightStabilizer, ViolationDetector
    from edge_node.core.video_io import OpenCVFrameSource
    from edge_node.core.visualizer import LiveVisualizer

    # ── Video ──
    video = args.video
    if video is None:
        dv = find_default_video()
        if dv:
            video = str(dv)
            logger.info("Auto-detected video: %s", video)
        else:
            logger.error("No video found. Place .mp4 under edge_node/data/videos/")
            return 1

    video_path = Path(video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        return 1

    # ── Stop line ──
    if args.stop_line:
        start, end = args.stop_line
    else:
        start, end = Point(100, 400), Point(800, 400)
        logger.info("Default stop-line: (100,400)→(800,400)")

    tripwire = TripwireConfig(
        start=start, end=end,
        direction=CrossingDirection(args.direction),
    )
    set_active_tripwire(tripwire)

    # ── Model ──
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        return 1

    # ── Build components ──
    # Auto-detect GPU if no device specified
    device = args.device
    if device is None:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda:0"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info("GPU detected: %s (%.1f GB VRAM)", gpu_name, gpu_mem)
            else:
                device = "cpu"
                logger.info("No GPU available — using CPU")
        except ImportError:
            device = "cpu"
            logger.info("PyTorch not found — using CPU")

    detector = YoloDetector(str(model_path), confidence=args.confidence, device=device)
    tracker = SupervisionByteTracker(ByteTrackerConfig())
    classifier = OpenCVTrafficLightClassifier(roi=args.light_roi)
    stabilizer = RedLightStabilizer(RedStabilizerConfig())
    violation_detector = ViolationDetector(ViolationConfig(tripwire=tripwire))

    show = not args.no_window
    visualizer = LiveVisualizer(
        window_name=f"RLVD — {video_path.name}",
        show=show,
        record_path=args.record,
    )

    source = OpenCVFrameSource(
        str(video_path), max_frames=args.max_frames, loop=args.loop,
    )

    logger.info("Video:    %s", video_path)
    logger.info("Model:    %s", model_path)
    logger.info("Device:   %s", device)
    logger.info("Tripwire: (%d,%d)→(%d,%d) dir=%s",
                int(start.x), int(start.y), int(end.x), int(end.y), args.direction)
    logger.info("Live view: %s | Record: %s", show, args.record or "no")
    logger.info("Starting pipeline... (q/ESC=quit Space=pause)")

    # ── Frame loop ──
    events: list[ViolationEvent] = []
    frames_processed = 0
    paused = False
    last_events_since = 0
    recent_violations: list[ViolationEvent] = []

    try:
        for packet in source:
            if args.max_frames is not None and frames_processed >= args.max_frames:
                break

            # ── Pause handling ──
            if show:
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # Space
                    paused = not paused
                    if paused:
                        logger.info("⏸  PAUSED  (Space to resume, q to quit)")
                if key == 27 or key == ord("q"):
                    logger.info("User quit via keyboard")
                    break
                if paused:
                    # Still show the last frame when paused
                    continue

            # ── Process frame ──
            light = classifier.classify(packet.image, packet.frame_index, packet.timestamp_ms)
            stable_signal = stabilizer.update(light, packet.frame_index)
            detections = detector.detect(packet.image, packet.frame_index, packet.timestamp_ms)
            tracks = tracker.update(detections, packet.frame_index, packet.timestamp_ms)
            frame_events = violation_detector.update(
                tracks=tracks, signal=stable_signal,
                frame_index=packet.frame_index, timestamp_ms=packet.timestamp_ms,
            )

            if frame_events:
                recent_violations = list(frame_events)[:]
                last_events_since = 0
                events.extend(frame_events)

                # Queue dispatch
                if not args.no_queue:
                    _dispatch_to_queue(frame_events, logger)

            # Fade out violations after ~90 frames
            if recent_violations and last_events_since > 90:
                recent_violations = []
            last_events_since += 1

            # ── Render ──
            visualizer.update(
                packet.image,
                detections=detections,
                tracks=tracks,
                light=light,
                signal=stable_signal,
                violations=recent_violations,
                stop_line=(start, end),
                light_roi=args.light_roi,
            )

            frames_processed += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        visualizer.close()

    # ── Summary ──
    logger.info("=" * 50)
    logger.info("Pipeline done | frames=%d violations=%d", frames_processed, len(events))
    for i, ev in enumerate(events):
        logger.info(
            "  #%d frame=%-5d track=%-3d %-7s conf=%.2f pos=(%.0f, %.0f)",
            i + 1, ev.frame_index, ev.track_id,
            ev.light_state.value, ev.light_confidence,
            ev.crossing_point.x, ev.crossing_point.y,
        )

    if args.save_events:
        import json
        data = []
        for ev in events:
            data.append({
                "event_id": ev.event_id, "track_id": ev.track_id,
                "frame_index": ev.frame_index, "timestamp_ms": ev.timestamp_ms,
                "light_state": ev.light_state.value, "light_confidence": ev.light_confidence,
                "crossing_point": {"x": ev.crossing_point.x, "y": ev.crossing_point.y},
                "bbox": list(ev.bbox.as_xyxy()),
            })
        out = Path(args.save_events)
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info("Saved %d events → %s", len(data), out)

    return 0


def _dispatch_to_queue(frame_events, logger) -> None:
    """Try to enqueue violations via Celery."""
    try:
        from edge_node.worker.tasks import push_violation_to_server
        from edge_node.core.pipeline import _event_to_payload
    except ImportError:
        return
    for evt in frame_events:
        try:
            push_violation_to_server.delay(_event_to_payload(evt))
        except Exception as exc:
            logger.warning("Failed to enqueue %s: %s", evt.event_id, exc)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    _setup_logging(args.log_level)

    if args.check_deps:
        missing = check_deps()
        if missing:
            print("MISSING:", " ".join(missing))
            sys.exit(1)
        print("All dependencies present.")
        sys.exit(0)

    sys.exit(run_pipeline_interactive(args))