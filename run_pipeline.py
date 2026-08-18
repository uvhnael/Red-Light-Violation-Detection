#!/usr/bin/env python3
"""Standalone runner for the edge node detection pipeline — no Docker required.

On startup the runner AUTO-CALIBRATES: it detects the traffic light (YOLO,
HSV fallback) and the stop line from a representative frame, then uses those
as the light ROI + tripwire. Pass --stop-line / --light-roi to override, or
--no-calibrate to skip detection and use the hardcoded default stop line.

Usage:
    python run_pipeline.py                          # auto-detect video, live view
    python run_pipeline.py video.mp4                # custom video
    python run_pipeline.py --stop-line 100,400,800,400
    python run_pipeline.py --light-roi 1634,214,144,128
    python run_pipeline.py --no-window --record out.mp4
    python run_pipeline.py --loop --max-frames 300 --no-queue

Controls (live window): q/ESC = quit, Space = pause/resume.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

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
    video_dir = PROJECT_ROOT / "edge_node/data/videos"
    if video_dir.is_dir():
        for ext in ("*.mp4", "*.MP4", "*.avi", "*.mkv"):
            files = sorted(video_dir.glob(ext))
            if files:
                return files[0]
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Red-Light Violation Detection Pipeline — live view runner.",
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
        "--no-calibrate", action="store_true",
        help="Skip auto-detection of traffic light + stop line (use default stop-line)",
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
        "--ocr", action="store_true",
        help="Enable plate OCR on violation events (requires fast-plate-ocr)",
    )
    p.add_argument(
        "--plate-model", type=str,
        default=str(PROJECT_ROOT / "edge_node/models/license_plate_yolo26.pt"),
        help="License-plate detection model (BSD/BSV)",
    )
    p.add_argument(
        "--no-plates", action="store_true",
        help="Disable license-plate detection + OCR overlay",
    )
    p.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO",
    )
    p.add_argument("--check-deps", action="store_true")
    return p


# ────────────────────────────────────────────────────────────────────
# Auto-calibration: detect traffic light + stop line from the video
# ────────────────────────────────────────────────────────────────────

def _auto_calibrate(video_path: str, explicit_roi, logger):
    """Run auto-calibration and return (start, end, light_roi).

    Detects the traffic light (YOLO, HSV fallback) and the stop line from a
    representative frame, then applies both to the running pipeline (active
    light ROI + full-width tripwire at the detected y). Falls back to the
    hardcoded default stop line if detection fails.
    """
    from edge_node.core.contracts import Point
    from edge_node.core.config import set_active_light_roi
    from edge_node.core.calibration import run_calibration

    logger.info("Auto-calibrating traffic light + stop line from %s ...", video_path)
    try:
        result = run_calibration(video_path)
    except Exception as exc:
        logger.warning("Auto-calibration failed (%s) — using default stop-line", exc)
        return Point(100, 400), Point(800, 400), explicit_roi

    # Traffic-light ROI: explicit flag wins, otherwise use the detected box
    light_roi = explicit_roi
    if result.light_roi is not None:
        if light_roi is None:
            light_roi = result.light_roi
            set_active_light_roi(result.light_roi)
        logger.info("Traffic light (%s): x=%d y=%d w=%d h=%d",
                    result.light_source, *result.light_roi)
    else:
        logger.warning("No traffic light detected — classifier will scan full frame")

    # Stop line: full-width tripwire at the detected y
    if result.stop_line_y is not None:
        y = result.stop_line_y
        start, end = Point(0, y), Point(result.frame_width, y)
        logger.info("Stop line detected at y=%d (rect=%s)", y, result.stop_line_rect)
        return start, end, light_roi

    logger.warning("No stop line detected — using default stop-line")
    return Point(100, 400), Point(800, 400), light_roi


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
    from edge_node.core.contracts import CrossingDirection, Point
    from edge_node.core.config import (
        RedStabilizerConfig, TripwireConfig, ViolationConfig,
        set_active_tripwire,
    )
    from edge_node.core.byte_tracker import ByteTrackerConfig, SupervisionByteTracker
    from edge_node.core.detector import YoloDetector
    from edge_node.core.plate_detector import PlateDetector
    from edge_node.core.traffic_light_yolo import create_light_classifier
    from edge_node.core.violation_logic import RedLightStabilizer, ViolationDetector
    from edge_node.core.video_io import OpenCVFrameSource
    from edge_node.core.visualizer import LiveVisualizer
    from edge_node.core.ocr_recognizer import FastPlateOCR
    from edge_node.core.plate_associator import PlateAssociator

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

    # ── Stop line + traffic light ──
    # Priority: explicit --stop-line > auto-calibration > hardcoded default.
    light_roi = args.light_roi  # explicit --light-roi wins when provided
    if args.stop_line:
        start, end = args.stop_line
        logger.info("Manual stop-line: (%d,%d)→(%d,%d)",
                    int(start.x), int(start.y), int(end.x), int(end.y))
    elif args.no_calibrate:
        start, end = Point(100, 400), Point(800, 400)
        logger.info("Default stop-line (--no-calibrate): (100,400)→(800,400)")
    else:
        start, end, light_roi = _auto_calibrate(
            str(video_path), args.light_roi, logger,
        )

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
    classifier = create_light_classifier(roi=light_roi, device=device)

    # Anti-flicker stabilizer: env-tunable via RED_STABLE_FRAMES /
    # RED_SWITCH_FRAMES / RED_MIN_CONFIDENCE (see edge_node/settings.py).
    from edge_node.settings import get_settings
    _settings = get_settings()
    stabilizer = RedLightStabilizer(RedStabilizerConfig(
        required_consecutive_frames=_settings.red_stable_frames,
        switch_consecutive_frames=_settings.red_switch_frames,
        min_confidence=_settings.red_min_confidence,
    ))
    violation_detector = ViolationDetector(ViolationConfig(tripwire=tripwire))

    # ── License-plate detection + OCR overlay (on by default) ──
    plate_detector = None
    plate_associator = None
    plate_ocr = None
    if not args.no_plates:
        plate_model_path = Path(args.plate_model)
        if not plate_model_path.is_absolute():
            plate_model_path = PROJECT_ROOT / plate_model_path
        if plate_model_path.exists():
            plate_detector = PlateDetector(str(plate_model_path), device=device)
            plate_associator = PlateAssociator()
            ocr_device = "cuda" if (device and device.startswith("cuda")) else "auto"
            plate_ocr = FastPlateOCR(device=ocr_device)
            logger.info("Plates:   %s + fast-plate-ocr (device=%s)",
                        plate_model_path.name, ocr_device)
        else:
            logger.warning("Plate model not found: %s — plate overlay disabled",
                           plate_model_path)

    # ── OCR on violation events (optional, reuses plate OCR engine) ──
    ocr = None
    if args.ocr:
        if plate_ocr is not None:
            ocr = plate_ocr
        else:
            ocr_device = "cuda" if (device and device.startswith("cuda")) else "auto"
            ocr = FastPlateOCR(device=ocr_device)
            logger.info("OCR:      fast-plate-ocr (device=%s)", ocr_device)

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

    # ── Run through the SAME pipeline class the edge node uses ──
    # This keeps run_pipeline behaviour identical to edge_node.main: the
    # core light->stabilize->detect->track->violation logic lives in
    # RedLightViolationPipeline, and we only hook a per-frame callback for
    # the live view (plate overlay + rendering + keyboard control).
    from edge_node.core.pipeline import RedLightViolationPipeline

    stop = {"flag": False}
    view_state = {"recent": [], "since": 0}

    def on_frame(packet, light, stable_signal, detections, tracks, frame_events):
        # ── Keyboard: quit / pause ──
        if show:
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                stop["flag"] = True
                logger.info("User quit via keyboard")
                return
            if key == 32:  # Space -> pause until Space again or q
                logger.info("PAUSED (Space to resume, q to quit)")
                while True:
                    k2 = cv2.waitKey(50) & 0xFF
                    if k2 == 32:
                        break
                    if k2 in (27, ord("q")):
                        stop["flag"] = True
                        return

        # ── License-plate detection + association (overlay only) ──
        track_plates: dict = {}
        unassigned_plates: list = []
        if plate_detector is not None and plate_associator is not None:
            try:
                plate_dets = plate_detector.detect(
                    packet.image, packet.frame_index, packet.timestamp_ms,
                )
                track_plates, unassigned_plates = plate_associator.update(
                    packet.image, tracks, plate_dets, plate_ocr,
                    packet.frame_index,
                )
            except Exception as exc:
                logger.debug("Plate detection error: %s", exc)

        # ── Violation fade-out (~90 frames) ──
        if frame_events:
            view_state["recent"] = list(frame_events)
            view_state["since"] = 0
        if view_state["recent"] and view_state["since"] > 90:
            view_state["recent"] = []
        view_state["since"] += 1

        # ── Render ──
        visualizer.update(
            packet.image,
            detections=detections,
            tracks=tracks,
            light=light,
            signal=stable_signal,
            violations=view_state["recent"],
            stop_line=(start, end),
            light_roi=light_roi,
            track_plates=track_plates,
            plates=unassigned_plates,
        )

    # ── Durable outbox + background batch sender (same as edge_node.main) ──
    outbox = None
    sender = None
    if not args.no_queue:
        from edge_node.outbox import ViolationOutbox
        from edge_node.violation_sender import ViolationSender
        from edge_node.settings import get_settings

        settings = get_settings()
        outbox = ViolationOutbox(settings.outbox_db_path)
        sender = ViolationSender(outbox, settings)
        sender.start()
        pending = outbox.pending_count()
        if pending:
            logger.info("Outbox has %d pending violation(s) from previous runs", pending)

    pipeline = RedLightViolationPipeline(
        detector=detector,
        tracker=tracker,
        light_classifier=classifier,
        stabilizer=stabilizer,
        violation_detector=violation_detector,
        ocr=ocr,
        enable_queue=False,  # legacy Celery path replaced by the outbox
        logger=logger,
        frame_callback=on_frame,
        outbox=outbox,
    )

    class _StopSource:
        """Wrap the frame source so the quit flag can halt the pipeline."""

        def __init__(self, inner):
            self._inner = inner

        def __iter__(self):
            for packet in self._inner:
                if stop["flag"]:
                    return
                yield packet

    result = None
    try:
        result = pipeline.process(_StopSource(source), max_frames=args.max_frames)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        visualizer.close()

    # ── Drain the outbox before exiting ──
    if sender is not None and outbox is not None:
        import time as _time

        deadline = _time.monotonic() + 10
        while outbox.pending_count() > 0 and _time.monotonic() < deadline:
            _time.sleep(0.5)
        remaining = outbox.pending_count()
        if remaining:
            logger.warning(
                "%d violation(s) still pending in outbox — will be sent on next run",
                remaining,
            )
        else:
            logger.info("Outbox drained — all violations delivered")
        sender.stop()
        outbox.close()

    frames_processed = result.frames_processed if result else 0
    events = list(result.violations) if result else []

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
