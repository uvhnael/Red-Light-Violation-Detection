#!/usr/bin/env python3
"""Standalone runner for the edge node detection pipeline — no Docker required.

On startup the runner AUTO-CALIBRATES: it detects the traffic light (YOLO,
HSV fallback) and the stop line from a representative frame, then uses those
as the light ROI + tripwire. Pass --stop-line / --light-roi to override, or
--no-calibrate to skip detection and use the hardcoded default stop line.

Usage
-----
    # Quick start — auto-detect traffic light + stop line, LIVE VIEW
    python run_pipeline.py

    # Custom video (still auto-calibrates)
    python run_pipeline.py edge_node/data/videos/tr.mp4

    # Override the stop line / light ROI manually (skips auto-calibration)
    python run_pipeline.py --stop-line 100,400,800,400
    python run_pipeline.py --light-roi 1634,214,144,128

    # Skip auto-calibration entirely (hardcoded default stop line)
    python run_pipeline.py --no-calibrate

    # Step-by-step stop-line detection (ENTER = next step, video plays live)
    python run_pipeline.py --debug-stop-line

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

    # Use Python 3.11 for CUDA support (3.14 has no GPU wheels yet)
    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
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
        # PROJECT_ROOT / "edge_node/data/videos/tr.mp4",
        # PROJECT_ROOT / "/home/uvhnael/projects/Red-Light-Violation-Detection/train_model/licenseplates/images/train/carlong_0001.png"
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
        "--no-calibrate", action="store_true",
        help="Skip auto-detection of traffic light + stop line (use default stop-line)",
    )
    p.add_argument(
        "--debug-stop-line", action="store_true",
        help="Step-by-step stop-line detection with visual stages (ENTER = next step)",
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
# Debug: step-by-step stop-line detection (ENTER to advance)
# ────────────────────────────────────────────────────────────────────

def _fit_for_display(img, max_w: int = 1600):
    """Downscale an image for on-screen display (original coords unchanged)."""
    import cv2
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)


def debug_stop_line_steps(video_path: str, logger):
    """Interactive step-by-step stop-line detection.

    Re-implements the ORIGINAL algorithm from the reference repo
    (yolo_video_new.py :: getLightThresh):

      Buoc 1:  frame goc
      Buoc 2:  traffic light (YOLO26)
      Buoc 3:  resize 1000x750 (giong code goc)
      Buoc 4:  grayscale
      Buoc 5:  adaptiveThreshold(blockSize=115, C=1)
      Buoc 6:  erode 3x3 (1 lan) + dilate 3x3 (2 lan)
      Buoc 7:  contours -> giu hinh chu nhat 4 canh (area>800, len<100, eps=0.04)
      Buoc 8:  giu rect duoi traffic light
      Buoc 9:  khoang cach den den -> rect gan nhat
      Buoc 10: ket qua stop line (scale ve full-res)

    Each stage is drawn in its own window. Press ENTER to advance to the next
    step, q/ESC to abort. The video keeps playing continuously in a separate
    live window while you inspect each step.

    Returns (light_roi, stop_line_y, frame_width); values are None on abort
    or when detection fails.
    """
    import cv2
    import numpy as np
    from edge_node.core.calibration import detect_traffic_light

    live_win = "LIVE video (chay lien tuc)"
    stage_win = "Stop-line DEBUG — ENTER = next step, q = quit"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("debug: cannot open %s", video_path)
        return None, None, None

    def pump_live():
        """Read + show one live frame; loops the video so it never stops."""
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                return None
        cv2.imshow(live_win, _fit_for_display(frame))
        return frame

    def wait_enter(hint: str) -> bool:
        """Keep the live video playing until ENTER. False = abort (q/ESC)."""
        logger.info("%s  ->  bam ENTER de tiep tuc (q = thoat)", hint)
        while True:
            pump_live()
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10):  # ENTER
                return True
            if key in (27, ord("q")):
                return False

    def step(title: str, img) -> bool:
        """Show one processing stage and wait for ENTER."""
        view = img.copy()
        if view.ndim == 2:
            view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
        cv2.putText(view, title, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 220, 0), 4)
        cv2.imshow(stage_win, _fit_for_display(view))
        return wait_enter(title)

    try:
        # ── Buoc 0: video chay lien tuc, ENTER de chup frame hien tai ──
        pump_live()
        if not wait_enter("Buoc 0: Video dang chay lien tuc. Bam ENTER de chup frame hien tai"):
            return None, None, None
        frame = pump_live()
        if frame is None:
            logger.error("debug: cannot read frame")
            return None, None, None
        frame = frame.copy()
        frame_w = frame.shape[1]

        # ── Buoc 1: frame goc ──
        if not step("Buoc 1: Frame goc", frame):
            return None, None, None

        # ── Buoc 2: traffic light (YOLO26) ──
        light_roi, light_source = detect_traffic_light(frame)
        if light_roi is None:
            logger.warning("debug: khong tim thay traffic light")
            return None, None, frame_w
        xl, yl, wl, hl = light_roi
        logger.info("Traffic light (%s): x=%d y=%d w=%d h=%d",
                    light_source, xl, yl, wl, hl)
        marked = frame.copy()
        cv2.rectangle(marked, (xl, yl), (xl + wl, yl + hl), (0, 0, 255), 6)
        cv2.putText(marked, f"traffic light ({light_source})",
                    (xl, max(50, yl - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
        if not step("Buoc 2: Traffic light da detect", marked):
            return light_roi, None, frame_w

        # ── Buoc 3: resize ve 1000x750 (giong code goc getLightThresh) ──
        RW, RH = 1000, 750
        resized = cv2.resize(frame, (RW, RH))
        # scale light ROI sang khong gian resized
        sx, sy = RW / frame_w, RH / frame.shape[0]
        xl_r, yl_r = int(xl * sx), int(yl * sy)
        wl_r, hl_r = int(wl * sx), int(hl * sy)
        if not step("Buoc 3: Resize 1000x750", resized):
            return light_roi, None, frame_w

        # ── Buoc 4: grayscale ──
        grayscaled = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        if not step("Buoc 4: Grayscale", grayscaled):
            return light_roi, None, frame_w

        # ── Buoc 5: adaptive threshold (blockSize=115, C=1 — code goc) ──
        th = cv2.adaptiveThreshold(
            grayscaled, 250,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            115, 1,
        )
        if not step("Buoc 5: Adaptive threshold (115, C=1)", th):
            return light_roi, None, frame_w

        # ── Buoc 6: erode 3x3 (1 lan) + dilate 3x3 (2 lan) ──
        kernel = np.ones((3, 3), np.uint8)
        th = cv2.erode(th, kernel, iterations=1)
        th = cv2.dilate(th, kernel, iterations=2)
        if not step("Buoc 6: Erode(1) + Dilate(2)", th):
            return light_roi, None, frame_w

        # ── Buoc 7: contours -> giu hinh chu nhat 4 canh ──
        # Code goc: area>800, len(contour)<100, approxPolyDP eps=0.04, 4 canh
        contours, _ = cv2.findContours(
            th, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS,
        )
        rects: list = []  # (x, y, w, h) trong khong gian resized
        rect_view = resized.copy()
        for c in contours:
            if cv2.contourArea(c) > 800 and len(c) < 100:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.drawContours(rect_view, [c], -1, (0, 255, 0), 2)
                    cv2.rectangle(rect_view, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    rects.append((x, y, w, h))
        if not step(f"Buoc 7: Hinh chu nhat 4 canh ({len(rects)})", rect_view):
            return light_roi, None, frame_w
        if not rects:
            logger.warning("debug: khong tim thay hinh chu nhat 4 canh nao")
            return light_roi, None, frame_w

        # ── Buoc 8: giu rect duoi den + chon rect gan den nhat ──
        below = [r for r in rects if r[1] > yl_r + hl_r]
        below_view = resized.copy()
        cv2.rectangle(below_view, (xl_r, yl_r), (xl_r + wl_r, yl_r + hl_r), (0, 0, 255), 3)
        for (x, y, w, h) in below:
            cv2.rectangle(below_view, (x, y), (x + w, y + h), (255, 128, 0), 2)
        if not step(f"Buoc 8: Rect duoi traffic light ({len(below)})", below_view):
            return light_roi, None, frame_w
        if not below:
            logger.warning("debug: khong co hinh chu nhat nao duoi traffic light")
            return light_roi, None, frame_w

        # khoang cach den den -> rect gan nhat (code goc: dist tu (xlight,ylight) den (x,y))
        dist_view = resized.copy()
        cv2.rectangle(dist_view, (xl_r, yl_r), (xl_r + wl_r, yl_r + hl_r), (0, 0, 255), 3)
        min_index, min_distance = 0, float("inf")
        for i, (x, y, w, h) in enumerate(below):
            cv2.line(dist_view, (xl_r, yl_r), (x, y), (0, 0, 255), 2)
            distance = ((x - xl_r) ** 2 + (y - yl_r) ** 2) ** 0.5
            cv2.putText(dist_view, f"{distance:.0f}", (x, max(30, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if distance < min_distance:
                min_distance = distance
                min_index = i
        if not step("Buoc 9: Khoang cach den traffic light", dist_view):
            return light_roi, None, frame_w

        # ── Buoc 10: ket qua — scale y ve frame goc ──
        x, y, w, h = below[min_index]
        stop_y_full = int(y * frame.shape[0] / RH)  # scale ve full-res
        final_view = frame.copy()
        # ve rect o full-res
        fx, fy = int(x / sx), int(y / sy)
        fw_r, fh_r = int(w / sx), int(h / sy)
        cv2.rectangle(final_view, (fx, fy), (fx + fw_r, fy + fh_r), (0, 0, 255), 6)
        cv2.line(final_view, (0, stop_y_full), (frame_w, stop_y_full), (0, 0, 0), 12, cv2.LINE_AA)
        cv2.putText(final_view, f"STOP LINE y={stop_y_full}", (40, max(80, stop_y_full - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)
        step(f"Buoc 10: Ket qua — stop line y={stop_y_full}", final_view)

        logger.info("debug: stop line y=%d (rect resized=%s)", stop_y_full, (x, y, w, h))
        return light_roi, stop_y_full, frame_w
    finally:
        cap.release()
        for win in (live_win, stage_win):
            try:
                cv2.destroyWindow(win)
            except cv2.error:
                pass  # window never created (headless) — ignore


def _debug_calibrate(video_path: str, explicit_roi, logger):
    """Run the interactive step-by-step stop-line detection.

    Returns (start, end, light_roi) for the pipeline. Falls back to the
    hardcoded default stop line if the user aborts or detection fails.
    """
    from edge_node.core.contracts import Point
    from edge_node.core.config import set_active_light_roi

    logger.info("Interactive stop-line debug — bam ENTER de qua tung buoc")
    light_roi, stop_y, frame_w = debug_stop_line_steps(video_path, logger)

    roi = explicit_roi if explicit_roi is not None else light_roi
    if roi is not None:
        set_active_light_roi(roi)

    if stop_y is not None and frame_w:
        start, end = Point(0, stop_y), Point(frame_w, stop_y)
        logger.info("Debug result: stop line y=%d, light ROI=%s", stop_y, roi)
        return start, end, roi

    logger.warning("Debug aborted/failed — using default stop-line")
    return Point(100, 400), Point(800, 400), roi


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
    from edge_node.core.traffic_light_cv import OpenCVTrafficLightClassifier
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
    # Priority: explicit --stop-line > --debug-stop-line (interactive) >
    # auto-calibration > hardcoded default.
    light_roi = args.light_roi  # explicit --light-roi wins when provided
    if args.stop_line:
        start, end = args.stop_line
        logger.info("Manual stop-line: (%d,%d)→(%d,%d)",
                    int(start.x), int(start.y), int(end.x), int(end.y))
    elif args.debug_stop_line:
        start, end, light_roi = _debug_calibrate(
            str(video_path), args.light_roi, logger,
        )
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
    classifier = OpenCVTrafficLightClassifier(roi=light_roi)
    stabilizer = RedLightStabilizer(RedStabilizerConfig())
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