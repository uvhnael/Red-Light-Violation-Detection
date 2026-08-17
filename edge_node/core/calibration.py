"""Auto-calibration: detect the traffic light and stop line from a video frame.

Ported from the legacy ``detect_stop_line.py`` approach:

1. Detect the traffic light (YOLO ``traffic_light`` class, HSV blob fallback).
2. Grayscale + adaptive threshold + morphology on the frame.
3. Keep 4-sided rectangle contours located *below* the traffic light.
4. The rectangle closest to the light is taken as the stop line.

The result is expressed in the same pixel coordinates as the raw frame so
the web dashboard can draw overlays on a snapshot without rescaling.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

# Thresholds ported from the legacy STOP_LINE_DETECTION config
STOP_LINE_CFG = {
    "ADAPTIVE_THRESH_BLOCK_SIZE": 11,
    "ADAPTIVE_THRESH_C": 2,
    "ERODE_ITERATIONS": 1,
    "DILATE_ITERATIONS": 2,
    "MIN_CONTOUR_AREA": 1000,
    "MAX_CONTOUR_POINTS": 500,
    "APPROX_POLY_EPSILON": 0.02,
}

# Shared YOLO detector so repeated calibration calls reuse the loaded model
_DETECTOR = None
_DETECTOR_LOCK = threading.Lock()


def _get_shared_detector():
    """Lazily load one shared YoloDetector for calibration calls."""
    global _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            from edge_node.core.detector import YoloDetector
            from edge_node.settings import get_settings

            settings = get_settings()
            LOGGER.info("Calibration: loading YOLO model %s", settings.yolo_model_path)
            _DETECTOR = YoloDetector(
                model_path=settings.yolo_model_path,
                confidence=settings.yolo_confidence,
                img_size=settings.yolo_img_size,
                device=settings.yolo_device or None,
            )
        return _DETECTOR


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of one calibration run (all coords in raw-frame pixels)."""

    frame_width: int
    frame_height: int
    light_roi: Optional[tuple[int, int, int, int]]  # (x, y, w, h)
    light_source: str  # "yolo" | "hsv" | "none"
    stop_line_y: Optional[int]
    stop_line_rect: Optional[tuple[int, int, int, int]]  # (x, y, w, h)


def grab_calibration_frame(
    video_path: str, skip_frames: int = 30,
) -> Optional[np.ndarray]:
    """Read one representative frame (skipping the first few, often black)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        LOGGER.error("Calibration: cannot open video %s", video_path)
        return None
    try:
        frame: Optional[np.ndarray] = None
        for _ in range(skip_frames + 1):
            ok, f = cap.read()
            if not ok:
                break
            frame = f
        return frame
    finally:
        cap.release()


def detect_traffic_light(
    frame: np.ndarray,
) -> tuple[Optional[tuple[int, int, int, int]], str]:
    """Locate the traffic light as (x, y, w, h).

    Tries YOLO first (robust, any lamp colour), then falls back to the
    HSV blob search used by the runtime classifier.
    """
    # 1) YOLO — the standard detector map already includes class 9 (traffic_light)
    try:
        detector = _get_shared_detector()
        detections = detector.detect(frame, 0, 0.0)
        lights = [d for d in detections if d.label == "traffic_light"]
        if lights:
            best = max(lights, key=lambda d: d.confidence)
            x1, y1, x2, y2 = best.bbox.as_xyxy()
            roi = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            LOGGER.info(
                "Calibration: traffic light via YOLO at %s (conf=%.2f)",
                roi, best.confidence,
            )
            return roi, "yolo"
    except Exception as exc:
        LOGGER.warning("Calibration: YOLO traffic-light detection failed: %s", exc)

    # 2) HSV fallback — finds a lit coloured lamp in the upper scene
    try:
        from edge_node.core.traffic_light_cv import OpenCVTrafficLightClassifier

        roi = OpenCVTrafficLightClassifier().locate_roi(frame)
        if roi is not None:
            LOGGER.info("Calibration: traffic light via HSV at %s", roi)
            return roi, "hsv"
    except Exception as exc:
        LOGGER.warning("Calibration: HSV traffic-light detection failed: %s", exc)

    return None, "none"


def detect_stop_line(
    frame: np.ndarray,
    light_roi: tuple[int, int, int, int],
) -> tuple[Optional[int], Optional[tuple[int, int, int, int]]]:
    """Detect the stop line y-coordinate (and its bounding rectangle).

    Same CV pipeline as the legacy implementation:
    grayscale -> adaptive threshold -> erode/dilate -> 4-sided contours,
    keeping rectangles below the traffic light and picking the one whose
    top edge is closest to the light.
    """
    xlight, ylight, wlight, hlight = light_roi

    grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(
        grayscaled, 250,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        STOP_LINE_CFG["ADAPTIVE_THRESH_BLOCK_SIZE"],
        STOP_LINE_CFG["ADAPTIVE_THRESH_C"],
    )
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.erode(th, kernel, iterations=STOP_LINE_CFG["ERODE_ITERATIONS"])
    th = cv2.dilate(th, kernel, iterations=STOP_LINE_CFG["DILATE_ITERATIONS"])

    contours, _ = cv2.findContours(
        th, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS,
    )

    all_contours: list[tuple[int, int, int, int]] = []
    for contour in contours:
        if (
            cv2.contourArea(contour) > STOP_LINE_CFG["MIN_CONTOUR_AREA"]
            and len(contour) < STOP_LINE_CFG["MAX_CONTOUR_POINTS"]
        ):
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(
                contour, STOP_LINE_CFG["APPROX_POLY_EPSILON"] * peri, True,
            )
            if len(approx) == 4:  # rectangle candidates only
                all_contours.append(cv2.boundingRect(contour))

    # Keep only rectangles below the traffic light
    all_contours = [r for r in all_contours if r[1] > ylight + hlight]
    if not all_contours:
        LOGGER.warning("Calibration: no stop-line rectangle below the traffic light")
        return None, None

    # Rectangle closest to the traffic light = most likely stop line
    min_index = 0
    min_distance = float("inf")
    for i, (x, y, _w, _h) in enumerate(all_contours):
        if ylight + hlight < y:
            distance = ((x - xlight) ** 2 + (y - ylight) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                min_index = i

    x, y, w, h = all_contours[min_index]
    LOGGER.info("Calibration: stop line at y=%s (rect=%s)", y, (x, y, w, h))
    return y, (x, y, w, h)


def run_calibration(video_path: str) -> CalibrationResult:
    """Full calibration pass over one frame of *video_path*."""
    frame = grab_calibration_frame(video_path)
    if frame is None:
        raise RuntimeError(f"Cannot read a frame from {video_path}")

    height, width = frame.shape[:2]
    light_roi, light_source = detect_traffic_light(frame)

    stop_y: Optional[int] = None
    stop_rect: Optional[tuple[int, int, int, int]] = None
    if light_roi is not None:
        stop_y, stop_rect = detect_stop_line(frame, light_roi)

    return CalibrationResult(
        frame_width=width,
        frame_height=height,
        light_roi=light_roi,
        light_source=light_source,
        stop_line_y=stop_y,
        stop_line_rect=stop_rect,
    )
