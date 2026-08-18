"""Auto-calibration: detect the traffic light and stop line from a video frame.

Faithful port of the reference implementation (``getLightThresh`` in
``yolo_video_new.py`` of the original Fully-Automated-red-light-Violation-
Detection repo):

1. Detect the traffic light with YOLO26 (``traffic_light`` class).
2. Resize the frame to 1000x750 — the resolution the original thresholds
   were tuned on — and scale the light ROI accordingly.
3. Grayscale + adaptive threshold (blockSize=115, C=1) + erode/dilate.
4. Keep 4-sided rectangle contours (area > 800, < 100 points, eps = 0.04)
   located *below* the traffic light.
5. The rectangle closest to the light is taken as the stop line.

The result is scaled back to raw-frame pixel coordinates so the web
dashboard can draw overlays on a snapshot without rescaling.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

# Thresholds ported from the reference repo's getLightThresh() — tuned for
# the 1000x750 working resolution below. Do NOT "fix" these to match the
# raw frame size: the original never rescaled them.
STOP_LINE_CFG = {
    "RESIZE_WIDTH": 1000,
    "RESIZE_HEIGHT": 750,
    "ADAPTIVE_THRESH_BLOCK_SIZE": 115,
    "ADAPTIVE_THRESH_C": 1,
    "ERODE_ITERATIONS": 1,
    "DILATE_ITERATIONS": 2,
    "MIN_CONTOUR_AREA": 800,
    "MAX_CONTOUR_POINTS": 100,
    "APPROX_POLY_EPSILON": 0.04,
}

# Shared YOLO detector so repeated calibration calls reuse the loaded model
_DETECTOR = None
_DETECTOR_LOCK = threading.Lock()

# Traffic lights are small and often detected below the vehicle confidence
# threshold, so calibration uses a dedicated lower threshold for them.
LIGHT_CALIB_CONFIDENCE = 0.25
# Frame offsets tried (in order) when locating the traffic light — the first
# frame that yields a detection wins.
LIGHT_CALIB_FRAME_OFFSETS = (30, 0, 60, 90, 120)


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
    """Locate the traffic light as (x, y, w, h) using YOLO26.

    Only the YOLO detector is used — classic CV/HSV blob searches proved too
    unreliable on real footage. Note: COCO class names contain a space
    ("traffic light"), so labels are normalised before comparing.
    """
    try:
        detector = _get_shared_detector()
        detections = detector.detect(frame, 0, 0.0, confidence=LIGHT_CALIB_CONFIDENCE)
        lights = [
            d for d in detections
            if d.label.replace(" ", "_") == "traffic_light"
        ]
        if lights:
            best = max(lights, key=lambda d: d.confidence)
            x1, y1, x2, y2 = best.bbox.as_xyxy()
            roi = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            LOGGER.info(
                "Calibration: traffic light via YOLO at %s (conf=%.2f)",
                roi, best.confidence,
            )
            return roi, "yolo"
        LOGGER.warning("Calibration: YOLO found no traffic_light in frame")
    except Exception as exc:
        LOGGER.warning("Calibration: YOLO traffic-light detection failed: %s", exc)

    return None, "none"


def detect_stop_line(
    frame: np.ndarray,
    light_roi: tuple[int, int, int, int],
) -> tuple[Optional[int], Optional[tuple[int, int, int, int]]]:
    """Detect the stop line y-coordinate (and its bounding rectangle).

    Faithful port of the reference repo's ``getLightThresh``
    (yolo_video_new.py):

    resize to 1000x750 -> grayscale -> adaptiveThreshold(blockSize=115,
    C=1) -> erode(1)/dilate(2) with a 3x3 kernel -> keep 4-sided contours
    (area > 800, < 100 points, approxPolyDP eps = 0.04) located *below*
    the traffic light -> the rectangle closest to the light wins.

    All thresholding happens in the 1000x750 working space (the resolution
    the original thresholds were tuned on); the result is scaled back to
    full-resolution coordinates. Returns ``(stop_line_y, rect)`` where
    ``stop_line_y`` is the top edge of the chosen rectangle.
    """
    frame_h, frame_w = frame.shape[:2]
    work_w = STOP_LINE_CFG["RESIZE_WIDTH"]
    work_h = STOP_LINE_CFG["RESIZE_HEIGHT"]
    resized = cv2.resize(frame, (work_w, work_h))
    sx, sy = work_w / frame_w, work_h / frame_h

    xlight, ylight, wlight, hlight = light_roi
    xl_r, yl_r = int(xlight * sx), int(ylight * sy)
    wl_r, hl_r = int(wlight * sx), int(hlight * sy)

    grayscaled = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
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

    rects: list[tuple[int, int, int, int]] = []
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
                rects.append(cv2.boundingRect(contour))

    # Keep only rectangles below the traffic light
    below = [r for r in rects if r[1] > yl_r + hl_r]
    if not below:
        LOGGER.warning("Calibration: no stop-line rectangle below the traffic light")
        return None, None

    # Rectangle closest to the traffic light = most likely stop line
    min_index = 0
    min_distance = float("inf")
    for i, (x, y, _w, _h) in enumerate(below):
        distance = ((x - xl_r) ** 2 + (y - yl_r) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            min_index = i

    x, y, w, h = below[min_index]
    # Scale the result back to full-resolution coordinates
    full_rect = (int(x / sx), int(y / sy), int(w / sx), int(h / sy))
    stop_y = int(y / sy)
    LOGGER.info("Calibration: stop line at y=%s (rect=%s)", stop_y, full_rect)
    return stop_y, full_rect


def run_calibration(video_path: str) -> CalibrationResult:
    """Full calibration pass over one frame of *video_path*.

    The traffic light is searched for on several candidate frames (see
    ``LIGHT_CALIB_FRAME_OFFSETS``) because a single frame may miss it — the
    light can be small, occluded, or below the detection threshold on any
    given frame.  The first frame that yields a detection is used for both
    the light ROI and the stop-line search.
    """
    frame = grab_calibration_frame(video_path)
    if frame is None:
        raise RuntimeError(f"Cannot read a frame from {video_path}")

    height, width = frame.shape[:2]
    light_roi, light_source = detect_traffic_light(frame)

    # Retry on other frames when the light was not found on the first one.
    if light_roi is None:
        for offset in LIGHT_CALIB_FRAME_OFFSETS:
            if offset == 30:  # already tried via grab_calibration_frame
                continue
            alt = grab_calibration_frame(video_path, skip_frames=offset)
            if alt is None:
                continue
            light_roi, light_source = detect_traffic_light(alt)
            if light_roi is not None:
                frame = alt  # use this frame for the stop-line search too
                LOGGER.info(
                    "Calibration: traffic light found on frame offset %s", offset,
                )
                break

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
