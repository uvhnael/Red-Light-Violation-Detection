"""Traffic-light detection based only on OpenCV image processing.

This module deliberately contains no learned model or model weights.  Supplying
the traffic-light ROI is recommended: it prevents vehicle tail lights, signs,
and reflections elsewhere in the image from being interpreted as a signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from edge_node.core.contracts import Frame, LightObservation, LightState


@dataclass(frozen=True)
class TrafficLightCvConfig:
    """Thresholds for HSV-based signal detection."""

    min_saturation: int = 90
    min_value: int = 120
    min_coverage: float = 0.0005
    search_top_fraction: float = 0.65


class OpenCVTrafficLightClassifier:
    """Classify an illuminated red, yellow, or green lamp using HSV masks.

    ``roi`` uses ``(x, y, width, height)`` pixel coordinates.  When omitted,
    the first sufficiently bright coloured blob in the upper part of the frame
    is used as a best-effort ROI and then retained for subsequent frames.
    """

    _RANGES = {
        LightState.RED: ((0, 170), (170, 180)),
        LightState.YELLOW: ((15, 40),),
        LightState.GREEN: ((40, 95),),
    }

    def __init__(
        self,
        roi: Optional[tuple[int, int, int, int]] = None,
        config: TrafficLightCvConfig = TrafficLightCvConfig(),
    ) -> None:
        self._roi = roi
        self._config = config

    @property
    def roi(self) -> Optional[tuple[int, int, int, int]]:
        return self._roi

    def classify(self, frame: Frame, frame_index: int, timestamp_ms: float) -> LightObservation:
        del frame_index, timestamp_ms
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return LightObservation(LightState.UNKNOWN, 0.0, source="opencv-hsv")
        # A ROI set via the control-plane API (web re-calibration) wins over
        # the instance ROI so operators can correct detection at runtime.
        from edge_node.core.config import get_active_light_roi
        roi = self._valid_roi(frame, get_active_light_roi()) or self._valid_roi(frame, self._roi)
        if roi is None:
            roi = self.locate_roi(frame)
            if roi is not None:
                self._roi = roi
        if roi is None:
            return LightObservation(LightState.UNKNOWN, 0.0, source="opencv-hsv")

        x, y, width, height = roi
        state, confidence = self.classify_region(frame[y : y + height, x : x + width])
        return LightObservation(state, confidence, source="opencv-hsv")

    def classify_region(self, image: np.ndarray) -> tuple[LightState, float]:
        """Return a state and confidence for a BGR traffic-light crop."""
        if image is None or image.size == 0 or image.ndim != 3:
            return LightState.UNKNOWN, 0.0

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        scores = {state: self._evidence(hsv, ranges) for state, ranges in self._RANGES.items()}
        state, score = max(scores.items(), key=lambda item: item[1])
        coverage = score / float(hsv.shape[0] * hsv.shape[1])
        if coverage < self._config.min_coverage:
            return LightState.UNKNOWN, 0.0

        total = sum(scores.values())
        dominance = score / total if total else 0.0
        # A small, bright circular lamp is valid; confidence therefore combines
        # colour dominance with a saturating coverage term rather than requiring
        # a large part of the ROI to be coloured.
        confidence = dominance * min(1.0, coverage / 0.01)
        return state, float(confidence)

    def _mask(self, hsv: np.ndarray, ranges: tuple[tuple[int, int], ...]) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low_h, high_h in ranges:
            mask |= cv2.inRange(
                hsv,
                np.array((low_h, self._config.min_saturation, self._config.min_value), dtype=np.uint8),
                np.array((high_h, 255, 255), dtype=np.uint8),
            )
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def _evidence(self, hsv: np.ndarray, ranges: tuple[tuple[int, int], ...]) -> float:
        mask = self._mask(hsv, ranges)
        # Weight coloured pixels by brightness, so a lit lens wins over a dull
        # lens of the same hue.
        return float(np.sum(hsv[:, :, 2][mask > 0]) / 255.0)

    @staticmethod
    def _valid_roi(frame: np.ndarray, roi: Optional[tuple[int, int, int, int]]) -> Optional[tuple[int, int, int, int]]:
        if roi is None:
            return None
        x, y, width, height = roi
        x = max(0, int(x))
        y = max(0, int(y))
        width = min(int(width), frame.shape[1] - x)
        height = min(int(height), frame.shape[0] - y)
        return (x, y, width, height) if width > 0 and height > 0 else None

    def locate_roi(self, frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """Find one likely lit lamp in the upper scene, then pad its bounding box."""
        search_height = max(1, int(frame.shape[0] * self._config.search_top_fraction))
        region = frame[:search_height]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for ranges in self._RANGES.values():
            combined |= self._mask(hsv, ranges)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(4.0, frame.shape[0] * frame.shape[1] * 0.000002)
        candidates = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) >= min_area]
        if not candidates:
            return None
        x, y, width, height = min(candidates, key=lambda box: box[1])
        pad = max(8, int(max(width, height) * 2.5))
        return self._valid_roi(frame, (x - pad, y - pad, width + 2 * pad, height + 2 * pad))
