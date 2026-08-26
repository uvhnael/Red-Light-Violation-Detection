"""Traffic-light colour classification with a fine-tuned YOLO26-nano classifier.

Primary light-state source for the pipeline, replacing the legacy HSV-mask
classifier (``traffic_light_cv.py``, kept as fallback and ROI finder).  The
model is a YOLO26n-cls fine-tuned on the LISA cropped traffic-light dataset
merged into three colour classes (red / yellow / green) — training scripts
live under ``train_traffic_light/``.

The classifier expects a crop of the lamp region, so the ROI plumbing is the
same as before: an ROI set via the control-plane API (web re-calibration)
wins, then the instance ROI from auto-calibration, then a best-effort HSV
blob search (reused from the legacy module) locates a lit lamp.

Use ``create_light_classifier`` to build the right classifier: it returns the
YOLO classifier when the weights file exists and falls back to the pure-OpenCV
HSV classifier otherwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from edge_node.core.contracts import Frame, LightObservation, LightState
from edge_node.core.traffic_light_cv import OpenCVTrafficLightClassifier

LOGGER = logging.getLogger(__name__)

# Default weights location relative to the project root.
_DEFAULT_MODEL_PATH = "edge_node/models/traffic_light_cls.pt"

# Trained class name -> pipeline state (map by name, never by index).
_NAME_TO_STATE = {
    "red": LightState.RED,
    "yellow": LightState.YELLOW,
    "green": LightState.GREEN,
}


@dataclass(frozen=True)
class FusionConfig:
    """Tunables for the YOLO + HSV + lamp-position fusion.

    The YOLO classifier is trained on US (LISA) lights and suffers domain
    shift on Vietnamese lights — it can report RED at 0.95+ confidence while
    the green lamp is clearly lit.  The fusion therefore cross-checks the
    YOLO vote against two domain-independent physical signals measured on the
    same crop:

    * HSV colour evidence of the lit lamp (with thresholds tuned for dimmer
      green lamps, which the legacy ``min_value=120`` default misses).
    * Vertical position of the lit lamp inside the crop: on a standard
      vertical 3-lamp head the red lamp sits at the top and the green lamp at
      the bottom, so the centroid row of the bright saturated pixels is a
      strong, appearance-independent cue.
    """

    enabled: bool = True
    # HSV thresholds for "lit lamp" pixels.  Measured on real crops: the red
    # lamp saturates at V~235-253 while the green lamp is dimmer (V~141-180),
    # so the value floor must sit well below the legacy 120 default.
    min_saturation: int = 50
    min_value: int = 100
    # Hue ranges.  Red wraps around 180; the legacy (0,170) range was far too
    # wide and swallowed green hues, producing 0.50/0.50 ties.
    red_ranges: tuple[tuple[int, int], ...] = ((0, 12), (168, 180))
    yellow_ranges: tuple[tuple[int, int], ...] = ((15, 40),)
    green_ranges: tuple[tuple[int, int], ...] = ((40, 95),)
    # Minimum summed brightness evidence for HSV to be considered "clear".
    min_evidence: float = 300.0
    # Dominance of the winning hue over total colour evidence.
    min_dominance: float = 0.60
    # Minimum lit-pixel count for the position cue to be trusted.
    min_position_pixels: int = 6
    # Normalised centroid-row bands: above -> red lamp, below -> green lamp.
    position_red_max: float = 0.40
    position_green_min: float = 0.60
    # Confidence assigned per fusion branch (colour+position agree, colour
    # only, position only).  Each gets a small boost when YOLO concurs.
    conf_colour_and_position: float = 0.90
    conf_colour_only: float = 0.75
    conf_position_only: float = 0.65
    conf_yolo_agrees_bonus: float = 0.05
    conf_yolo_agrees_position_bonus: float = 0.15


def _project_root() -> Path:
    # edge_node/core/ -> project root
    return Path(__file__).resolve().parents[2]


def _hue_evidence(crop: np.ndarray, config: FusionConfig) -> dict[LightState, float]:
    """Summed brightness of lit lamp pixels per hue band (domain-independent)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    ranges = {
        LightState.RED: config.red_ranges,
        LightState.YELLOW: config.yellow_ranges,
        LightState.GREEN: config.green_ranges,
    }
    evidence: dict[LightState, float] = {}
    for state, bands in ranges.items():
        mask = np.zeros(crop.shape[:2], dtype=bool)
        for low, high in bands:
            mask |= (
                (hue >= low)
                & (hue <= high)
                & (sat >= config.min_saturation)
                & (val >= config.min_value)
            )
        evidence[state] = float(val[mask].sum()) if mask.any() else 0.0
    return evidence


def _lamp_position(crop: np.ndarray, config: FusionConfig) -> Optional[LightState]:
    """Vertical centroid of the lit lamp: top -> red, bottom -> green.

    On a standard vertical 3-lamp head the physical layout is fixed, so this cue
    does not depend on the lamp's appearance or the training domain.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    lit = (val > config.min_value) & (sat > config.min_saturation)
    if int(lit.sum()) < config.min_position_pixels:
        return None
    centroid_row = float(np.where(lit)[0].mean()) / crop.shape[0]
    if centroid_row < config.position_red_max:
        return LightState.RED
    if centroid_row > config.position_green_min:
        return LightState.GREEN
    return LightState.YELLOW


def _fuse_observation(
    yolo_state: LightState,
    yolo_confidence: float,
    crop: np.ndarray,
    config: FusionConfig,
) -> LightObservation:
    """Cross-check the YOLO vote against HSV colour + lamp position.

    Priority: colour+position agreement (strongest physical signal) > colour
    alone > position alone > raw YOLO.  YOLO concurrence adds a small
    confidence bonus; disagreement never vetoes a clear physical signal.
    """
    if crop is None or crop.size == 0 or crop.ndim != 3:
        return LightObservation(yolo_state, yolo_confidence, source="yolo-cls")

    evidence = _hue_evidence(crop, config)
    total = sum(evidence.values())
    colour_state, colour_score = max(evidence.items(), key=lambda item: item[1])
    dominance = colour_score / total if total else 0.0
    colour_clear = (
        total > 0.0
        and colour_score >= config.min_evidence
        and dominance >= config.min_dominance
    )
    position_state = _lamp_position(crop, config)

    if colour_clear and position_state == colour_state:
        confidence = config.conf_colour_and_position
        if yolo_state == colour_state:
            confidence += config.conf_yolo_agrees_bonus
        return LightObservation(
            colour_state,
            min(1.0, confidence),
            source="yolo-cls+fused",
            metadata={"yolo": yolo_state.value, "yolo_conf": round(yolo_confidence, 3)},
        )
    if colour_clear:
        confidence = config.conf_colour_only
        if yolo_state == colour_state:
            confidence += config.conf_yolo_agrees_bonus
        return LightObservation(
            colour_state,
            min(1.0, confidence),
            source="yolo-cls+fused",
            metadata={"yolo": yolo_state.value, "yolo_conf": round(yolo_confidence, 3)},
        )
    if position_state is not None:
        confidence = config.conf_position_only
        if yolo_state == position_state:
            confidence += config.conf_yolo_agrees_position_bonus
        return LightObservation(
            position_state,
            min(1.0, confidence),
            source="yolo-cls+fused",
            metadata={"yolo": yolo_state.value, "yolo_conf": round(yolo_confidence, 3)},
        )
    return LightObservation(yolo_state, yolo_confidence, source="yolo-cls")


def _resolve_model_path(model_path: Optional[str]) -> Optional[Path]:
    """Resolve *model_path* (or the default) to an existing file, if any."""
    candidates: list[Path] = []
    if model_path:
        candidates.append(Path(model_path))
        candidates.append(_project_root() / model_path)
    candidates.append(_project_root() / _DEFAULT_MODEL_PATH)
    for candidate in candidates:
        if candidate.is_file():
            if model_path and candidate != Path(model_path) and candidate != _project_root() / model_path:
                LOGGER.warning(
                    "Traffic-light weights not found at %s — using default %s",
                    model_path, candidate,
                )
            return candidate
    return None


class YoloTrafficLightClassifier:
    """Classify the lamp colour of a traffic-light crop with YOLO26n-cls.

    ``roi`` uses ``(x, y, width, height)`` pixel coordinates.  When omitted,
    the first sufficiently bright coloured blob in the upper part of the frame
    is used as a best-effort ROI and then retained for subsequent frames
    (same behaviour as the legacy OpenCV classifier).
    """

    def __init__(
        self,
        model_path: str | Path,
        roi: Optional[tuple[int, int, int, int]] = None,
        img_size: int = 64,
        device: Optional[str] = None,
        fusion: FusionConfig = FusionConfig(),
    ) -> None:
        self._model_path = Path(model_path)
        self._roi = roi
        self._img_size = img_size
        self._fusion = fusion
        self._model = None  # lazy
        # Device resolution mirrors YoloDetector: CUDA when available.
        from edge_node.core.detector import resolve_device

        self._device = resolve_device(device)
        # Legacy OpenCV classifier kept only for its static _valid_roi helper;
        # HSV ROI discovery is disabled — the operator draws the ROI on the
        # web UI instead.
        self._roi_finder = OpenCVTrafficLightClassifier(roi=roi)

    @property
    def roi(self) -> Optional[tuple[int, int, int, int]]:
        return self._roi

    def classify(self, frame: Frame, frame_index: int, timestamp_ms: float) -> LightObservation:
        del frame_index, timestamp_ms
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return LightObservation(LightState.UNKNOWN, 0.0, source="yolo-cls")

        # A ROI set via the control-plane API (web re-calibration) wins over
        # the instance ROI so operators can correct detection at runtime.
        from edge_node.core.config import get_active_light_roi

        valid = OpenCVTrafficLightClassifier._valid_roi
        roi = valid(frame, get_active_light_roi()) or valid(frame, self._roi)
        if roi is None:
            # No operator-drawn ROI: do NOT guess one via HSV blob search.
            # Until the operator calibrates the light box on the web UI there
            # is no trustworthy lamp region, and a guessed ROI risks locking
            # onto vehicle tail-lights or signs — report UNKNOWN instead.
            return LightObservation(LightState.UNKNOWN, 0.0, source="yolo-cls")

        x, y, width, height = roi
        return self._classify_crop(frame[y : y + height, x : x + width])

    def _classify_crop(self, crop: np.ndarray) -> LightObservation:
        if crop is None or crop.size == 0 or crop.ndim != 3:
            return LightObservation(LightState.UNKNOWN, 0.0, source="yolo-cls")
        model = self._get_model()
        results = model.predict(
            source=crop,
            imgsz=self._img_size,
            verbose=False,
            device=self._device,
        )
        probs = results[0].probs if results else None
        if probs is None:
            return LightObservation(LightState.UNKNOWN, 0.0, source="yolo-cls")
        name = str(model.names.get(int(probs.top1), "")).lower()
        state = _NAME_TO_STATE.get(name, LightState.UNKNOWN)
        confidence = float(probs.top1conf)
        if not self._fusion.enabled:
            return LightObservation(state, confidence, source="yolo-cls")
        return _fuse_observation(state, confidence, crop, self._fusion)

    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLO

            LOGGER.info("Loading traffic-light classifier from %s", self._model_path)
            self._model = YOLO(str(self._model_path))
            if self._device != "cpu":
                self._model.to(self._device)
            LOGGER.info(
                "Traffic-light classifier loaded (device=%s, classes=%s)",
                self._device, getattr(self._model, "names", {}),
            )
        return self._model


def create_light_classifier(
    roi: Optional[tuple[int, int, int, int]] = None,
    model_path: Optional[str] = None,
    img_size: int = 64,
    device: Optional[str] = None,
    fusion: Optional[FusionConfig] = None,
):
    """Build the best available traffic-light classifier.

    Returns ``YoloTrafficLightClassifier`` when the fine-tuned weights exist,
    otherwise falls back to the legacy ``OpenCVTrafficLightClassifier``.
    """
    path = _resolve_model_path(model_path)
    if path is not None:
        LOGGER.info("Traffic-light classifier: YOLO26n-cls (%s)", path)
        return YoloTrafficLightClassifier(
            path,
            roi=roi,
            img_size=img_size,
            device=device,
            fusion=fusion or FusionConfig(),
        )
    LOGGER.warning(
        "Traffic-light weights not found (%s) — falling back to OpenCV HSV classifier",
        model_path or _DEFAULT_MODEL_PATH,
    )
    return OpenCVTrafficLightClassifier(roi=roi)
