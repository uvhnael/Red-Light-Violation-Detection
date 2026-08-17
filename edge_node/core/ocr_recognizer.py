"""License-plate OCR using fast-plate-ocr (ONNX-based, no GPU training needed).

Implements the ``PlateRecognizer`` protocol from ``edge_node.core.contracts``.

Install
-------
    pip install fast-plate-ocr
    # optional GPU acceleration:
    pip install onnxruntime-gpu

The library ships with:
- ``global-plates-mobile-vit-v2-model`` — best general-purpose model
- ``european-plates-mobile-vit-v2-model`` — EU plates
- ``argentinian-plates-cnn-model`` — AR plates

Default is the global model; Vietnamese plates work well with it.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import cv2
import numpy as np

from edge_node.core.contracts import PlateObservation, PlateRecognizer, Track

LOGGER = logging.getLogger(__name__)

_DEFAULT_MODEL = "global-plates-mobile-vit-v2-model"

# Keep letters, digits, dash and dot — the rest of VN plates is noise
_PLATE_CLEAN_RE = re.compile(r"[^A-Z0-9.-]")


def resolve_ocr_device(requested: str) -> str:
    """Pick the ONNX Runtime execution provider.

    ``"auto"`` prefers CUDA (when onnxruntime-gpu / CUDA EP is present),
    otherwise falls back to CPU.
    """
    if requested and requested != "auto":
        return requested
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            LOGGER.info("OCR: CUDAExecutionProvider available — using GPU")
            return "cuda"
    except Exception as exc:
        LOGGER.debug("OCR CUDA detection failed: %s", exc)
    LOGGER.info("OCR: CUDA not available — using CPU")
    return "cpu"


def normalize_plate_text(text: str) -> str:
    """Normalise raw OCR output into a canonical Vietnamese plate format.

    Examples: ``29H-123.45`` → ``29H-123.45``, ``301 2345`` → ``3012345``.
    """
    cleaned = _PLATE_CLEAN_RE.sub("", text.upper().strip())
    return re.sub(r"[.]+", ".", cleaned)


class FastPlateOCR:
    """PlateRecognizer backed by fast-plate-ocr ONNX runtime."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "auto",
        pad_to: int = 8,
    ) -> None:
        """Initialise the OCR engine.

        Parameters
        ----------
        model_name : one of the fast-plate-ocr model IDs.
        device : ``"auto"`` (CUDA if available, else CPU), ``"cuda"``, or ``"cpu"``.
        pad_to : minimum character count to pad output to (VN plates: 8).
        """
        self._model_name = model_name
        self._device = resolve_ocr_device(device)
        self._pad_to = pad_to
        self._engine = None  # lazy

    # ------------------------------------------------------------------
    # PlateRecognizer protocol
    # ------------------------------------------------------------------
    def recognize(self, frame: np.ndarray, track: Track) -> Optional[PlateObservation]:
        """Extract plate text from the vehicle region defined by *track.bbox*.

        Returns ``None`` when no readable plate is found.
        """
        return self.recognize_bbox(frame, track.bbox, ref_id=track.track_id)

    def recognize_bbox(
        self,
        frame: np.ndarray,
        bbox,
        ref_id: int | str = -1,
    ) -> Optional[PlateObservation]:
        """Extract plate text from an arbitrary bounding box.

        Parameters
        ----------
        frame : BGR image.
        bbox : object with ``x1, y1, x2, y2`` attributes (``BoundingBox``).
        ref_id : optional reference id used only for debug logging.

        Returns ``None`` when no readable plate is found.
        """
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(frame.shape[1], int(bbox.x2))
        y2 = min(frame.shape[0], int(bbox.y2))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # fast-plate-ocr 1.1.0 bug: resize_image() does NOT convert BGR→gray
        # when keep_aspect_ratio=False, so 3-channel input crashes the ONNX
        # session ("Got: 3 Expected: 1"). Convert to grayscale here.
        if crop.ndim == 3 and crop.shape[2] == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        try:
            engine = self._get_engine()
            result = engine.run_one(crop, return_confidence=True, remove_pad_char=True)

            text = normalize_plate_text(result.plate)
            if not text:
                return None

            # Average character confidence (omit padding chars)
            char_confidence = 0.0
            if result.char_probs is not None and len(result.char_probs) > 0:
                char_confidence = float(np.mean(result.char_probs[:len(text)]))

            return PlateObservation(
                text=text,
                confidence=round(char_confidence, 4),
                metadata={
                    "model": self._model_name,
                    "device": self._device,
                    "region": result.region,
                    "region_prob": result.region_prob,
                },
            )
        except Exception as exc:
            LOGGER.debug("OCR failed for region %s: %s", ref_id, exc)
            return None

    # ------------------------------------------------------------------
    # Lazy engine loading
    # ------------------------------------------------------------------
    def _get_engine(self):
        if self._engine is None:
            from fast_plate_ocr import LicensePlateRecognizer

            LOGGER.info(
                "Loading fast-plate-ocr model '%s' (device=%s)",
                self._model_name, self._device,
            )
            self._engine = LicensePlateRecognizer(
                hub_ocr_model=self._model_name,
                device=self._device,
            )
            LOGGER.info("fast-plate-ocr engine ready")
        return self._engine
