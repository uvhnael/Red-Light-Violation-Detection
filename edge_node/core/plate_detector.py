"""License-plate detector backed by the fine-tuned YOLO26 plate model.

Detects two classes trained on Vietnamese plates:

* ``BSD`` — biển số dọc (vertical / two-line plates, trucks & some cars)
* ``BSV`` — biển số ngang (horizontal plates, cars & motorbikes)

Default weights live at ``edge_node/models/license_plate_yolo26.pt``
(copied from ``runs/detect/runs/train/license_plate_yolo26/weights/best.pt``).

Usage
-----
    from edge_node.core.plate_detector import PlateDetector

    detector = PlateDetector()                      # auto device + FP16
    detections = detector.detect(frame, 0, 0.0)     # -> [Detection, ...]

The class reuses :class:`edge_node.core.detector.YoloDetector` (lazy load,
CUDA auto-detect, FP16 half precision) with ``vehicle_only=False`` so the
plate class names from the model checkpoint are kept as-is.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from edge_node.core.detector import YoloDetector

LOGGER = logging.getLogger(__name__)

# Plate model candidates — prefer the fine-tuned YOLO26 plate weights
_PLATE_MODEL_CANDIDATES: tuple[str, ...] = (
    "edge_node/models/license_plate_yolo26.pt",
)


def _resolve_plate_model() -> str:
    """Return the first available plate model from the project root."""
    root = Path(__file__).resolve().parents[2]  # edge_node/core/ → project root
    for candidate in _PLATE_MODEL_CANDIDATES:
        full = root / candidate
        if full.exists():
            return str(full)
    raise FileNotFoundError(
        "No license-plate model found. Expected one of: "
        + ", ".join(_PLATE_MODEL_CANDIDATES)
        + " — copy the trained best.pt to edge_node/models/license_plate_yolo26.pt"
    )


class PlateDetector(YoloDetector):
    """ObjectDetector specialised for license-plate detection (BSD/BSV).

    Lower default confidence than the vehicle detector because plates are
    small objects and the fine-tuned model is well calibrated (val mAP50
    ≈ 0.99).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence: float = 0.25,
        iou_nms: float = 0.45,
        img_size: int = 640,
        device: Optional[str] = None,
        fp16: Optional[bool] = None,
    ) -> None:
        if model_path is None:
            model_path = _resolve_plate_model()
        super().__init__(
            model_path=model_path,
            confidence=confidence,
            iou_nms=iou_nms,
            img_size=img_size,
            vehicle_only=False,  # keep BSD/BSV labels from the checkpoint
            device=device,
            fp16=fp16,
        )
        LOGGER.info("PlateDetector configured with model %s", model_path)
