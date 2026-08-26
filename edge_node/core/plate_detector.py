"""Detector biển số dựa trên model YOLO26 fine-tuned.

Phát hiện hai lớp huấn luyện trên biển số Việt Nam:

* ``BSD`` — biển số dọc (hai dòng, xe tải & một số ô tô)
* ``BSV`` — biển số ngang (ô tô & xe máy)

Weights mặc định nằm tại ``edge_node/models/license_plate_yolo26.pt``
(sao chép từ ``runs/detect/runs/train/license_plate_yolo26/weights/best.pt``).

Cách dùng
---------
    from edge_node.core.plate_detector import PlateDetector

    detector = PlateDetector()                      # tự chọn device + FP16
    detections = detector.detect(frame, 0, 0.0)     # -> [Detection, ...]

Lớp này tái sử dụng :class:`edge_node.core.detector.YoloDetector` (nạp trễ,
tự dò CUDA, FP16 half precision) với ``vehicle_only=False`` để giữ nguyên
tên lớp từ checkpoint của model.
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
