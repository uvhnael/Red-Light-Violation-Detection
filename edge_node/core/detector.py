"""YOLO object detector with ONNX / TensorRT support.

Usage
-----
* Development : ``YoloDetector('models/yolov8s.pt')``
* Edge (ONNX) : ``YoloDetector('models/yolov8s.onnx')``
* Edge (TRT)  : ``YoloDetector('models/yolov8s.engine')``

To export a model to optimised formats::

    YoloDetector.export_onnx('models/yolov8s.pt')   # -> models/yolov8s.onnx
    YoloDetector.export_tensorrt('models/yolov8s.pt')  # -> models/yolov8s.engine
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from edge_node.core.contracts import BoundingBox, Detection, Frame

LOGGER = logging.getLogger(__name__)

# COCO class-id -> human label for common vehicles
_VEHICLE_CLASS_MAP: dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class YoloDetector:
    """ObjectDetector backed by Ultralytics YOLO.

    Accepts ``.pt``, ``.onnx``, or ``.engine`` (TensorRT) weights.
    The model is loaded lazily on the first ``detect`` call so import
    time stays low.
    """

    def __init__(
        self,
        model_path: str | Path = "models/yolov8s.pt",
        confidence: float = 0.35,
        iou_nms: float = 0.45,
        img_size: int = 640,
        vehicle_only: bool = True,
        device: str | None = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._confidence = confidence
        self._iou_nms = iou_nms
        self._img_size = img_size
        self._vehicle_only = vehicle_only
        self._device = device
        self._model = None  # lazy

    # ------------------------------------------------------------------
    # ObjectDetector protocol
    # ------------------------------------------------------------------
    def detect(
        self, frame: Frame, frame_index: int, timestamp_ms: float,
    ) -> Sequence[Detection]:
        del frame_index, timestamp_ms
        model = self._get_model()
        results = model.predict(
            source=frame,
            conf=self._confidence,
            iou=self._iou_nms,
            imgsz=self._img_size,
            verbose=False,
            device=self._device,
        )
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if self._vehicle_only and cls_id not in _VEHICLE_CLASS_MAP:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                label = _VEHICLE_CLASS_MAP.get(cls_id, f"class_{cls_id}")
                detections.append(
                    Detection(
                        bbox=BoundingBox(x1, y1, x2, y2),
                        label=label,
                        confidence=conf,
                    )
                )
        return detections

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLO

            LOGGER.info("Loading YOLO model from %s", self._model_path)
            self._model = YOLO(str(self._model_path))
            if self._device:
                self._model.to(self._device)
            LOGGER.info("YOLO model loaded successfully")
        return self._model

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    @staticmethod
    def export_onnx(
        model_path: str | Path,
        img_size: int = 640,
        simplify: bool = True,
    ) -> Path:
        """Export a .pt model to ONNX format."""
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        export_path = model.export(format="onnx", imgsz=img_size, simplify=simplify)
        LOGGER.info("Exported ONNX model to %s", export_path)
        return Path(export_path)

    @staticmethod
    def export_tensorrt(
        model_path: str | Path,
        img_size: int = 640,
        half: bool = True,
    ) -> Path:
        """Export a .pt model to TensorRT engine."""
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        export_path = model.export(format="engine", imgsz=img_size, half=half)
        LOGGER.info("Exported TensorRT engine to %s", export_path)
        return Path(export_path)
