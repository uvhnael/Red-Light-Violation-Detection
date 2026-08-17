"""YOLO object detector with ONNX / TensorRT support.

Usage
-----
* Development : ``YoloDetector('models/yolo26m.pt')``
* Edge (ONNX) : ``YoloDetector('models/yolo26m.onnx')``
* Edge (TRT)  : ``YoloDetector('models/yolo26m.engine')``

The device defaults to CUDA when available and automatically enables
FP16 half-precision inference on GPU for ``.pt`` models.

To export a model to optimised formats::

    YoloDetector.export_onnx('models/yolo26m.pt')   # -> models/yolo26m.onnx
    YoloDetector.export_tensorrt('models/yolo26m.pt')  # -> models/yolo26m.engine
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

from edge_node.core.contracts import BoundingBox, Detection, Frame

LOGGER = logging.getLogger(__name__)

# COCO class-id -> human label for common vehicles & traffic objects
_VEHICLE_CLASS_MAP: dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",       # unlikely but harmless to track
    5: "bus",
    6: "train",           # tram / light rail in street scenes
    7: "truck",
    8: "boat",            # unlikely but harmless
    9: "traffic_light",   # YOLO26 detects traffic lights natively
}

# Model candidates — prefer YOLO26, fall back to YOLOv8, then Ultralytics hub
_MODEL_CANDIDATES: tuple[str, ...] = (
    "edge_node/models/yolo26m.pt",
    "edge_node/models/yolo26l.pt",
    "edge_node/models/yolo26s.pt",
    "edge_node/models/yolo11s.pt",
    "edge_node/models/yolov8s.pt",
)

# Inference backends that run without PyTorch (no FP16 via .half())
_NON_TORCH_SUFFIXES = (".onnx", ".engine", ".tflite", ".mlpackage")


def _resolve_default_model() -> str:
    """Return the first available YOLO26 or YOLOv8 model from the project root."""
    from pathlib import Path
    # Walk up to project root from this file's location
    root = Path(__file__).resolve().parents[2]  # edge_node/core/ → project root
    for candidate in _MODEL_CANDIDATES:
        full = root / candidate
        if full.exists():
            return str(full)
    return "yolov8s.pt"  # fallback (download from Ultralytics hub)


def resolve_device(requested: Optional[str] = None) -> str:
    """Pick the inference device.

    Returns *requested* when set, otherwise ``"cuda"`` if a CUDA GPU is
    available and ``"cpu"`` as the last resort.
    """
    if requested:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            LOGGER.info(
                "CUDA GPU detected: %s — using GPU inference",
                torch.cuda.get_device_name(0),
            )
            return "cuda"
    except Exception as exc:  # torch missing / broken
        LOGGER.debug("CUDA detection failed: %s", exc)
    LOGGER.info("No CUDA GPU found — falling back to CPU inference")
    return "cpu"


class YoloDetector:
    """ObjectDetector backed by Ultralytics YOLO.

    Accepts ``.pt``, ``.onnx``, or ``.engine`` (TensorRT) weights.
    The model is loaded lazily on the first ``detect`` call so import
    time stays low.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence: float = 0.35,
        iou_nms: float = 0.45,
        img_size: int = 640,
        vehicle_only: bool = True,
        device: str | None = None,
        fp16: bool | None = None,
    ) -> None:
        self._model_path = Path(model_path) if model_path else Path(_resolve_default_model())
        self._confidence = confidence
        self._iou_nms = iou_nms
        self._img_size = img_size
        self._vehicle_only = vehicle_only
        # Auto device: CUDA when available, otherwise CPU
        self._device = resolve_device(device)
        self._fp16 = fp16
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
        class_names = getattr(model, "names", None) or {}
        detections: list[Detection] = []
        for result in results:
            names = getattr(result, "names", None) or class_names or {}
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if self._vehicle_only and cls_id not in _VEHICLE_CLASS_MAP:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                label = names.get(cls_id) or _VEHICLE_CLASS_MAP.get(cls_id, f"class_{cls_id}")
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
            if self._device != "cpu":
                self._model.to(self._device)
            # FP16 half-precision: auto-enable on CUDA unless explicitly set
            is_torch_model = not str(self._model_path).lower().endswith(_NON_TORCH_SUFFIXES)
            if self._fp16 is None:
                self._fp16 = self._device != "cpu" and is_torch_model
            if self._fp16:
                if is_torch_model:
                    self._model.model.half()
                    LOGGER.info(
                        "FP16 half-precision enabled for GPU inference (device=%s)",
                        self._device,
                    )
                else:
                    LOGGER.warning(
                        "FP16 requested but model format does not support .half() "
                        "— exporting with half=True is recommended for ONNX/TensorRT"
                    )
                    self._fp16 = False
            LOGGER.info(
                "YOLO model loaded successfully (device=%s, fp16=%s, classes=%s)",
                self._device, self._fp16, len(getattr(self._model, "names", {})),
            )
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
