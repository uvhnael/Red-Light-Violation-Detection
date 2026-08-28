"""Trình phát hiện phương tiện bằng YOLO (hỗ trợ ONNX / TensorRT).

Cách dùng
---------
* Dev        : ``YoloDetector('models/yolo26m.pt')``
* Edge (ONNX): ``YoloDetector('models/yolo26m.onnx')``
* Edge (TRT) : ``YoloDetector('models/yolo26m.engine')``

Thiết bị mặc định là CUDA nếu có, tự bật suy luận FP16 trên GPU cho model
``.pt``. Xuất model sang định dạng tối ưu::

    YoloDetector.export_onnx('models/yolo26m.pt')      # -> .onnx
    YoloDetector.export_tensorrt('models/yolo26m.pt')  # -> .engine

Hỗ trợ hai loại model:
* Model COCO mặc định (80 lớp) — lọc theo bảng map id -> nhãn bên dưới.
* Model fine-tuned dataset riêng (car/bike/van/bus/truck) — nhận nguyên
  bảng tên của model, không remap qua id COCO.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

from edge_node.core.contracts import BoundingBox, Detection, Frame

LOGGER = logging.getLogger(__name__)

# Bảng map class-id COCO -> nhãn hiển thị cho các phương tiện thường gặp
_VEHICLE_CLASS_MAP: dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",       # hiếm gặp nhưng vô hại nếu track
    5: "bus",
    6: "train",          # tàu điện trong cảnh đường phố
    7: "truck",
    8: "boat",           # hiếm gặp nhưng vô hại
    9: "traffic_light",  # YOLO26 nhận diện sẵn đèn giao thông
}

# Bộ nhãn của model fine-tuned trên dataset riêng. Khi model expose đúng
# các tên này thì id KHÔNG được map lại qua bảng COCO ở trên (không gian
# id của fine-tuned bắt đầu từ 0 và khác hoàn toàn).
_DATASET_VEHICLE_CLASSES: frozenset[str] = frozenset(
    {"car", "bike", "van/bus", "truck"}
)

# Thứ tự ưu tiên model khi không chỉ định đường dẫn
_MODEL_CANDIDATES: tuple[str, ...] = (
    "models/yolo26m.pt",
    "models/yolo26l.pt",
    "models/yolo26s.pt",
)

# Các backend suy luận chạy không cần PyTorch (không dùng được .half())
_NON_TORCH_SUFFIXES = (".onnx", ".engine", ".tflite", ".mlpackage")


def _resolve_default_model() -> str:
    """Trả về đường dẫn model khả dụng đầu tiên tính từ gốc project."""
    root = Path(__file__).resolve().parents[2]  # edge_node/core/ → gốc project
    for candidate in _MODEL_CANDIDATES:
        full = root / candidate
        if full.exists():
            return str(full)
    return "yolo26m.pt"  # fallback (tự tải từ Ultralytics hub)


def resolve_device(requested: Optional[str] = None) -> str:
    """Chọn thiết bị suy luận.

    Trả về đúng *requested* nếu được chỉ định, ngược lại ``"cuda"`` khi có
    GPU và cuối cùng là ``"cpu"``.
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
    except Exception as exc:  # thiếu torch / torch hỏng
        LOGGER.debug("CUDA detection failed: %s", exc)
    LOGGER.info("No CUDA GPU found — falling back to CPU inference")
    return "cpu"


class YoloDetector:
    """ObjectDetector dựa trên Ultralytics YOLO.

    Chấp nhận weights ``.pt``, ``.onnx`` hoặc ``.engine`` (TensorRT).
    Model được nạp trễ (lazy) ở lần gọi ``detect`` đầu tiên để thời gian
    import giữ ở mức thấp.
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
        # Thiết bị tự động: CUDA khi có, ngược lại CPU
        self._device = resolve_device(device)
        self._fp16 = fp16
        self._model = None  # nạp trễ

    # ------------------------------------------------------------------
    # Giao thức ObjectDetector
    # ------------------------------------------------------------------
    def detect(
        self, frame: Frame, frame_index: int, timestamp_ms: float,
        confidence: Optional[float] = None,
    ) -> Sequence[Detection]:
        del frame_index, timestamp_ms
        model = self._get_model()
        results = model.predict(
            source=frame,
            conf=confidence if confidence is not None else self._confidence,
            iou=self._iou_nms,
            imgsz=self._img_size,
            verbose=False,
            device=self._device,
        )
        class_names = getattr(model, "names", None) or {}
        detections: list[Detection] = []
        for result in results:
            names = getattr(result, "names", None) or class_names or {}
            # Model fine-tuned (vd yolo26m_vehicle.pt) expose bộ tên lớp
            # dataset trực tiếp — giữ nguyên id. Model COCO tiếp tục dùng
            # bảng map id -> nhãn phía trên.
            is_finetuned = _DATASET_VEHICLE_CLASSES.issubset(
                {str(v) for v in names.values()}
            )
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if is_finetuned:
                    label = str(names.get(cls_id, f"class_{cls_id}"))
                    if label not in _DATASET_VEHICLE_CLASSES:
                        continue
                else:
                    if self._vehicle_only and cls_id not in _VEHICLE_CLASS_MAP:
                        continue
                    label = names.get(cls_id) or _VEHICLE_CLASS_MAP.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                detections.append(
                    Detection(
                        bbox=BoundingBox(x1, y1, x2, y2),
                        label=label,
                        confidence=conf,
                    )
                )
        return detections

    # ------------------------------------------------------------------
    # Nạp model trễ
    # ------------------------------------------------------------------
    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLO

            LOGGER.info("Loading YOLO model from %s", self._model_path)
            self._model = YOLO(str(self._model_path))
            if self._device != "cpu":
                self._model.to(self._device)
            # FP16: tự bật trên GPU trừ khi bị ghi đè tường minh
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
    # Xuất model sang định dạng tối ưu
    # ------------------------------------------------------------------
    @staticmethod
    def export_onnx(
        model_path: str | Path,
        img_size: int = 640,
        simplify: bool = True,
    ) -> Path:
        """Xuất model .pt sang ONNX."""
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
        """Xuất model .pt sang TensorRT engine."""
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        export_path = model.export(format="engine", imgsz=img_size, half=half)
        LOGGER.info("Exported TensorRT engine to %s", export_path)
        return Path(export_path)
