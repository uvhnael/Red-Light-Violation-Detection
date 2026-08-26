"""OCR biển số bằng fast-plate-ocr (ONNX, không cần huấn luyện GPU).

Cài đặt giao thức ``PlateRecognizer`` trong ``edge_node.core.contracts``.

Cài đặt
-------
    pip install fast-plate-ocr
    # tăng tốc GPU (tuỳ chọn):
    pip install onnxruntime-gpu

Thư viện có sẵn các model:
- ``global-plates-mobile-vit-v2-model`` — tổng quát nhất (mặc định)
- ``european-plates-mobile-vit-v2-model`` — biển EU
- ``argentinian-plates-cnn-model`` — biển Argentina

Model global đọc tốt biển số Việt Nam; đầu ra được lọc qua
``edge_node.core.vn_plate`` để chỉ nhận biển đúng cấu trúc VN.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import cv2
import numpy as np

from edge_node.core.contracts import PlateObservation, PlateRecognizer, Track
from edge_node.core.vn_plate import validate_and_format

LOGGER = logging.getLogger(__name__)

_DEFAULT_MODEL = "global-plates-mobile-vit-v2-model"

# Chỉ giữ chữ, số, gạch và chấm — phần còn lại là nhiễu
_PLATE_CLEAN_RE = re.compile(r"[^A-Z0-9.-]")


def resolve_ocr_device(requested: str) -> str:
    """Chọn execution provider cho ONNX Runtime.

    ``"auto"`` ưu tiên CUDA (khi có onnxruntime-gpu / CUDA EP),
    ngược lại dùng CPU.
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
    """Chuẩn hoá text thô từ OCR thành dạng gần-canonical.

    Ví dụ: ``29h-123.45`` → ``29H-123.45``, ``301 2345`` → ``3012345``.
    """
    cleaned = _PLATE_CLEAN_RE.sub("", text.upper().strip())
    return re.sub(r"[.]+", ".", cleaned)


class FastPlateOCR:
    """PlateRecognizer dựa trên ONNX runtime của fast-plate-ocr."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "auto",
        pad_to: int = 8,
    ) -> None:
        """Khởi tạo engine OCR.

        Tham số
        -------
        model_name : một trong các model ID của fast-plate-ocr.
        device : ``"auto"`` (CUDA nếu có, ngược lại CPU), ``"cuda"``, ``"cpu"``.
        pad_to : số ký tự tối thiểu khi pad output (biển VN: 8).
        """
        self._model_name = model_name
        self._device = resolve_ocr_device(device)
        self._pad_to = pad_to
        self._engine = None  # nạp trễ

    # ------------------------------------------------------------------
    # Giao thức PlateRecognizer
    # ------------------------------------------------------------------
    def recognize(self, frame: np.ndarray, track: Track) -> Optional[PlateObservation]:
        """Đọc biển số từ vùng phương tiện xác định bởi *track.bbox*.

        Trả về ``None`` khi không tìm thấy biển đọc được.
        """
        return self.recognize_bbox(frame, track.bbox, ref_id=track.track_id)

    def recognize_bbox(
        self,
        frame: np.ndarray,
        bbox,
        ref_id: int | str = -1,
    ) -> Optional[PlateObservation]:
        """Đọc biển số từ một bounding box tuỳ ý.

        Tham số
        -------
        frame : ảnh BGR.
        bbox : đối tượng có thuộc tính ``x1, y1, x2, y2`` (``BoundingBox``).
        ref_id : id tham chiếu, chỉ dùng cho log debug.

        Trả về ``None`` khi không đọc được biển hợp lệ.
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

        # Lỗi fast-plate-ocr 1.1.0: resize_image() KHÔNG convert BGR→gray
        # khi keep_aspect_ratio=False, khiến input 3 kênh làm crash phiên
        # ONNX ("Got: 3 Expected: 1"). Convert sang grayscale tại đây.
        if crop.ndim == 3 and crop.shape[2] == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        try:
            engine = self._get_engine()
            result = engine.run_one(crop, return_confidence=True, remove_pad_char=True)

            raw_text = normalize_plate_text(result.plate)
            if not raw_text:
                return None

            # Bộ lọc cấu trúc biển VN: 2 số tỉnh (11-99) + seri
            # (2 chữ | 1 chữ + 1 số | 1 chữ) + 4-5 số, tổng 8-9 ký tự.
            # Chuỗi không khớp cấu trúc (trừ khi sửa được lỗi OCR phổ biến)
            # sẽ bị loại bỏ ngay tại đây.
            validation = validate_and_format(raw_text)
            if not validation.valid:
                LOGGER.debug(
                    "OCR rejected non-VN plate %r for region %s (%s)",
                    raw_text, ref_id, validation.reason,
                )
                return None
            text = validation.formatted or raw_text

            # Confidence trung bình theo ký tự (bỏ qua ký tự pad)
            char_confidence = 0.0
            if result.char_probs is not None and len(result.char_probs) > 0:
                char_confidence = float(np.mean(result.char_probs[:len(raw_text)]))

            return PlateObservation(
                text=text,
                confidence=round(char_confidence, 4),
                metadata={
                    "model": self._model_name,
                    "device": self._device,
                    "region": result.region,
                    "region_prob": result.region_prob,
                    "raw_ocr": raw_text,
                    "repaired": validation.reason == "repaired",
                },
            )
        except Exception as exc:
            LOGGER.debug("OCR failed for region %s: %s", ref_id, exc)
            return None

    # ------------------------------------------------------------------
    # Nạp engine trễ
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
