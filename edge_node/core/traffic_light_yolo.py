"""Phân loại màu đèn giao thông bằng YOLO26n-cls fine-tuned.

Nguồn trạng thái đèn chính của pipeline. Model là YOLO26n-cls fine-tuned
trên dataset đèn cropped LISA gộp thành 3 lớp màu (red / yellow / green) —
script huấn luyện nằm ở ``train_model/traffic_light_cls/``.

Classifier cần ảnh crop vùng đèn, nên thứ tự ưu tiên ROI như sau:
1. ROI set qua control-plane API (operator kẻ trên web UI);
2. ROI của instance (từ CLI ``--light-roi``);
3. Không có ROI nào -> trả UNKNOWN, KHÔNG tự dò HSV.

Dùng ``create_light_classifier`` để tạo classifier: trả về YOLO classifier
khi có file weights, ngược lại fallback sang classifier OpenCV HSV.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from edge_node.core.contracts import Frame, LightObservation, LightState
from edge_node.core.traffic_light_cv import OpenCVTrafficLightClassifier

LOGGER = logging.getLogger(__name__)

# Vị trí weights mặc định tính từ gốc project.
_DEFAULT_MODEL_PATH = "edge_node/models/traffic_light_cls.pt"

# Map tên lớp sau training -> trạng thái pipeline (map theo tên, không theo index).
_NAME_TO_STATE = {
    "red": LightState.RED,
    "yellow": LightState.YELLOW,
    "green": LightState.GREEN,
}


@dataclass(frozen=True)
class FusionConfig:
    """Tham số tinh chỉnh cho fusion YOLO + HSV + vị trí đèn.

    Model YOLO huấn luyện trên đèn Mỹ (LISA) nên bị domain shift với đèn
    Việt Nam — có thể báo RED 0.95+ trong khi đèn xanh đang sáng rõ.
    Fusion vì vậy đối chiếu phiếu YOLO với hai tín hiệu vật lý độc lập
    miền dữ liệu, đo trên cùng một ảnh crop:

    * Bằng chứng màu HSV của đèn (ngưỡng tinh chỉnh cho đèn xanh mờ,
      mà ngưỡng cũ ``min_value=120`` bỏ sót).
    * Vị trí dọc của đèn sáng trong crop: trên cột đèn dọc 3 bóng chuẩn,
      đỏ nằm trên cùng và xanh dưới cùng, nên hàng centroid của các pixel
      bão hòa sáng là dấu hiệu mạnh, không phụ thuộc ngoại hình.
    """

    enabled: bool = True
    # Ngưỡng HSV cho pixel "đèn sáng". Đo trên crop thật: đèn đỏ bão hoà
    # V~235-253 còn đèn xanh mờ hơn (V~141-180), nên sàn V phải thấp hơn
    # nhiều so với mặc định cũ 120.
    min_saturation: int = 50
    min_value: int = 100
    # Dải hue. Đỏ wrap quanh 180; dải (0,170) cũ quá rộng, nuốt cả vùng
    # màu xanh gây tỉ lệ 0.50/0.50.
    red_ranges: tuple[tuple[int, int], ...] = ((0, 12), (168, 180))
    yellow_ranges: tuple[tuple[int, int], ...] = ((15, 40),)
    green_ranges: tuple[tuple[int, int], ...] = ((40, 95),)
    # Tổng bằng chứng độ sáng tối thiểu để HSV được coi là "rõ".
    min_evidence: float = 300.0
    # Tỷ lệ ưu thế của hue thắng trên tổng bằng chứng màu.
    min_dominance: float = 0.60
    # Số pixel sáng tối thiểu để tin vào dấu hiệu vị trí.
    min_position_pixels: int = 6
    # Dải hàng centroid chuẩn hoá: trên -> đèn đỏ, dưới -> đèn xanh.
    position_red_max: float = 0.40
    position_green_min: float = 0.60
    # Confidence gán cho từng nhánh fusion (màu+vị trí khớp, chỉ màu,
    # chỉ vị trí). Mỗi nhánh được cộng nhẹ khi YOLO đồng ý.
    conf_colour_and_position: float = 0.90
    conf_colour_only: float = 0.75
    conf_position_only: float = 0.65
    conf_yolo_agrees_bonus: float = 0.05
    conf_yolo_agrees_position_bonus: float = 0.15


def _project_root() -> Path:
    # edge_node/core/ -> gốc project
    return Path(__file__).resolve().parents[2]


def _hue_evidence(crop: np.ndarray, config: FusionConfig) -> dict[LightState, float]:
    """Tổng độ sáng của pixel đèn theo từng dải hue (độc lập miền dữ liệu)."""
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
    """Centroid dọc của đèn sáng: trên -> đỏ, dưới -> xanh.

    Trên cột đèn 3 bóng dọc chuẩn bố cục vật lý là cố định, nên dấu hiệu
    này không phụ thuộc ngoại hình đèn hay miền dữ liệu huấn luyện.
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
    """Đối chiếu phiếu YOLO với màu HSV + vị trí đèn.

    Độ ưu tiên: màu+vị trí khớp (tín hiệu vật lý mạnh nhất) > chỉ màu >
    chỉ vị trí > YOLO thô. YOLO đồng ý cộng thêm chút confidence; việc
    không đồng ý không bao giờ phủ quyết tín hiệu vật lý rõ ràng.
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
    """Tìm file weights tồn tại từ *model_path* hoặc vị trí mặc định."""
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
    """Phân loại màu đèn giao thông trên ảnh crop bằng YOLO26n-cls.

    ``roi`` dùng tọa độ pixel ``(x, y, width, height)``. Khi không có ROI
    (từ API hoặc CLI), classifier trả UNKNOWN — không tự dò HSV.
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
        self._model = None  # nạp trễ
        # Chọn thiết bị giống YoloDetector: CUDA khi có sẵn.
        from edge_node.core.detector import resolve_device

        self._device = resolve_device(device)
        # Classifier OpenCV cũ chỉ còn được giữ để dùng helper _valid_roi;
        # tính năng tự dò ROI bằng HSV đã bị loại — operator kẻ ROI trên web.
        self._roi_finder = OpenCVTrafficLightClassifier(roi=roi)

    @property
    def roi(self) -> Optional[tuple[int, int, int, int]]:
        return self._roi

    def classify(self, frame: Frame, frame_index: int, timestamp_ms: float) -> LightObservation:
        del frame_index, timestamp_ms
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return LightObservation(LightState.UNKNOWN, 0.0, source="yolo-cls")

        # ROI set qua control-plane API (web) ưu tiên hơn ROI của instance
        # để operator chỉnh lại vùng đèn lúc runtime mà không cần restart.
        from edge_node.core.config import get_active_light_roi

        valid = OpenCVTrafficLightClassifier._valid_roi
        roi = valid(frame, get_active_light_roi()) or valid(frame, self._roi)
        if roi is None:
            # Chưa có ROI do operator kẻ: KHÔNG đoán ROI bằng HSV blob search.
            # Trước khi operator kẻ ô đèn trên web thì không có vùng đèn tin
            # cậy; ROI đoán bừa có thể bắt nhầm đèn pha xe hoặc biển hiệu —
            # báo UNKNOWN thay vì trả kết quả sai.
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
    """Tạo classifier đèn giao thông tốt nhất có sẵn.

    Trả về ``YoloTrafficLightClassifier`` khi có weights fine-tuned,
    ngược lại fallback sang ``OpenCVTrafficLightClassifier`` (HSV).
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
