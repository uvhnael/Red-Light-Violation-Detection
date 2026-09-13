"""Mô hình chuyển động của track: quỹ đạo, vận tốc, hướng đi.

ByteTrack chỉ cho ta box hiện tại (Kalman-smoothed). Với camera 3 fps một xe
máy đi được hàng chục pixel giữa hai frame, nên muốn downstream (violation
logic, evidence, gating) hiểu "xe đang đi đâu, nhanh thế nào, có đáng tin
không" thì PHẢI giữ lịch sử quỹ đạo riêng.

Đơn vị: pixel và GIÂY theo ``timestamp_ms`` của frame — không phải frame — nên
vận tốc so sánh được giữa camera 3 fps và 30 fps. Quy đổi sang m/s cần
homography/calibration mặt đường (việc riêng, chưa làm ở tầng này).

``TrackSample`` / ``MotionState`` sống ở ``contracts.py`` (Track cần chúng);
module này chỉ chứa logic tích luỹ + ước lượng.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Sequence

from edge_node.core.contracts import MotionState, Point, TrackSample


def heading_label(heading_deg: float) -> str:
    """Góc → nhãn 8 hướng theo hệ toạ độ ẢNH (y trỏ XUỐNG).

    0° = sang phải, 90° = xuống dưới (tiến về phía camera trong đa số cảnh
    giao thông), ±180° = sang trái, -90° = lên trên (đi ra xa camera).
    """
    names = (
        "right", "down_right", "down", "down_left",
        "left", "up_left", "up", "up_right",
    )
    index = int(math.floor((heading_deg + 22.5) / 45.0)) % 8
    return names[index]


def _weighted_slope(ts: Sequence[float], values: Sequence[float]) -> float:
    """Least-squares slope của ``values`` theo ``ts`` (đơn vị/giây).

    Hồi quy thay vì sai phân 2 điểm liền kề: bbox rung từng frame (detector
    jitter) sẽ được trải đều ra thay vì thổi phồng vận tốc tức thời.
    """
    n = len(ts)
    if n < 2:
        return 0.0
    span = ts[-1] - ts[0]
    if span <= 1e-9:
        return 0.0
    mean_t = sum(ts) / n
    mean_v = sum(values) / n
    num = sum((t - mean_t) * (v - mean_v) for t, v in zip(ts, values))
    den = sum((t - mean_t) ** 2 for t in ts)
    if den <= 1e-12:
        return 0.0
    return num / den


class TrackMotion:
    """Giữ quỹ đạo một track và ước lượng vận tốc/hướng từ đó."""

    def __init__(self, max_samples: int = 12) -> None:
        if max_samples < 2:
            raise ValueError("max_samples must be >= 2")
        self._max_samples = max_samples
        self._samples: deque[TrackSample] = deque(maxlen=max_samples)

    # ------------------------------------------------------------------ #
    @property
    def samples(self) -> tuple[TrackSample, ...]:
        return tuple(self._samples)

    @property
    def max_samples(self) -> int:
        return self._max_samples

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def last_point(self) -> Optional[Point]:
        return self._samples[-1].point if self._samples else None

    def clear(self) -> None:
        self._samples.clear()

    # ------------------------------------------------------------------ #
    def push(self, sample: TrackSample) -> None:
        """Thêm một mẫu quỹ đạo.

        * Cùng timestamp với mẫu cuối (một frame update hai lần) → THAY mẫu cũ;
          nối thêm sẽ sinh dt=0 → vận tốc vô cực, phá phép hồi quy.
        * Timestamp thụt lùi (seek/restart nguồn) → xoá quỹ đạo cũ; hồi quy
          xuyên qua điểm gãy cho ra vận tốc vô nghĩa.
        """
        if self._samples:
            last_ts = self._samples[-1].timestamp_ms
            if sample.timestamp_ms == last_ts:
                # deque(maxlen=..) không hỗ trợ gán theo chỉ số
                self._samples.pop()
                self._samples.append(sample)
                return
            if sample.timestamp_ms < last_ts:
                self._samples.clear()
        self._samples.append(sample)

    def motion(self) -> MotionState:
        """Hồi quy vị trí theo thời gian → vận tốc pixel/giây + hướng.

        Thời gian lấy từ ``timestamp_ms`` của nguồn. ``push()`` đã đảm bảo
        timestamp tăng nghiêm ngặt nên span=0 chỉ xảy ra khi có <2 mẫu —
        lúc đó trả về vận tốc 0 thay vì một con số vô nghĩa.
        """
        pts = list(self._samples)
        if len(pts) < 2:
            return MotionState(samples=len(pts))

        t0 = pts[0].timestamp_ms
        ts = [(s.timestamp_ms - t0) / 1000.0 for s in pts]
        vx = _weighted_slope(ts, [s.point.x for s in pts])
        vy = _weighted_slope(ts, [s.point.y for s in pts])
        speed = math.hypot(vx, vy)
        if speed <= 1e-9:
            return MotionState(
                samples=len(pts),
                span_ms=pts[-1].timestamp_ms - t0,
                direction="stationary",
            )
        heading = math.degrees(math.atan2(vy, vx))
        return MotionState(
            vx=vx,
            vy=vy,
            speed=speed,
            heading_deg=heading,
            direction=heading_label(heading),
            samples=len(pts),
            span_ms=pts[-1].timestamp_ms - t0,
        )

    def predicted_point(self, timestamp_ms: float) -> Optional[Point]:
        """Ngoại suy vị trí tại *timestamp_ms* theo vận tốc hiện tại.

        ``None`` khi chưa đủ 2 mẫu. Dùng cho gating: so điểm dự đoán từ quỹ
        đạo THẬT với box Kalman để phát hiện association nhảy xa.
        """
        if len(self._samples) < 2:
            return None
        last = self._samples[-1]
        dt_s = (timestamp_ms - last.timestamp_ms) / 1000.0
        motion = self.motion()
        if dt_s <= 0.0 or motion.speed <= 1e-9:
            return last.point
        return Point(last.point.x + motion.vx * dt_s, last.point.y + motion.vy * dt_s)

    def as_dicts(self) -> list[dict[str, float | bool]]:
        """Quỹ đạo dạng JSON-safe (evidence/log vi phạm)."""
        return [s.as_dict() for s in self._samples]
