"""Lấy frame dùng cho màn hình calibration trên web.

Toàn bộ quy trình tự động dò đèn giao thông + vạch kẻ đã bị loại bỏ:
operator tự kẻ vạch dừng và vùng đèn trên web UI sau khi camera đăng ký
với Central Server. Module này chỉ còn một nhiệm vụ: đọc một frame từ
nguồn video để làm ảnh nền cho CalibrationEditor.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def grab_calibration_frame(
    video_path: str, skip_frames: int = 30,
) -> Optional[np.ndarray]:
    """Đọc một frame từ *video_path* (bỏ qua ``skip_frames`` frame đầu).

    Trả về frame BGR dạng numpy array, hoặc ``None`` nếu không đọc được
    (nguồn video hỏng / RTSP mất kết nối).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        frame: Optional[np.ndarray] = None
        for _ in range(skip_frames + 1):
            ok, f = cap.read()
            if not ok:
                break
            frame = f
        return frame
    finally:
        cap.release()
