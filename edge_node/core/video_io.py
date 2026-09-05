"""Adapter nguồn video đầu vào.

Hỗ trợ file video local và luồng RTSP.
OpenCV được import trễ để logic cốt lõi test được mà không cần cài
dependency video native.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator, Optional

from edge_node.core.contracts import FramePacket

LOGGER = logging.getLogger(__name__)


def _load_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for video input. "
            "Install it or provide a custom FrameSource."
        ) from exc
    return cv2


class OpenCVFrameSource:
    """FrameSource backed by ``cv2.VideoCapture``.

    Works with local video files, device indices (``0``), and RTSP URLs.
    When ``loop=True`` and the source is a file, playback restarts from
    the beginning when the video ends — useful for testing without a
    live camera stream.

    ``realtime=True`` bật chế độ bám thời gian thực cho cả file lẫn stream:
    nếu pipeline xử lý nhanh hơn tốc độ camera/video, ``__iter__`` sẽ sleep
    cho khớp wall-clock. Nếu xử lý chậm hơn, các frame cũ bị bỏ qua bằng
    ``grab()`` (rẻ hơn ``read()`` vì không decode) để tránh độ trễ tích luỹ.
    Mặc định TẮT — file video chạy càng nhanh càng tốt (thường là muốn khi
    benchmark trên dataset). BẬT khi cần demo tốc độ thật cho người xem.

    ``playback_fps`` (chỉ áp dụng khi realtime=True): None = dùng
    ``CAP_PROP_FPS`` của video. Đặt số cụ thể (15 / 30 / 60) để override —
    hữu ích khi video không có FPS metadata đúng hoặc muốn xem chậm hơn /
    nhanh hơn thực tế.

    ``max_lag_ms`` chỉ có nghĩa với stream — file video đã có sẵn
    CAP_PROP_POS_MSEC để pace.
    """

    def __init__(
        self,
        input_path: str | Path,
        max_frames: Optional[int] = None,
        loop: bool = False,
        realtime: bool = False,
        max_lag_ms: float = 1000.0,
        playback_fps: Optional[float] = None,
    ) -> None:
        self._input_path = str(input_path)
        self._max_frames = max_frames
        self._loop = loop
        self._realtime = realtime
        self._max_lag_ms = max_lag_ms
        self._playback_fps = playback_fps

        # Validate only for local files (not RTSP or device index)
        if not self._is_stream and not Path(self._input_path).exists():
            raise FileNotFoundError(
                f"Input video does not exist: {self._input_path}"
            )
        if self._max_frames is not None and self._max_frames < 1:
            raise ValueError("max_frames must be >= 1 when supplied")

    @property
    def _is_stream(self) -> bool:
        return (
            self._input_path.startswith("rtsp://")
            or self._input_path.startswith("http://")
            or self._input_path.startswith("https://")
            or self._input_path.isdigit()
        )

    def __iter__(self) -> Iterator[FramePacket]:
        cv2 = _load_cv2()
        source = (
            int(self._input_path)
            if self._input_path.isdigit()
            else self._input_path
        )

        frame_index = 0

        while True:
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                raise RuntimeError(
                    f"Could not open video source: {self._input_path}"
                )

            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            # playback_fps override CAP_PROP_FPS (vd: file metadata sai, hoặc
            # muốn xem 2x/0.5x thực tế).
            effective_fps = self._playback_fps if (
                self._playback_fps is not None and self._playback_fps > 0
            ) else fps

            # ---- Realtime pacing (cả file lẫn stream) ----
            # realtime=True: pace theo wall-clock. Nếu pipeline xử lý nhanh
            # hơn effective_fps → sleep; nếu chậm hơn → grab() bỏ frame cũ
            # để tránh lag tích luỹ. File dùng CAP_PROP_POS_MSEC làm "ground
            # truth" nên đường này không cần max_lag — pace thẳng theo ms.
            # Stream thì cần max_lag để tránh trễ quá nhiều khi bitrate tụt.
            realtime_active = self._realtime
            start_wall = time.monotonic()
            frames_yielded = 0
            max_lag_frames = (self._max_lag_ms / 1000.0) * effective_fps if effective_fps > 0 else 0

            try:
                while True:
                    if (
                        self._max_frames is not None
                        and frame_index >= self._max_frames
                    ):
                        return

                    if realtime_active:
                        elapsed = time.monotonic() - start_wall
                        expected = elapsed * effective_fps
                        behind = expected - frames_yielded

                        # Pipeline đang nhanh hơn real-time → sleep cho khớp.
                        # Chỉ áp dụng khi ahead (behind < 0); nếu behind đủ âm,
                        # ngủ theo thời gian tương ứng.
                        if behind < -0.5 and effective_fps > 0:
                            sleep_s = -behind / effective_fps
                            if sleep_s > 0:
                                time.sleep(sleep_s)
                            behind = (time.monotonic() - start_wall) * effective_fps - frames_yielded

                        # Pipeline đang chậm hơn (đặc biệt stream) → grab() bỏ
                        # frame cũ để bám wall-clock, tránh lag tích luỹ.
                        if self._is_stream:
                            drop_target = int(behind - max_lag_frames)
                            if drop_target > 0:
                                grabbed = 0
                                for _ in range(drop_target):
                                    if not capture.grab():
                                        break
                                    grabbed += 1
                                if grabbed:
                                    frames_yielded += grabbed
                                    LOGGER.debug(
                                        "Realtime: dropped %d frame(s) to catch up "
                                        "(lag=%.0fms)",
                                        grabbed, behind * (1000.0 / effective_fps if effective_fps else 0),
                                    )

                    ok, frame = capture.read()
                    if not ok:
                        break

                    timestamp_ms = float(
                        capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0
                    )
                    if timestamp_ms <= 0.0 and effective_fps > 0.0:
                        timestamp_ms = frame_index * 1000.0 / effective_fps

                    yield FramePacket(
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        image=frame,
                    )
                    frame_index += 1
                    frames_yielded += 1
            finally:
                capture.release()

            # Only loop for local files, not streams
            if not self._loop or self._is_stream:
                return
