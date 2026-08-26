"""Adapter nguồn video đầu vào.

Hỗ trợ file video local và luồng RTSP.
OpenCV được import trễ để logic cốt lõi test được mà không cần cài
dependency video native.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Optional

from edge_node.core.contracts import FramePacket


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
    """

    def __init__(
        self,
        input_path: str | Path,
        max_frames: Optional[int] = None,
        loop: bool = False,
    ) -> None:
        self._input_path = str(input_path)
        self._max_frames = max_frames
        self._loop = loop

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

            try:
                while True:
                    if (
                        self._max_frames is not None
                        and frame_index >= self._max_frames
                    ):
                        return

                    ok, frame = capture.read()
                    if not ok:
                        break

                    timestamp_ms = float(
                        capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0
                    )
                    if timestamp_ms <= 0.0 and fps > 0.0:
                        timestamp_ms = frame_index * 1000.0 / fps

                    yield FramePacket(
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        image=frame,
                    )
                    frame_index += 1
            finally:
                capture.release()

            # Only loop for local files, not streams
            if not self._loop or self._is_stream:
                return


class IterableFrameSource:
    """In-memory frame source used by tests, demos, and external integrations."""

    def __init__(self, packets: Iterable[FramePacket]) -> None:
        self._packets = tuple(packets)

    def __iter__(self) -> Iterator[FramePacket]:
        return iter(self._packets)
