"""Live visualization wrapper for the violation detection pipeline.

Adds OpenCV-based visualization with:
- Bounding boxes + track IDs for detected vehicles
- Stop-line overlay with crossing direction
- Traffic light status indicator (top-right panel)
- Violation highlight flashes (red blink)
- FPS counter + detection/track/violation counts
- Press 'q' or ESC to quit
"""

from __future__ import annotations

import datetime
from typing import Optional, Sequence

import cv2
import numpy as np

from edge_node.core.contracts import (
    BoundingBox,
    Detection,
    LightObservation,
    LightState,
    Point,
    StableSignal,
    Track,
    ViolationEvent,
)

# ── Colors (BGR) ──
_RED = (0, 0, 255)
_YELLOW = (0, 255, 255)
_GREEN = (0, 255, 0)
_CYAN = (255, 255, 0)
_WHITE = (255, 255, 255)
_ORANGE = (0, 165, 255)
_GRAY = (128, 128, 128)

_BLINK_DURATION_MS = 2000  # How long violation flash lasts before fading


class LiveVisualizer:
    """Renders detection/tracking/violation overlays on live frames."""

    def __init__(
        self,
        window_name: str = "RLVD Pipeline — Live View",
        show: bool = True,
        record_path: Optional[str] = None,
    ) -> None:
        self.window_name = window_name
        self.show = show
        self.record_path = record_path
        self._writer: Optional["cv2.VideoWriter"] = None  # type: ignore[name-defined]
        self._fps_value: float = 0.0
        self._fps_ts: float = 0.0
        self._fps_count: int = 0
        self._blinks: dict[str, float] = {}  # event_id → blink_start_epoch_ms

        if self.show:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 720)

    def _ensure_writer(self, frame: np.ndarray) -> None:
        if self.record_path and self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.record_path, fourcc, 30.0, (w, h))

    def update(
        self,
        frame: np.ndarray,
        *,
        detections: Sequence[Detection] = (),
        tracks: Sequence[Track] = (),
        light: Optional[LightObservation] = None,
        signal: Optional[StableSignal] = None,
        violations: Sequence[ViolationEvent] = (),
        stop_line: Optional[tuple[Point, Point]] = None,
        light_roi: Optional[tuple[int, int, int, int]] = None,
    ) -> Optional[np.ndarray]:
        """Render all overlays onto a copy of the frame.

        Returns the annotated frame (BGR), or None if *show* is False and
        no recording is active.
        """
        if not self.show and self._writer is None:
            return None

        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # ── 1. Stop-line ──
        if stop_line is not None:
            p1, p2 = stop_line
            cv2.line(
                canvas,
                (int(p1.x), int(p1.y)),
                (int(p2.x), int(p2.y)),
                _ORANGE, 3,
            )
            mid_x = int((p1.x + p2.x) / 2)
            mid_y = int((p1.y + p2.y) / 2)
            cv2.putText(
                canvas, "STOP LINE", (mid_x - 50, mid_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _ORANGE, 1, cv2.LINE_AA,
            )

        # ── 2. Light ROI ──
        if light_roi is not None:
            rx, ry, rw, rh = light_roi
            cv2.rectangle(
                canvas, (rx, ry), (rx + rw, ry + rh),
                _YELLOW, 1, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, "TRAFFIC LIGHT", (rx, max(ry - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _YELLOW, 1, cv2.LINE_AA,
            )

        # ── 3. Detections (thin gray boxes) ──
        for det in detections:
            cv2.rectangle(
                canvas,
                (int(det.bbox.x1), int(det.bbox.y1)),
                (int(det.bbox.x2), int(det.bbox.y2)),
                _GRAY, 1, cv2.LINE_AA,
            )

        # ── 4. Tracks (green boxes + ID labels) ──
        for track in tracks:
            tx1, ty1 = int(track.bbox.x1), int(track.bbox.y1)
            tx2, ty2 = int(track.bbox.x2), int(track.bbox.y2)
            cv2.rectangle(
                canvas, (tx1, ty1), (tx2, ty2), _GREEN, 2, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, f"ID:{track.track_id}",
                (tx1, max(ty1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GREEN, 1, cv2.LINE_AA,
            )

        # ── 5. Violation highlights (flashing red) ──
        if violations:
            now_ms = datetime.datetime.now().timestamp() * 1000
            for evt in violations:
                ex1, ey1 = int(evt.bbox.x1), int(evt.bbox.y1)
                ex2, ey2 = int(evt.bbox.x2), int(evt.bbox.y2)

                blink_start = self._blinks.get(evt.event_id)
                if blink_start is None:
                    self._blinks[evt.event_id] = now_ms
                    blink_start = now_ms

                elapsed = now_ms - blink_start
                if elapsed < 800:
                    thickness = 4
                    color = (0, 0, 255)
                elif elapsed < _BLINK_DURATION_MS:
                    thickness = 2
                    color = (0, 60, 180)
                else:
                    thickness = 1
                    color = (0, 0, 180)

                cv2.rectangle(
                    canvas, (ex1, ey1), (ex2, ey2),
                    color, thickness, cv2.LINE_AA,
                )
                # Red label bar
                cv2.rectangle(
                    canvas, (ex1, ey1 - 24), (ex1 + 136, ey1),
                    (0, 0, 255), -1, cv2.LINE_AA,
                )
                cv2.putText(
                    canvas, "VIOLATION!",
                    (ex1 + 5, ey1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA,
                )

        # ── 6. Traffic Light Status Panel (top-right) ──
        if light is not None:
            if light.state == LightState.RED:
                lamp_color = _RED
            elif light.state == LightState.YELLOW:
                lamp_color = _YELLOW
            else:
                lamp_color = _GREEN

            px = w - 220
            py = 10
            pw, ph = 215, 80

            cv2.rectangle(
                canvas, (px, py), (px + pw, py + ph),
                (0, 0, 0), -1,
            )
            cv2.rectangle(
                canvas, (px, py), (px + pw, py + ph),
                _CYAN, 1, cv2.LINE_AA,
            )

            # Lamp circle
            cv2.circle(canvas, (px + 18, py + 18), 8, lamp_color, -1)
            cv2.circle(canvas, (px + 18, py + 18), 9, _WHITE, 1)

            cv2.putText(
                canvas, f"LIGHT: {light.state.value.upper()}",
                (px + 34, py + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, lamp_color, 1, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, f"Conf: {light.confidence:.0%}",
                (px + 34, py + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _WHITE, 1, cv2.LINE_AA,
            )

            if signal is not None:
                stable_label = "STABLE" if signal.stable else "unstable"
                stable_color = _GREEN if signal.stable else _GRAY
                cv2.putText(
                    canvas, stable_label,
                    (px + 34, py + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, stable_color, 1, cv2.LINE_AA,
                )

        # ── 7. Status bar (bottom) ──
        bar_h = 28
        cv2.rectangle(
            canvas, (0, h - bar_h), (w, h),
            (15, 15, 15), -1,
        )
        cv2.line(
            canvas, (0, h - bar_h), (w, h - bar_h),
            _CYAN, 1, cv2.LINE_AA,
        )

        # FPS
        now = datetime.datetime.now().timestamp()
        self._fps_count += 1
        if now - self._fps_ts >= 1.0:
            self._fps_value = self._fps_count
            self._fps_count = 0
            self._fps_ts = now

        cv2.putText(
            canvas, f"FPS:{self._fps_value:.0f}",
            (12, h - int(bar_h * 0.35)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, f"D:{len(detections)}  T:{len(tracks)}  V:{len(violations)}",
            (120, h - int(bar_h * 0.35)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CYAN, 1, cv2.LINE_AA,
        )

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(
            canvas, ts,
            (w - tw - 12, h - int(bar_h * 0.35)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA,
        )

        # ── 8. Show / Record ──
        if self.show:
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                raise KeyboardInterrupt("User closed video window (q/ESC pressed)")

        if self._writer is not None:
            self._writer.write(canvas)

        return canvas

    def close(self) -> None:
        if self.show:
            cv2.destroyWindow(self.window_name)
        if self._writer is not None:
            self._writer.release()
            self._writer = None