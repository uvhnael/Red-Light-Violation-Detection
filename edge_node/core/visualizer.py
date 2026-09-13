"""Wrapper trực quan hoá pipeline phát hiện vi phạm lúc chạy thật.

Adds OpenCV-based visualization with:
- Bounding boxes + track IDs for detected vehicles
- Xe đang vi phạm: box đỏ đậm + nhãn VIOLATION (thay vì xanh/cyan)
- Stop-line overlay with crossing direction
- Traffic light status indicator (top-right panel)
- Violation highlight flashes (red blink)
- FPS counter + detection/track/violation counts
- Press 'q' or ESC to quit

Toàn bộ độ dày nét vẽ, cỡ chữ, kích thước panel được scale theo độ phân
giải frame (chuẩn 1920x1080 = scale 1.0) nên video 720p hay 4K đều dễ nhìn.
"""

from __future__ import annotations

import datetime
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np

from edge_node.core.contracts import (
    BoundingBox,
    Detection,
    LightObservation,
    LightState,
    PlateObservation,
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

# Độ phân giải tham chiếu để tính UI scale (1080p = 1.0)
_REF_W, _REF_H = 1920.0, 1080.0
_MIN_SCALE = 0.6


def _ui_scale(width: int, height: int) -> float:
    """Hệ số scale UI theo độ phân giải frame (1080p = 1.0, min 0.6)."""
    return max(_MIN_SCALE, max(width / _REF_W, height / _REF_H))


class LiveVisualizer:
    """Renders detection/tracking/violation overlays on live frames."""

    def __init__(
        self,
        window_name: str = "RLVD Pipeline — Live View",
        show: bool = True,
        record_path: Optional[str] = None,
        record_fps: Optional[float] = None,
    ) -> None:
        """Khởi tạo LiveVisualizer.

        ``record_fps``: FPS dùng cho VideoWriter khi ``record_path`` được set.
        Mặc định 30.0 (giữ tương thích cũ). Caller nên truyền đúng FPS đang
        pace để video xuất ra khớp realtime preview — đặc biệt khi chạy với
        ``--realtime`` ở ``run_pipeline.py``: video sẽ chạy theo đúng
        ``effective_fps`` thay vì nhân fps cứng 30 làm tua nhanh.
        """
        self.window_name = window_name
        self.show = show
        self.record_path = record_path
        self._record_fps: float = float(record_fps) if (
            record_fps is not None and record_fps > 0
        ) else 30.0
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
            self._writer = cv2.VideoWriter(
                self.record_path, fourcc, self._record_fps, (w, h),
            )

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
        direction_arrow: Optional[tuple[Point, Point]] = None,
        light_roi: Optional[tuple[int, int, int, int]] = None,
        track_plates: Mapping[int, PlateObservation] = {},
        plates: Sequence[tuple[BoundingBox, Optional[PlateObservation]]] = (),
        min_plate_confidence: float = 0.80,
        show_motion: bool = False,
    ) -> Optional[np.ndarray]:
        """Render all overlays onto a copy of the frame.

        ``show_motion``: vẽ quỹ đạo (trail) + vận tốc/hướng của từng track —
        dữ liệu mới từ tầng track-quality (xem tracker_update.md). Mặc định
        tắt vì cảnh đông xe sẽ rối; bật bằng ``--show-motion`` ở
        ``run_pipeline.py`` để kiểm tra quỹ đạo có đúng không.

        Returns the annotated frame (BGR), or None if *show* is False and
        no recording is active.
        """
        if not self.show and self._writer is None and self.record_path is None:
            return None

        canvas = frame.copy()
        self._ensure_writer(canvas)
        h, w = canvas.shape[:2]

        # ── UI scale theo độ phân giải video ──
        s = _ui_scale(w, h)

        def th(v: float) -> int:
            """Độ dày nét vẽ scaled (luôn >= 1 px)."""
            return max(1, int(round(v * s)))

        def fs(v: float) -> float:
            """Font scale scaled (không nhỏ hơn 0.35)."""
            return max(0.35, v * s)

        def px(v: float) -> int:
            """Offset/kích thước pixel scaled."""
            return int(round(v * s))

        violating_ids = {evt.track_id for evt in violations}

        # ── 1. Stop-line ──
        if stop_line is not None:
            p1, p2 = stop_line
            cv2.line(
                canvas,
                (int(p1.x), int(p1.y)),
                (int(p2.x), int(p2.y)),
                _ORANGE, th(3),
            )
            mid_x = int((p1.x + p2.x) / 2)
            mid_y = int((p1.y + p2.y) / 2)
            cv2.putText(
                canvas, "STOP LINE", (mid_x - px(50), mid_y - px(12)),
                cv2.FONT_HERSHEY_SIMPLEX, fs(0.5), _ORANGE, th(1), cv2.LINE_AA,
            )

        # ── 1b. Mũi tên hướng giám sát (đường 2 chiều: chỉ hướng xe bị tính) ──
        if direction_arrow is not None:
            a1, a2 = direction_arrow
            cv2.arrowedLine(
                canvas,
                (int(a1.x), int(a1.y)),
                (int(a2.x), int(a2.y)),
                _GREEN, th(3), cv2.LINE_AA, tipLength=0.3,
            )

        # ── 2. Light ROI + current light state above the box ──
        # Prefer the debounced (stable) signal over the raw per-frame
        # observation so the label does not flicker between colours.
        display_state: Optional[LightState] = None
        display_conf = 0.0
        if signal is not None and signal.state != LightState.UNKNOWN:
            display_state = signal.state
            display_conf = signal.confidence
        elif light is not None and light.state != LightState.UNKNOWN:
            display_state = light.state
            display_conf = light.confidence

        if light_roi is not None:
            rx, ry, rw, rh = light_roi
            cv2.rectangle(
                canvas, (rx, ry), (rx + rw, ry + rh),
                _YELLOW, th(1), cv2.LINE_AA,
            )
            if display_state is not None:
                if display_state == LightState.RED:
                    lamp_color = _RED
                elif display_state == LightState.YELLOW:
                    lamp_color = _YELLOW
                else:
                    lamp_color = _GREEN
                text = f"{display_state.value.upper()} {display_conf:.0%}"
                font = fs(0.5)
                (tw, theight), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font, th(1),
                )
                bar_y2 = max(ry - 2, theight + px(4))
                bar_y1 = bar_y2 - theight - px(6)
                cv2.rectangle(
                    canvas, (rx, bar_y1), (rx + tw + px(8), bar_y2),
                    lamp_color, -1, cv2.LINE_AA,
                )
                cv2.putText(
                    canvas, text, (rx + px(4), bar_y2 - px(4)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), th(1), cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    canvas, "TRAFFIC LIGHT", (rx, max(ry - px(5), px(15))),
                    cv2.FONT_HERSHEY_SIMPLEX, fs(0.4), _YELLOW, th(1), cv2.LINE_AA,
                )

        # ── 3. Detections (thin gray boxes) ──
        # CHỈ vẽ detection CHƯA match track nào (xe mới vào / sắp activate).
        # Detection đã match sẽ trùng track box (Kalman-smoothed) và vẽ cả hai
        # gây hiện tượng "box kép" khó nhìn — một box mỏng trong, một box đậm
        # ngoài, lệch nhau vì Kalman trễ 1 nhịp so với det frame hiện tại.
        def _iou_det_track(dbox, tbox) -> float:
            ix1 = max(dbox.x1, tbox.x1)
            iy1 = max(dbox.y1, tbox.y1)
            ix2 = min(dbox.x2, tbox.x2)
            iy2 = min(dbox.y2, tbox.y2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            union = (
                (dbox.x2 - dbox.x1) * (dbox.y2 - dbox.y1)
                + (tbox.x2 - tbox.x1) * (tbox.y2 - tbox.y1)
                - inter
            )
            return inter / max(union, 1e-6)

        for det in detections:
            matched = any(
                _iou_det_track(det.bbox, track.bbox) > 0.30 for track in tracks
            )
            if matched:
                continue  # đã có track box vẽ đậm — khỏi vẽ đè box xám
            cv2.rectangle(
                canvas,
                (int(det.bbox.x1), int(det.bbox.y1)),
                (int(det.bbox.x2), int(det.bbox.y2)),
                _GRAY, th(1), cv2.LINE_AA,
            )

        # ── 4. Tracks (green boxes + ID labels + plate text above box) ──
        # Xe đang vi phạm → box đỏ đậm + nhãn VIOLATION thay cho ID thường.
        for track in tracks:
            tx1, ty1 = int(track.bbox.x1), int(track.bbox.y1)
            tx2, ty2 = int(track.bbox.x2), int(track.bbox.y2)
            plate_obs = track_plates.get(track.track_id)
            show_plate = (
                plate_obs is not None
                and plate_obs.confidence >= min_plate_confidence
            )
            is_violator = track.track_id in violating_ids
            if is_violator:
                box_color = _RED
                line_w = th(4)
            elif show_plate:
                box_color = _CYAN
                line_w = th(2)
            else:
                box_color = _GREEN
                line_w = th(2)
            cv2.rectangle(
                canvas, (tx1, ty1), (tx2, ty2), box_color, line_w, cv2.LINE_AA,
            )
            id_y = max(ty1 - px(8), px(12))
            # Plate text label above the vehicle box (high-confidence only)
            if show_plate and plate_obs is not None:
                text = f"{plate_obs.text} ({plate_obs.confidence:.0%})"
                font = fs(0.55)
                (tw, theight), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font, th(2),
                )
                bar_y2 = max(ty1 - 2, theight + px(6))
                bar_y1 = bar_y2 - theight - px(6)
                cv2.rectangle(
                    canvas, (tx1, bar_y1), (tx1 + tw + px(8), bar_y2),
                    _CYAN, -1, cv2.LINE_AA,
                )
                cv2.putText(
                    canvas, text,
                    (tx1 + px(4), bar_y2 - px(4)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), th(2), cv2.LINE_AA,
                )
                id_y = max(bar_y1 - px(6), px(12))
            if is_violator:
                # Nhãn đỏ nền đặc: "ID:x VIOLATION"
                vtext = f"ID:{track.track_id} VIOLATION"
                font = fs(0.55)
                (tw, theight), _ = cv2.getTextSize(
                    vtext, cv2.FONT_HERSHEY_SIMPLEX, font, th(2),
                )
                bar_y2 = max(id_y, theight + px(6))
                bar_y1 = bar_y2 - theight - px(6)
                cv2.rectangle(
                    canvas, (tx1, bar_y1), (tx1 + tw + px(10), bar_y2),
                    _RED, -1, cv2.LINE_AA,
                )
                cv2.putText(
                    canvas, vtext,
                    (tx1 + px(5), bar_y2 - px(4)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, _WHITE, th(2), cv2.LINE_AA,
                )
            else:
                # "*" = track vừa được tìm lại sau khi mất detection
                # (time_since_update > 0) → box là Kalman prediction, có thể lệch
                stale = getattr(track, "time_since_update", 0) > 0
                id_text = f"ID:{track.track_id}{'*' if stale else ''}"
                cv2.putText(
                    canvas, id_text,
                    (tx1, id_y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs(0.45),
                    _YELLOW if stale else box_color, th(1), cv2.LINE_AA,
                )

        # ── 4b. Quỹ đạo + vận tốc (tầng track-quality, tắt mặc định) ──
        # Vẽ polyline nối các điểm neo (bottom-center) theo thời gian và
        # nhãn "v px/s · hướng" để kiểm tra trực quan motion ước lượng có
        # đúng không (camera 3fps: quỹ đạo thưa, dễ thấy chỗ Kalman đoán sai).
        if show_motion:
            for track in tracks:
                trail = getattr(track, "trajectory", ())
                pts = [(int(s.point.x), int(s.point.y)) for s in trail]
                for i in range(1, len(pts)):
                    # Cũ → mới: đậm dần để thấy chiều di chuyển
                    cv2.line(
                        canvas, pts[i - 1], pts[i],
                        _YELLOW, th(1), cv2.LINE_AA,
                    )
                if pts:
                    cv2.circle(canvas, pts[-1], max(2, th(2)), _YELLOW, -1)
                motion = getattr(track, "motion", None)
                if motion is None or motion.speed <= 0.0:
                    continue
                anchor = track.crossing_point
                text = f"{motion.speed:.0f}px/s {motion.direction}"
                cv2.putText(
                    canvas, text,
                    (int(anchor.x) + px(6), int(anchor.y) + px(14)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs(0.4), _YELLOW, th(1), cv2.LINE_AA,
                )

        # ── 4c. Unassigned license plates (cyan box, no vehicle match) ──
        for plate_bbox, plate_obs in plates:
            px1, py1 = int(plate_bbox.x1), int(plate_bbox.y1)
            px2, py2 = int(plate_bbox.x2), int(plate_bbox.y2)
            cv2.rectangle(
                canvas, (px1, py1), (px2, py2), _CYAN, th(1), cv2.LINE_AA,
            )
            if plate_obs is None:
                continue
            # Plate text label below the box
            label = plate_obs.text
            conf = plate_obs.confidence
            text = f"{label} ({conf:.0%})"
            font = fs(0.55)
            (tw, theight), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font, th(2),
            )
            # Background bar under the box
            bar_y1 = min(py2 + 2, h - theight - px(6))
            cv2.rectangle(
                canvas, (px1, bar_y1), (px1 + tw + px(8), bar_y1 + theight + px(6)),
                _CYAN, -1, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, text,
                (px1 + px(4), bar_y1 + theight + px(2)),
                cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), th(2), cv2.LINE_AA,
            )

        # ── 5. Violation highlights (flashing red at crossing snapshot) ──
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
                    thickness = th(4)
                    color = (0, 0, 255)
                elif elapsed < _BLINK_DURATION_MS:
                    thickness = th(2)
                    color = (0, 60, 180)
                else:
                    thickness = th(1)
                    color = (0, 0, 180)

                cv2.rectangle(
                    canvas, (ex1, ey1), (ex2, ey2),
                    color, thickness, cv2.LINE_AA,
                )
                # Red label bar (width theo cỡ chữ, không hard-code)
                vtext = "VIOLATION!"
                font = fs(0.5)
                (tw, theight), _ = cv2.getTextSize(
                    vtext, cv2.FONT_HERSHEY_SIMPLEX, font, th(1),
                )
                bar_h_v = theight + px(10)
                cv2.rectangle(
                    canvas, (ex1, ey1 - bar_h_v), (ex1 + tw + px(12), ey1),
                    (0, 0, 255), -1, cv2.LINE_AA,
                )
                cv2.putText(
                    canvas, vtext,
                    (ex1 + px(6), ey1 - px(5)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, _WHITE, th(1), cv2.LINE_AA,
                )

        # ── 6. Traffic Light Status Panel (top-right) ──
        if light is not None:
            panel_state = display_state if display_state is not None else light.state
            if panel_state == LightState.RED:
                lamp_color = _RED
            elif panel_state == LightState.YELLOW:
                lamp_color = _YELLOW
            else:
                lamp_color = _GREEN

            pw, ph = px(215), px(80)
            pxp = w - pw - px(5)
            pyp = px(10)

            cv2.rectangle(
                canvas, (pxp, pyp), (pxp + pw, pyp + ph),
                (0, 0, 0), -1,
            )
            cv2.rectangle(
                canvas, (pxp, pyp), (pxp + pw, pyp + ph),
                _CYAN, th(1), cv2.LINE_AA,
            )

            # Lamp circle
            lamp_r = max(4, px(8))
            cv2.circle(canvas, (pxp + px(18), pyp + px(18)), lamp_r, lamp_color, -1)
            cv2.circle(canvas, (pxp + px(18), pyp + px(18)), lamp_r + 1, _WHITE, th(1))

            cv2.putText(
                canvas, f"LIGHT: {panel_state.value.upper()}",
                (pxp + px(34), pyp + px(22)),
                cv2.FONT_HERSHEY_SIMPLEX, fs(0.45), lamp_color, th(1), cv2.LINE_AA,
            )
            cv2.putText(
                canvas, f"Conf: {display_conf:.0%}",
                (pxp + px(34), pyp + px(40)),
                cv2.FONT_HERSHEY_SIMPLEX, fs(0.4), _WHITE, th(1), cv2.LINE_AA,
            )

            if signal is not None:
                stable_label = "STABLE" if signal.stable else "unstable"
                stable_color = _GREEN if signal.stable else _GRAY
                cv2.putText(
                    canvas, stable_label,
                    (pxp + px(34), pyp + px(58)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs(0.4), stable_color, th(1), cv2.LINE_AA,
                )

        # ── 7. Status bar (bottom) ──
        bar_h = max(20, px(28))
        cv2.rectangle(
            canvas, (0, h - bar_h), (w, h),
            (15, 15, 15), -1,
        )
        cv2.line(
            canvas, (0, h - bar_h), (w, h - bar_h),
            _CYAN, th(1), cv2.LINE_AA,
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
            (px(12), h - int(bar_h * 0.35)),
            cv2.FONT_HERSHEY_SIMPLEX, fs(0.5), _WHITE, th(1), cv2.LINE_AA,
        )
        cv2.putText(
            canvas, f"D:{len(detections)}  T:{len(tracks)}  V:{len(violations)}",
            (px(120), h - int(bar_h * 0.35)),
            cv2.FONT_HERSHEY_SIMPLEX, fs(0.45), _CYAN, th(1), cv2.LINE_AA,
        )

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        ts_font = fs(0.45)
        (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, ts_font, th(1))
        cv2.putText(
            canvas, ts,
            (w - tw - px(12), h - int(bar_h * 0.35)),
            cv2.FONT_HERSHEY_SIMPLEX, ts_font, _WHITE, th(1), cv2.LINE_AA,
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
