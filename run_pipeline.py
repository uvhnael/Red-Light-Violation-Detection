#!/usr/bin/env python3
"""Runner test pipeline phát hiện vi phạm vượt đèn đỏ — KHÔNG gửi server.

Chỉ chạy đến bước phát hiện vi phạm: detect -> track -> đèn -> tripwire.
Không outbox, không sender, không đăng ký central.

Calibration (vạch dừng + hướng giám sát + vùng đèn):
* Lần đầu: kẻ bằng chuột trên frame đầu tiên của video.
  - Vạch dừng: click 2 điểm.
  - Hướng giám sát (đường 2 chiều): vẽ mũi tên chỉ hướng xe bị tính.
    Bỏ qua (Enter) nếu đường 1 chiều → giám sát cả 2 hướng.
  - Vùng đèn: click 2 điểm bao quanh đèn.
* Được lưu vào data/calibration.json (theo tên video) — lần sau tự nạp lại.
* --recalibrate để kẻ lại; --stop-line / --light-roi / --direction để truyền thẳng.

Đường 2 chiều: khi đèn chiều mình giám sát là đỏ thì chiều ngược lại là xanh.
Xe chiều ngược chạy qua hợp lệ nhưng sẽ bị tính nhầm nếu direction="any".
Khắc phục: calibration mũi tên hướng giám sát (hoặc --direction) để chỉ tính
xe đi đúng chiều, bỏ qua xe chiều ngược lại.

Cách dùng:
    python run_pipeline.py                        # video mặc định trong data/videos/
    python run_pipeline.py data/videos/aziz1.MP4
    python run_pipeline.py --recalibrate          # kẻ lại vạch + hướng + vùng đèn
    python run_pipeline.py --stop-line 100,400,800,400 --light-roi 1500,100,80,160
    python run_pipeline.py --direction positive_to_negative   # chỉ giám sát 1 chiều
    python run_pipeline.py --no-window --record out.mp4

Điều khiển cửa sổ live: q/ESC = thoát.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CALIB_FILE = PROJECT_ROOT / "data" / "calibration.json"
LOGGER = logging.getLogger("run_pipeline")


# ────────────────────────────────────────────────────────────────────
# Calibration bằng chuột
# ────────────────────────────────────────────────────────────────────

def _draw_points(frame, title: str, hint: str, n_points: int,
                 arrow: bool = False, optional: bool = False):
    """Cho user click n_points trên frame.

    Trả về list [(x, y)], hoặc None (huỷ bằng ESC). Khi ``optional=True``
    và user bấm Enter/Space lúc chưa click điểm nào, trả về [] (bỏ qua bước).
    """
    import cv2

    # OpenCV 5.0.0 + QT5: ten cua so phai ASCII, neu khong
    # setMouseCallback se chet voi "NULL window handler".
    window = f"Calibrate - {title}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    pts: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n_points:
            pts.append((x, y))

    cv2.setMouseCallback(window, on_mouse)
    skip_hint = " | Enter(0 diem): bo qua" if optional else ""
    while True:
        canvas = frame.copy()
        for i, (px, py) in enumerate(pts):
            cv2.circle(canvas, (px, py), 5, (0, 255, 255), -1)
            cv2.putText(canvas, str(i + 1), (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if len(pts) == 2 and n_points == 2:
            if arrow:
                cv2.arrowedLine(canvas, pts[0], pts[1], (0, 255, 0), 2, tipLength=0.15)
            else:
                cv2.line(canvas, pts[0], pts[1], (0, 255, 0), 2)
        cv2.putText(canvas, hint, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(canvas, f"Da chon {len(pts)}/{n_points} — Enter: xong | r: lam lai | ESC: huy{skip_hint}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyWindow(window)
            return None
        if key == ord("r"):
            pts.clear()
        if key in (13, 32):  # Enter / Space
            if len(pts) == n_points:
                cv2.destroyWindow(window)
                return pts
            if optional and len(pts) == 0:
                cv2.destroyWindow(window)
                return []


def _direction_from_arrow(line_start, line_end, tail, head) -> str:
    """Suy ra CrossingDirection từ mũi tên chỉ hướng xe được giám sát.

    ``tail`` (điểm đầu mũi tên) nằm ở phía xe xuất phát. Xác định phía của
    tail so với vạch dừng có hướng (start->end) rồi map sang chiều cắt ngang.
    """
    import math

    from edge_node.core.contracts import Point
    from edge_node.core.geometry import side_of_line

    t = Point(float(tail[0]), float(tail[1]))
    side = side_of_line(t, line_start, line_end, deadband_px=0.0)
    if side == 0:
        # tail rơi đúng lên vạch: lùi lại dọc theo hướng mũi tên một chút
        dx = head[0] - tail[0]
        dy = head[1] - tail[1]
        length = math.hypot(dx, dy) or 1.0
        t = Point(tail[0] - dx / length * 5.0, tail[1] - dy / length * 5.0)
        side = side_of_line(t, line_start, line_end, deadband_px=0.0)
    if side > 0:
        return "positive_to_negative"
    if side < 0:
        return "negative_to_positive"
    return "any"


def _arrow_from_direction(line_start, line_end, direction):
    """Dựng mũi tên hiển thị (tail, head) ở giữa vạch dừng theo hướng giám sát."""
    import math

    from edge_node.core.contracts import Point

    if direction not in ("positive_to_negative", "negative_to_positive"):
        return None
    mx = (line_start.x + line_end.x) / 2.0
    my = (line_start.y + line_end.y) / 2.0
    dx = line_end.x - line_start.x
    dy = line_end.y - line_start.y
    length = math.hypot(dx, dy) or 1.0
    # Pháp tuyến phía (+) của vạch có hướng start->end là (-dy, dx)/length.
    if direction == "positive_to_negative":  # đi từ + sang - = -pháp tuyến
        tx, ty = dy / length, -dx / length
    else:  # negative_to_positive = +pháp tuyến
        tx, ty = -dy / length, dx / length
    half = 40.0
    return (Point(mx - tx * half, my - ty * half), Point(mx + tx * half, my + ty * half))


def calibrate_interactive(video_path: Path) -> dict:
    """Mở frame đầu tiên, kẻ vạch dừng + vùng đèn bằng chuột."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Không đọc được frame đầu từ {video_path}")

    calib: dict = {}

    print("\n=== Ke VACH DUNG: click 2 diem (trai -> phai theo huong xe chay) ===")
    pts = _draw_points(
        frame, "Stop line",
        "VACH DUNG: click 2 diem ngang qua lan duong",
        2,
    )
    if pts is None:
        print("Huy calibration — vi pham se TAT (khong co vach dung).")
        return calib
    calib["stop_line"] = [list(pts[0]), list(pts[1])]

    print("\n=== Duong 2 CHIEU? Ve MUI TEN chi huong xe duoc giam sat ===")
    print("    (diem 1 = duoi xe xuat phat, diem 2 = huong xe se vuot den do)")
    print("    Enter/Space khong click = duong 1 chieu, giam sat ca 2 chieu")
    arrow = _draw_points(
        frame, "Monitored direction",
        "MUI TEN: click diem 1 (phia xe xuat phat) roi diem 2 (huong xe chay)",
        2, arrow=True, optional=True,
    )
    if arrow is None:
        print("Huy calibration — vi pham se TAT (khong co vach dung).")
        return {}
    if arrow:
        from edge_node.core.contracts import Point

        direction = _direction_from_arrow(
            Point(*pts[0]), Point(*pts[1]), arrow[0], arrow[1],
        )
        calib["direction"] = direction
        print(f"    Huong giam sat: {direction} (xe di nguoc chieu se bi bo qua)")

    print("\n=== Ke VUNG DEN: click 2 diem goc trai-tren va phai-duoi cua den ===")
    print("    (Space/Enter de BO QUA neu khong muon vung den)")
    roi = _draw_points(
        frame, "Traffic light ROI",
        "VUNG DEN: click 2 diem bao quanh den giao thong",
        2,
    )
    if roi is not None:
        (x1, y1), (x2, y2) = roi
        calib["light_roi"] = [
            min(x1, x2), min(y1, y2),
            abs(x2 - x1), abs(y2 - y1),
        ]  # x, y, w, h

    return calib


def load_calibration(video_path: Path) -> dict:
    if CALIB_FILE.exists():
        try:
            data = json.loads(CALIB_FILE.read_text())
            return data.get(video_path.name, {})
        except Exception as exc:
            LOGGER.warning("calibration.json hong (%s) — se ke lai", exc)
    return {}


def save_calibration(video_path: Path, calib: dict) -> None:
    data = {}
    if CALIB_FILE.exists():
        try:
            data = json.loads(CALIB_FILE.read_text())
        except Exception:
            data = {}
    data[video_path.name] = calib
    CALIB_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIB_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    LOGGER.info("Calibration luu tai %s", CALIB_FILE)


# ────────────────────────────────────────────────────────────────────
# Pipeline
# ────────────────────────────────────────────────────────────────────

def find_default_video() -> Optional[Path]:
    video_dir = PROJECT_ROOT / "data" / "videos"
    if video_dir.is_dir():
        for ext in ("*.mp4", "*.MP4", "*.avi", "*.mkv"):
            files = sorted(video_dir.glob(ext))
            if files:
                return files[0]
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Test pipeline vi pham den do — khong gui server.",
    )
    p.add_argument("video", nargs="?", default=None,
                   help="Video input (mac dinh: file dau tien trong data/videos/)")
    p.add_argument("--stop-line", "-s", metavar="x1,y1,x2,y2", default=None,
                   help="Vach dung (de dang ghi de calibration)")
    p.add_argument("--light-roi", "-r", metavar="x,y,w,h", default=None,
                   help="Vung den giao thong")
    p.add_argument("--direction", "-d",
                   choices=["any", "positive_to_negative", "negative_to_positive"],
                   default=None,
                   help="Huong giam sat (mac dinh: dung calibration; duong 2 chieu "
                        "nen chi dinh 1 huong de bo qua xe chieu nguoc lai)")
    p.add_argument("--recalibrate", action="store_true",
                   help="Ke lai vach dung + vung den bang chuot")
    p.add_argument("--max-frames", "-n", type=int, default=None)
    p.add_argument("--loop", "-l", action="store_true")
    p.add_argument("--model", "-m", default=str(PROJECT_ROOT / "models/yolo26m_vehicle.pt"))
    p.add_argument("--confidence", "-c", type=float, default=0.35)
    p.add_argument("--device", default=None, help="cuda / cpu (mac dinh tu dong)")
    p.add_argument("--no-window", action="store_true", help="Khong mo cua so GUI")
    p.add_argument("--record", default=None, help="Luu video da annotate ra MP4")
    p.add_argument("--save-events", "-o", default=None, help="Luu events ra JSON")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def _parse_4ints(value: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Can 4 so: a,b,c,d")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def run(args) -> int:
    import cv2

    from edge_node.core.byte_tracker import ByteTrackerConfig, SupervisionByteTracker
    from edge_node.core.config import (
        RedStabilizerConfig, TripwireConfig, ViolationConfig,
        set_active_light_roi, set_active_tripwire,
    )
    from edge_node.core.contracts import CrossingDirection, Point
    from edge_node.core.detector import YoloDetector
    from edge_node.core.pipeline import RedLightViolationPipeline
    from edge_node.core.traffic_light_yolo import create_light_classifier
    from edge_node.core.video_io import OpenCVFrameSource
    from edge_node.core.violation_logic import RedLightStabilizer, ViolationDetector
    from edge_node.core.visualizer import LiveVisualizer
    from edge_node.settings import get_settings

    # ── Video ──
    video = Path(args.video) if args.video else find_default_video()
    if video is None:
        LOGGER.error("Khong tim thay video. Dat .mp4 vao data/videos/ hoac truyen duong dan.")
        return 1
    if not video.is_absolute():
        video = PROJECT_ROOT / video
    if not video.exists():
        LOGGER.error("Video khong ton tai: %s", video)
        return 1

    # ── Calibration: CLI > calibration.json > ke bang chuot ──
    calib = {} if args.recalibrate else load_calibration(video)

    stop_line = None
    if args.stop_line:
        x1, y1, x2, y2 = _parse_4ints(args.stop_line)
        stop_line = [(x1, y1), (x2, y2)]
    elif "stop_line" in calib:
        stop_line = [tuple(p) for p in calib["stop_line"]]

    light_roi = None
    if args.light_roi:
        light_roi = _parse_4ints(args.light_roi)
    elif "light_roi" in calib:
        light_roi = tuple(calib["light_roi"])

    # Huong giam sat: CLI > calibration.json > "any" (duong 1 chieu)
    direction = args.direction or calib.get("direction", "any")

    if stop_line is None and not args.no_window:
        calib = calibrate_interactive(video)
        stop_line = [tuple(p) for p in calib.get("stop_line", [])] or None
        light_roi = tuple(calib["light_roi"]) if "light_roi" in calib else None
        direction = calib.get("direction", "any")
        if stop_line:
            save_calibration(video, calib)

    if stop_line is None:
        LOGGER.warning("Khong co vach dung — vi pham se TAT.")

    # ── Components ──
    settings = get_settings()
    (sx1, sy1), (sx2, sy2) = stop_line if stop_line else ((0, 0), (0, 0))
    tripwire = (
        TripwireConfig(
            start=Point(sx1, sy1), end=Point(sx2, sy2),
            direction=CrossingDirection(direction),
        )
        if stop_line else None
    )
    set_active_tripwire(tripwire)
    if light_roi:
        set_active_light_roi(light_roi)

    detector = YoloDetector(args.model, confidence=args.confidence, device=args.device)
    tracker = SupervisionByteTracker(ByteTrackerConfig())
    classifier = create_light_classifier(roi=light_roi, device=args.device)
    stabilizer = RedLightStabilizer(RedStabilizerConfig(
        required_consecutive_frames=settings.red_stable_frames,
        switch_consecutive_frames=settings.red_switch_frames,
        min_confidence=settings.red_min_confidence,
    ))
    violation_detector = ViolationDetector(ViolationConfig(tripwire=tripwire))

    visualizer = LiveVisualizer(
        window_name=f"RLVD test - {video.name}",
        show=not args.no_window,
        record_path=args.record,
    )
    source = OpenCVFrameSource(str(video), max_frames=args.max_frames, loop=args.loop)

    LOGGER.info("Video:     %s", video)
    LOGGER.info("Vach dung: %s (dir=%s)", stop_line or "KHONG CO", direction)
    LOGGER.info("Vung den:  %s", light_roi or "KHONG CO")

    # ── Frame callback: chi de ve overlay + dem violation ──
    recent = {"events": [], "age": 0}
    # Mũi tên hướng giám sát vẽ lên live view (đường 2 chiều)
    dir_arrow = None
    if stop_line and direction != "any":
        dir_arrow = _arrow_from_direction(
            Point(*stop_line[0]), Point(*stop_line[1]), direction,
        )

    def on_frame(packet, light, stable_signal, detections, tracks, frame_events):
        if frame_events:
            for ev in frame_events:
                LOGGER.info(
                    "VI PHAM! frame=%d track=%d den=%s conf=%.2f tai=(%.0f,%.0f)",
                    ev.frame_index, ev.track_id, ev.light_state.value,
                    ev.light_confidence, ev.crossing_point.x, ev.crossing_point.y,
                )
            recent["events"] = list(frame_events)
            recent["age"] = 0
        if recent["events"] and recent["age"] > 90:
            recent["events"] = []
        recent["age"] += 1

        visualizer.update(
            packet.image,
            detections=detections, tracks=tracks,
            light=light, signal=stable_signal,
            violations=recent["events"],
            stop_line=(Point(*stop_line[0]), Point(*stop_line[1])) if stop_line else None,
            direction_arrow=dir_arrow,
            light_roi=light_roi,
        )

    # ── Chạy pipeline (KHÔNG outbox, KHÔNG sender) ──
    pipeline = RedLightViolationPipeline(
        detector=detector,
        tracker=tracker,
        light_classifier=classifier,
        stabilizer=stabilizer,
        violation_detector=violation_detector,
        logger=LOGGER,
        frame_callback=on_frame,
        outbox=None,  # test local — khong gui di dau ca
    )

    result = None
    try:
        result = pipeline.process(source, max_frames=args.max_frames)
    except KeyboardInterrupt:
        LOGGER.info("Thoat theo yeu cau nguoi dung.")
    finally:
        visualizer.close()

    events = list(result.violations) if result else []
    frames = result.frames_processed if result else 0
    LOGGER.info("=" * 50)
    LOGGER.info("Xong: %d frames, %d vi pham", frames, len(events))
    for i, ev in enumerate(events, 1):
        LOGGER.info(
            "  #%d frame=%-5d track=%-3d den=%-6s conf=%.2f pos=(%.0f,%.0f)",
            i, ev.frame_index, ev.track_id, ev.light_state.value,
            ev.light_confidence, ev.crossing_point.x, ev.crossing_point.y,
        )

    if args.save_events:
        out = Path(args.save_events)
        out.write_text(json.dumps([{
            "event_id": ev.event_id, "track_id": ev.track_id,
            "frame_index": ev.frame_index, "timestamp_ms": ev.timestamp_ms,
            "light_state": ev.light_state.value,
            "light_confidence": ev.light_confidence,
            "crossing_point": {"x": ev.crossing_point.x, "y": ev.crossing_point.y},
            "bbox": list(ev.bbox.as_xyxy()),
        } for ev in events], indent=2, ensure_ascii=False))
        LOGGER.info("Da luu %d events -> %s", len(events), out)

    return 0


if __name__ == "__main__":
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    for lib in ("ultralytics", "matplotlib", "PIL"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    sys.exit(run(args))
