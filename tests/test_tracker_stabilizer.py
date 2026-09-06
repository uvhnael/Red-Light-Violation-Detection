"""Test cho các cải thiện tracking + stabilizer (2026-09-05).

Cover:
* ``scaled_stabilizer_config``: quy đổi giây → frame đúng theo FPS nguồn
  — hành vi debounce đồng nhất ở camera 3fps như 30fps.
* ``SupervisionByteTracker.update`` dùng đường ``update_with_tensors``:
  track không bị rơi khi box Kalman trễ nhịp so với detection (bug cũ
  của ``update_with_detections`` re-match IoU ≥ 0.5 cứng).
* Config tracker thân thiện xe máy: detection conf thấp (0.20-0.35)
  vẫn tạo được track.
"""

from __future__ import annotations

import pytest

from edge_node.core.byte_tracker import ByteTrackerConfig, SupervisionByteTracker
from edge_node.core.config import scaled_stabilizer_config
from edge_node.core.contracts import BoundingBox, Detection, LightState
from edge_node.core.violation_logic import RedLightStabilizer


class TestScaledStabilizerConfig:
    def test_seconds_scale_with_fps(self):
        """Cùng tham số giây → số frame tỉ lệ theo FPS."""
        cfg_3fps = scaled_stabilizer_config((0.4, 0.8, 0.8), fps=3.0, min_confidence=0.55)
        cfg_30fps = scaled_stabilizer_config((0.4, 0.8, 0.8), fps=30.0, min_confidence=0.55)

        # 0.4s * 3fps = 1.2 -> 1 frame; 0.4s * 30fps = 12 frame
        assert cfg_3fps.required_consecutive_frames == 1
        assert cfg_30fps.required_consecutive_frames == 12
        # 0.8s * 3fps = 2.4 -> 2 frame; 0.8s * 30fps = 24 frame
        assert cfg_3fps.switch_consecutive_frames == 2
        assert cfg_30fps.switch_consecutive_frames == 24
        assert cfg_3fps.unknown_tolerance_frames == 2
        assert cfg_30fps.unknown_tolerance_frames == 24
        assert cfg_3fps.min_confidence == 0.55

    def test_switch_delay_constant_in_seconds(self):
        """Độ trễ chuyển trạng thái (giây) ~constant ở mọi FPS."""
        for fps in (3.0, 6.0, 10.0, 30.0):
            cfg = scaled_stabilizer_config((0.4, 0.8, 0.8), fps=fps, min_confidence=0.55)
            switch_seconds = cfg.switch_consecutive_frames / fps
            # Sai số làm tròn tối đa 1 frame
            assert abs(switch_seconds - 0.8) <= 1.0 / fps, (
                f"fps={fps}: switch delay {switch_seconds:.3f}s lệch quá 1 frame"
            )

    def test_min_values_clamped(self):
        """Giá trị nhỏ không bao giờ xuống dưới 1 frame (tránh deadlock)."""
        cfg = scaled_stabilizer_config((0.0, 0.0, 0.0), fps=3.0, min_confidence=0.5)
        assert cfg.required_consecutive_frames >= 1
        assert cfg.switch_consecutive_frames >= 1
        assert cfg.unknown_tolerance_frames >= 0

    def test_invalid_fps_rejected(self):
        with pytest.raises(ValueError):
            scaled_stabilizer_config((0.4, 0.8, 0.8), fps=0.0, min_confidence=0.5)


class TestStabilizerSecondsBehaviour:
    """End-to-end: replay chuỗi observation qua stabilizer theo giây."""

    def _obs(self, state: LightState, conf: float = 0.9):
        from edge_node.core.contracts import LightObservation

        return LightObservation(state=state, confidence=conf, source="test")

    def test_switch_delay_proportional_to_fps(self):
        """Ở fps thấp, số frame cần để chuyển đèn ít hơn fps cao (cùng giây)."""
        def switch_frame_at(fps: float) -> int:
            cfg = scaled_stabilizer_config((0.4, 0.8, 0.8), fps=fps, min_confidence=0.55)
            stab = RedLightStabilizer(cfg)
            frame = 0
            # RED ổn định lâu
            for i in range(30):
                stab.update(self._obs(LightState.RED), i)
            # GREEN bắt đầu từ frame 30 — đếm đến khi signal đổi
            for i in range(30, 200):
                sig = stab.update(self._obs(LightState.GREEN), i)
                if sig.state == LightState.GREEN:
                    return i - 30
                frame = i
            return frame

        delay_3fps = switch_frame_at(3.0)   # ~2 frame = 0.67s
        delay_30fps = switch_frame_at(30.0)  # ~24 frame = 0.8s
        # Cùng độ trễ giây: frame ở 3fps ít hơn NHIỀU lần
        assert delay_3fps <= 3
        assert delay_30fps >= 20
        # Quan trọng nhất: tính bằng giây, 3fps KHÔNG chậm hơn 30fps
        assert delay_3fps / 3.0 <= delay_30fps / 30.0 + 0.4


class TestByteTrackerTensorPath:
    """Tracker qua update_with_tensors giữ track mượt hơn đường cũ."""

    def _det(self, cx: float, cy: float, w: float, h: float, conf: float) -> Detection:
        return Detection(
            bbox=BoundingBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
            label="bike",
            confidence=conf,
        )

    def test_low_confidence_bike_still_tracked(self):
        """Xe máy conf 0.22 (dưới ngưỡng activation cũ 0.30) vẫn track được."""
        tracker = SupervisionByteTracker(
            ByteTrackerConfig(frame_rate=6.0)
        )
        # Frame 0-3: xe máy conf thấp xuất hiện liên tục
        seen_any = False
        for i in range(6):
            dets = [self._det(500, 400, 40, 60, conf=0.22)]
            tracks = tracker.update(dets, i, i * 166.0)
            if tracks:
                seen_any = True
        assert seen_any, "xe máy conf 0.22 không bao giờ được track (activation threshold quá cao)"

    def test_fast_bike_no_flicker(self):
        """Xe máy di chuyển nhanh (24px/frame @ 6fps): track không biến mất giữa chừng.

        Bug cũ: update_with_detections re-match IoU >= 0.5 cứng — Kalman box
        trễ 1 nhịp → IoU < 0.5 → track rơi khỏi output dù ByteTrack còn giữ.
        """
        tracker = SupervisionByteTracker(ByteTrackerConfig(frame_rate=6.0))
        track_ids = []
        y = 300.0
        for i in range(20):
            y += 24.0  # nhanh: 24 px/frame
            dets = [self._det(500, y, 40, 60, conf=0.8)]
            tracks = tracker.update(dets, i, i * 166.0)
            track_ids.append([t.track_id for t in tracks])
        # Track PHẢI liên tục sau khi lock (không nhấp nháy biến mất)
        flat = [tid for ids in track_ids[2:] for tid in ids]
        assert flat, "không có track nào sau frame 2"
        main_id = flat[0]
        gaps = sum(1 for ids in track_ids[3:] if main_id not in ids)
        assert gaps == 0, f"track {main_id} biến mất {gaps} lần giữa chừng (flicker)"

    def test_two_objects_keep_distinct_ids(self):
        """Hai xe máy gần nhau không bị swap/merge ID."""
        tracker = SupervisionByteTracker(ByteTrackerConfig(frame_rate=6.0))
        ids_a: set[int] = set()
        ids_b: set[int] = set()
        for i in range(15):
            dets = [
                self._det(400, 500 + i * 10, 40, 60, conf=0.8),   # xe A
                self._det(460, 500 + i * 10, 40, 60, conf=0.8),   # xe B cạnh nhau
            ]
            tracks = tracker.update(dets, i, i * 166.0)
            by_cx = sorted(tracks, key=lambda t: t.bbox.center.x)
            if len(by_cx) == 2:
                ids_a.add(by_cx[0].track_id)
                ids_b.add(by_cx[1].track_id)
        assert len(ids_a) == 1, f"xe A bị nhảy ID: {ids_a}"
        assert len(ids_b) == 1, f"xe B bị nhảy ID: {ids_b}"
        assert ids_a != ids_b, "2 xe bị gán chung 1 track ID"

    def test_config_defaults_motorbike_friendly(self):
        """Config mặc định phải là bộ số tối ưu xe máy đã chọn."""
        cfg = ByteTrackerConfig()
        assert cfg.track_activation_threshold == 0.10
        assert cfg.minimum_matching_threshold == 0.85
        assert cfg.unconfirmed_match_threshold == 0.85
        assert cfg.min_detection_confidence == 0.15
        assert cfg.lost_track_buffer == 30

    def test_empty_detections_no_crash(self):
        tracker = SupervisionByteTracker(ByteTrackerConfig(frame_rate=6.0))
        assert tracker.update([], 0, 0.0) == []
        tracks = tracker.update(
            [self._det(500, 400, 40, 60, conf=0.9)], 1, 166.0
        )
        assert len(tracks) >= 0  # không crash, kết quả hợp lệ
