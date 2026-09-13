"""Test cho tầng *track quality* bồi trên ByteTrack (2026-09-12).

Cover các mục trong ``tracker_update.md`` đã triển khai ở edge_node:

* mục 1/2/4 — quỹ đạo (trajectory) + vận tốc/hướng theo pixel/giây
* mục 3     — điểm neo ``crossing_point`` = bottom-center
* mục 5     — ``time_since_update`` thật (gap trước khi track được tìm lại)
* mục 6     — track confidence làm mượt EMA, tách khỏi detection confidence
* mục 7     — nhãn bỏ phiếu có trọng số (chống lật motorcycle↔vehicle)
* mục 8     — log vòng đời track (created/lost/refind) + cảnh báo ID switch
* mục 9     — gating theo khoảng cách + hướng (mặc định TẮT)
"""

from __future__ import annotations

import logging

import pytest

from edge_node.core.byte_tracker import (
    ByteTrackerConfig,
    SupervisionByteTracker,
    _vote_label,
)
from edge_node.core.contracts import (
    BoundingBox,
    Detection,
    MotionState,
    Track,
    TrackSample,
)
from edge_node.core.motion import TrackMotion, heading_label


def _point(x: float, y: float):
    from edge_node.core.contracts import Point

    return Point(x, y)


def _det(cx: float, cy: float, w: float, h: float, conf: float,
         label: str = "motorcycle") -> Detection:
    """Detection có TÂM box tại (cx, cy)."""
    return Detection(
        bbox=BoundingBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
        label=label,
        confidence=conf,
    )


def _tracker(**overrides) -> SupervisionByteTracker:
    cfg = ByteTrackerConfig(frame_rate=6.0, **overrides)
    return SupervisionByteTracker(cfg)


class TestCrossingPointAnchor:
    """Mục 3 — điểm neo phải là bottom-center (điểm xe chạm mặt đường)."""

    def test_crossing_point_is_bottom_center(self):
        track = Track(
            track_id=1,
            bbox=BoundingBox(100, 200, 160, 320),
            label="motorcycle",
            confidence=0.9,
            age=5,
            hits=5,
            time_since_update=0,
        )
        point = track.crossing_point
        assert point.x == pytest.approx(130.0)   # (100+160)/2
        assert point.y == pytest.approx(320.0)   # đáy box, không phải 260
        assert point == track.bbox.bottom_center

    def test_anchor_stable_when_bbox_stretches(self):
        """Xe nghiêng / bbox co giãn: bottom-center ổn định hơn center.

        Cùng một vị trí bánh xe nhưng box cao thêm 20px → center dịch 10px,
        bottom-center đứng yên.
        """
        short = BoundingBox(100, 240, 160, 320)
        tall = BoundingBox(100, 220, 160, 320)
        assert short.bottom_center.y == tall.bottom_center.y
        assert short.center.y != tall.center.y


class TestTrackMotion:
    """Mục 1/2/4 — quỹ đạo + vận tốc/hướng."""

    def test_velocity_px_per_second_at_3fps(self):
        """Xe đi 30px xuống dưới mỗi 333ms → vy ≈ +90 px/s (không phải 30)."""
        motion = TrackMotion(max_samples=8)
        for i in range(5):
            motion.push(TrackSample(
                timestamp_ms=i * 333.0,
                point=_point(500.0, 200.0 + 30.0 * i),
                width=40.0, height=60.0, detection_confidence=0.8,
            ))
        state = motion.motion()
        assert state.vy == pytest.approx(90.0, rel=0.05)
        assert state.vx == pytest.approx(0.0, abs=1e-6)
        assert state.speed == pytest.approx(90.0, rel=0.05)
        assert state.direction == "down"
        assert state.is_moving

    def test_direction_labels_use_image_axes(self):
        """Hệ toạ độ ảnh: y trỏ xuống → vy>0 là 'down', vy<0 là 'up'."""
        assert heading_label(0.0) == "right"
        assert heading_label(90.0) == "down"
        assert heading_label(180.0) == "left"
        assert heading_label(-90.0) == "up"
        assert heading_label(45.0) == "down_right"
        assert heading_label(-45.0) == "up_right"

    def test_stationary_track_not_moving(self):
        motion = TrackMotion()
        for i in range(4):
            motion.push(TrackSample(
                timestamp_ms=i * 333.0, point=_point(500.0, 400.0),
                width=40.0, height=60.0, detection_confidence=0.8,
            ))
        state = motion.motion()
        assert state.direction == "stationary"
        assert not state.is_moving

    def test_jitter_averaged_by_regression(self):
        """Bbox rung ±6px quanh đường thẳng: speed vẫn bám vận tốc thật.

        Sai phân 2 điểm liền kề sẽ thổi phồng vận tốc (~1.6x); hồi quy
        least-squares thì không.
        """
        motion = TrackMotion(max_samples=8)
        jitter = [0.0, 6.0, -6.0, 4.0, -4.0, 0.0]
        for i in range(6):
            motion.push(TrackSample(
                timestamp_ms=i * 333.0,
                point=_point(500.0 + jitter[i], 200.0 + 30.0 * i),
                width=40.0, height=60.0, detection_confidence=0.8,
            ))
        state = motion.motion()
        assert state.vy == pytest.approx(90.0, rel=0.10)
        assert abs(state.vx) < 12.0, f"jitter ngang thổi phồng vx: {state.vx}"

    def test_duplicate_timestamp_replaces_sample(self):
        """Cùng timestamp phải THAY mẫu, không nối thêm (dt=0 → speed vô cực)."""
        motion = TrackMotion()
        motion.push(TrackSample(1000.0, _point(10.0, 10.0), 20.0, 20.0, 0.8))
        motion.push(TrackSample(1000.0, _point(12.0, 12.0), 20.0, 20.0, 0.8))
        assert len(motion) == 1
        assert motion.motion().speed == 0.0

    def test_degenerate_timestamps_do_not_invent_velocity(self):
        """Nguồn không có timestamp phân biệt → không bịa vận tốc.

        ``push()`` gộp các mẫu trùng timestamp (nối thêm sẽ cho dt=0 → slope
        vô nghĩa), nên quỹ đạo còn 1 mẫu và motion() báo stationary thay vì
        một con số bịa. ``video_io`` luôn sinh timestamp tăng nghiêm ngặt
        (CAP_PROP_POS_MSEC hoặc frame_index/fps), đây là lưới an toàn.
        """
        motion = TrackMotion()
        for i in range(4):
            motion.push(TrackSample(0.0, _point(100.0, 100.0 + 10.0 * i),
                                    20.0, 20.0, 0.8))
        assert len(motion) == 1
        assert motion.motion().speed == 0.0

    def test_backwards_timestamp_resets_trajectory(self):
        """Timestamp thụt lùi (seek/restart nguồn) → bỏ quỹ đạo cũ."""
        motion = TrackMotion()
        for i in range(3):
            motion.push(TrackSample(i * 1000.0, _point(0.0, 10.0 * i),
                                    20.0, 20.0, 0.8))
        assert len(motion) == 3
        motion.push(TrackSample(0.0, _point(0.0, 0.0), 20.0, 20.0, 0.8))
        assert len(motion) == 1

    def test_predicted_point_extrapolates(self):
        motion = TrackMotion()
        for i in range(3):
            motion.push(TrackSample(i * 1000.0, _point(0.0, 100.0 * i),
                                    20.0, 20.0, 0.8))
        predicted = motion.predicted_point(3000.0)
        assert predicted is not None
        assert predicted.y == pytest.approx(300.0, rel=0.05)

    def test_predicted_point_needs_two_samples(self):
        motion = TrackMotion()
        motion.push(TrackSample(0.0, _point(0.0, 0.0), 20.0, 20.0, 0.8))
        assert motion.predicted_point(1000.0) is None

    def test_max_samples_validated(self):
        with pytest.raises(ValueError):
            TrackMotion(max_samples=1)


class TestLoopedSource:
    """Nguồn phát lặp (docker-compose chạy --loop): timestamp quay về 0.

    ``CAP_PROP_POS_MSEC`` thụt lùi trong khi ``frame_index`` vẫn tăng. Quỹ đạo
    phải reset để không hồi quy xuyên qua điểm gãy (sinh vận tốc ảo), còn
    ``age``/``hits`` vẫn đếm theo frame_index.
    """

    def test_timestamp_regression_resets_trajectory(self):
        tracker = _tracker()
        y = 200.0
        tracks = []
        for i in range(6):
            y += 30.0
            tracks = tracker.update([_det(500, y, 60, 90, 0.9)], i, i * 300.0)
        assert tracks
        tid = tracks[0].track_id
        assert tracks[0].motion.speed == pytest.approx(100.0, rel=0.15)

        # Xe đi tiếp (vị trí liên tục) nhưng timestamp nhảy về 0
        for i in range(6, 10):
            y += 30.0
            tracks = tracker.update([_det(500, y, 60, 90, 0.9)], i,
                                    (i - 6) * 300.0)
            same = [t for t in tracks if t.track_id == tid]
            assert same, "track bị mất khi nguồn loop"
            t = same[0]
            # Không vận tốc ảo (hồi quy xuyên điểm gãy sẽ cho con số khổng lồ)
            assert t.motion.speed < 500.0, f"vận tốc ảo: {t.motion.speed}"
            # age/hits đếm theo frame_index, không bị timestamp ảnh hưởng
            assert t.hits == i + 1
            assert t.age == i + 1

    def test_looped_position_change_spawns_new_id_without_phantom_speed(self):
        """Loop làm xe "nhảy" ngược vị trí → id mới, và không vận tốc ảo."""
        tracker = _tracker()
        speeds: list[float] = []
        for i in range(12):
            pos = (i % 6) * 30          # vị trí lặp lại 0..150
            ts = (i % 6) * 300.0        # timestamp lặp lại theo
            tracks = tracker.update([_det(500, 200 + pos, 60, 90, 0.9)], i, ts)
            speeds.extend(t.motion.speed for t in tracks)
        assert speeds
        assert max(speeds) < 500.0, f"vận tốc ảo khi nguồn loop: {max(speeds)}"


class TestLabelVoting:
    """Mục 7 — bỏ phiếu nhãn có trọng số."""

    def test_majority_wins_over_single_frame(self):
        votes = [
            ("motorcycle", 0.8), ("motorcycle", 0.7), ("vehicle", 0.9),
            ("motorcycle", 0.8), ("motorcycle", 0.75),
        ]
        assert _vote_label(votes) == "motorcycle"

    def test_confidence_weighted(self):
        """Một nhãn conf cao thắng nhiều nhãn conf thấp."""
        votes = [("bike", 0.2), ("bike", 0.2), ("motorcycle", 0.95)]
        assert _vote_label(votes) == "motorcycle"

    def test_tie_falls_back_to_most_recent(self):
        assert _vote_label([("car", 0.5), ("truck", 0.5)]) == "truck"

    def test_empty_votes(self):
        assert _vote_label([]) == "vehicle"

    def test_tracker_label_stable_through_one_bad_frame(self):
        """Xe máy bị nhận nhầm 1 frame 'car' → label output vẫn 'motorcycle'."""
        tracker = _tracker(label_vote_window=5)
        labels: list[str] = []
        y = 200.0
        for i in range(8):
            y += 12.0
            label = "car" if i == 5 else "motorcycle"
            tracks = tracker.update(
                [_det(500, y, 40, 60, conf=0.8, label=label)], i, i * 166.0
            )
            if tracks:
                labels.append(tracks[0].label)
        assert labels, "không có track nào được tạo"
        assert "car" not in labels, f"label bị lật theo 1 frame nhiễu: {labels}"
        assert labels[-1] == "motorcycle"


class TestConfidenceSmoothing:
    """Mục 6 — EMA track confidence vs detection confidence thô."""

    def test_ema_absorbs_single_low_confidence_frame(self):
        tracker = _tracker(confidence_ema_alpha=0.35)
        confs = [0.90, 0.88, 0.31, 0.90]
        seen: list[tuple[float, float]] = []
        y = 300.0
        for i, conf in enumerate(confs):
            y += 10.0
            tracks = tracker.update([_det(500, y, 40, 60, conf)], i, i * 333.0)
            for t in tracks:
                seen.append((t.detection_confidence, t.confidence))
        assert seen, "track không tồn tại"
        # Frame conf 0.31: detection_confidence phản ánh đúng, EMA thì không tụt thảm
        low = [pair for pair in seen if pair[0] == pytest.approx(0.31, abs=0.02)]
        assert low, "không bắt được frame conf thấp"
        det_conf, track_conf = low[0]
        assert det_conf < 0.35
        assert track_conf > det_conf + 0.2, (
            f"track confidence tụt theo detection: {track_conf} vs {det_conf}"
        )

    def test_alpha_one_disables_smoothing(self):
        tracker = _tracker(confidence_ema_alpha=1.0)
        y = 300.0
        last = None
        for i, conf in enumerate([0.9, 0.4]):
            y += 10.0
            tracks = tracker.update([_det(500, y, 40, 60, conf)], i, i * 333.0)
            if tracks:
                last = tracks[0]
        assert last is not None
        assert last.confidence == pytest.approx(last.detection_confidence, abs=1e-3)

    def test_invalid_alpha_rejected(self):
        with pytest.raises(ValueError):
            SupervisionByteTracker(ByteTrackerConfig(confidence_ema_alpha=0.0))
        with pytest.raises(ValueError):
            SupervisionByteTracker(ByteTrackerConfig(confidence_ema_alpha=1.5))


class TestTimeSinceUpdate:
    """Mục 5 — ``time_since_update`` phải phản ánh gap thật, không hard-code 0."""

    def test_zero_on_matched_frame(self):
        tracker = _tracker()
        y = 200.0
        for i in range(4):
            y += 10.0
            tracks = tracker.update([_det(500, y, 40, 60, 0.9)], i, i * 166.0)
        assert tracks
        assert tracks[0].time_since_update == 0
        assert tracks[0].metadata["gap_frames"] == 0

    def test_positive_after_detection_gap(self):
        """Xe bị che 2 frame rồi xuất hiện lại: cùng id, time_since_update > 0.

        ByteTrack chỉ output track ĐÃ match trong frame, nên giá trị >0 là dấu
        hiệu "track vừa được tìm lại sau khi mất detection" — thông tin
        downstream cần để không kết luận vi phạm ngay frame đó.
        """
        tracker = _tracker()
        tid = None
        refind_track = None
        y = 200.0
        for i in range(3):
            y += 10.0
            tracks = tracker.update([_det(500, y, 40, 60, 0.9)], i, i * 333.0)
            if tracks:
                tid = tracks[0].track_id
        # 3 frame xe biến mất hoàn toàn (bị che)
        for i in range(3, 6):
            tracker.update([], i, i * 333.0)
        # Xe xuất hiện lại ở vị trí tiếp nối
        for i in range(6, 10):
            y += 10.0
            tracks = tracker.update([_det(500, y, 40, 60, 0.9)], i, i * 333.0)
            for t in tracks:
                if t.track_id == tid and t.time_since_update > 0:
                    refind_track = t
        assert tid is not None, "track gốc chưa từng tồn tại"
        assert refind_track is not None, (
            "track được tìm lại nhưng time_since_update vẫn 0 (hard-code cũ)"
        )
        assert refind_track.metadata["refinds"] >= 1
        assert refind_track.metadata["missed_frames_total"] >= 1
        # Gap thật là 3 frame
        assert refind_track.time_since_update >= 2

    def test_trajectory_reset_after_gap(self):
        """Quỹ đạo đứt đoạn → không hồi quy xuyên qua gap (vận tốc ảo)."""
        tracker = _tracker()
        tracks: list[Track] = []
        y = 200.0
        for i in range(3):
            y += 10.0
            tracks = tracker.update([_det(500, y, 40, 60, 0.9)], i, i * 333.0)
        assert tracks and len(tracks[0].trajectory) >= 2
        for i in range(3, 6):
            tracker.update([], i, i * 333.0)
        y += 10.0
        tracks = tracker.update([_det(500, y, 40, 60, 0.9)], 6, 6 * 333.0)
        if tracks:
            # Sau gap, quỹ đạo bắt đầu lại từ 1 mẫu → chưa có vận tốc
            assert len(tracks[0].trajectory) <= 2


class TestTrackTrajectoryOutput:
    """Track output phải mang quỹ đạo + motion để downstream dùng."""

    def test_track_carries_trajectory_and_motion(self):
        tracker = _tracker(trajectory_max_samples=6)
        tracks: list[Track] = []
        y = 200.0
        for i in range(5):
            y += 20.0
            tracks = tracker.update([_det(500, y, 40, 60, 0.9)], i, i * 333.0)
        assert tracks, "không có track"
        track = tracks[0]
        assert isinstance(track.motion, MotionState)
        assert len(track.trajectory) >= 2
        assert track.motion.speed > 0.0
        assert track.motion.direction in {
            "down", "down_left", "down_right", "up", "up_left", "up_right",
            "left", "right", "stationary", "unknown",
        }
        assert track.metadata["direction"] == track.motion.direction
        assert track.metadata["velocity_px_per_s"] > 0.0
        # Vận tốc phải theo GIÂY: 20px/333ms ≈ 60 px/s
        assert track.metadata["velocity_px_per_s"] == pytest.approx(60.0, rel=0.15)

    def test_trajectory_uses_bottom_center_anchor(self):
        tracker = _tracker()
        tracks: list[Track] = []
        for i in range(3):
            tracks = tracker.update([_det(500, 300.0 + i * 10, 40, 60, 0.9)],
                                    i, i * 333.0)
        assert tracks and tracks[0].trajectory
        sample = tracks[0].trajectory[-1]
        assert sample.point == pytest.approx(
            tracks[0].bbox.bottom_center, rel=1e-3, abs=1.0
        )

    def test_trajectory_capped_at_max_samples(self):
        tracker = _tracker(trajectory_max_samples=4)
        tracks: list[Track] = []
        for i in range(12):
            tracks = tracker.update([_det(500, 200.0 + i * 10, 40, 60, 0.9)],
                                    i, i * 333.0)
        assert tracks
        assert len(tracks[0].trajectory) <= 4

    def test_trajectory_json_serialisable(self):
        """Quỹ đạo phải đưa được vào evidence/metadata (JSON-safe)."""
        import json

        motion = TrackMotion()
        motion.push(TrackSample(100.0, _point(10.0, 20.0), 30.0, 40.0, 0.7))
        payload = json.dumps(motion.as_dicts())
        assert '"matched"' in payload


class TestIdSwitchLogging:
    """Mục 8 — vòng đời track phải log được để rà soát ID switch."""

    def test_logs_created_lost_refind(self, caplog):
        tracker = _tracker()
        y = 200.0
        with caplog.at_level(logging.DEBUG, logger="edge_node.core.byte_tracker"):
            for i in range(3):
                y += 10.0
                tracker.update([_det(500, y, 40, 60, 0.9)], i, i * 333.0)
            for i in range(3, 6):
                tracker.update([], i, i * 333.0)
            y += 10.0
            tracker.update([_det(500, y, 40, 60, 0.9)], 6, 6 * 333.0)
        text = caplog.text
        assert "created" in text
        assert "lost" in text

    def test_suspicious_refind_warns(self, caplog):
        """Track tìm lại ở vị trí xa bất thường → WARNING (nghi ngờ ID switch).

        Gọi thẳng ``_log_refind``: trong matching thật, IoU=0 cho cost 1.0 >
        ngưỡng 0.85 nên cặp nhảy xa hàng trăm px gần như không bao giờ được
        ghép — nhưng cặp nhảy 1-2 đường chéo box (đủ IoU thấp nhưng qua
        ngưỡng) thì có, và đó chính là case cần cảnh báo.
        """
        from edge_node.core.byte_tracker import _TrackQuality

        tracker = _tracker()
        quality = _TrackQuality(
            first_frame=0, last_frame=2,
            bbox=BoundingBox(480, 170, 520, 230),
            motion=TrackMotion(4),
        )
        quality.motion.push(TrackSample(
            666.0, _point(500.0, 230.0), 40.0, 60.0, 0.9,
        ))
        far_box = BoundingBox(480, 470, 520, 530)   # dịch 300px = 4.2x đường chéo
        with caplog.at_level(logging.DEBUG, logger="edge_node.core.byte_tracker"):
            tracker._log_refind(9, quality, far_box, 0.9, 6, 3)
        assert "SUSPICIOUS id-switch?" in caplog.text
        assert tracker._byte_track.suspicious_matches == 1

    def test_normal_refind_only_debug(self, caplog):
        """Refind ở khoảng cách bình thường → DEBUG, không phải WARNING."""
        from edge_node.core.byte_tracker import _TrackQuality

        tracker = _tracker()
        quality = _TrackQuality(
            first_frame=0, last_frame=2,
            bbox=BoundingBox(480, 170, 520, 230),
            motion=TrackMotion(4),
        )
        quality.motion.push(TrackSample(
            666.0, _point(500.0, 230.0), 40.0, 60.0, 0.9,
        ))
        near_box = BoundingBox(480, 230, 520, 290)   # dịch 60px ≈ 0.8x đường chéo
        with caplog.at_level(logging.DEBUG, logger="edge_node.core.byte_tracker"):
            tracker._log_refind(9, quality, near_box, 0.9, 6, 3)
        assert "refind" in caplog.text
        assert "SUSPICIOUS" not in caplog.text
        assert tracker._byte_track.suspicious_matches == 0

    def test_periodic_summary_when_enabled(self, caplog):
        tracker = _tracker(id_switch_log_interval=5)
        with caplog.at_level(logging.WARNING, logger="edge_node.core.byte_tracker"):
            for i in range(10):
                tracker.update([_det(500, 200.0 + i * 10, 40, 60, 0.9)],
                               i, i * 333.0)
        assert "track quality @frame=" in caplog.text


class TestGating:
    """Mục 9 — gating mặc định TẮT, bật lên thì chặn association nhảy xa."""

    def test_gate_disabled_by_default(self):
        cfg = ByteTrackerConfig()
        assert cfg.gate_enabled is False

    def test_distance_gate_blocks_big_box_far_center(self):
        """Gate khoảng cách chặn detection có TÂM lệch xa box track.

        HÌNH HỌC THẬT (đã scan, 2026-09-12): hai box CÙNG cỡ mà IoU > 0 thì
        khoảng cách hai tâm không bao giờ vượt đường chéo box (tỉ lệ tối đa
        đúng bằng 1.0). Nên với ``gate_distance_factor >= 1.0`` gate khoảng
        cách KHÔNG chặn được cặp nào cùng cỡ — nó chỉ có tác dụng khi
        detection là box lớn hơn nhiều (xe tải trùm lên xe máy: IoU > 0 nhưng
        tâm lệch 1.77× đường chéo box xe máy), hoặc khi nới
        ``minimum_matching_threshold`` cho adaptive matching (mục 1).

        Test dựng đúng case đó: track xe máy 40×60, frame sau xuất hiện det
        100×150 chồng lấn một phần với tâm lệch ~127px.
        """
        loose = dict(minimum_matching_threshold=0.99,
                     unconfirmed_match_threshold=0.99)

        def run(gate: bool, factor: float) -> int:
            tracker = _tracker(gate_enabled=gate, gate_distance_factor=factor,
                               **loose)
            y = 200.0
            for i in range(6):
                y += 10.0
                tracker.update([_det(500, y, 40, 60, 0.9)], i, i * 333.0)
            # Box lớn chồng lấn một phần, tâm lệch (+71, +106) px
            tracker.update(
                [_det(500 + 71, y + 106, 100, 150, 0.9, label="truck")],
                6, 6 * 333.0,
            )
            return tracker._byte_track.gate_rejections

        assert run(False, 1.5) == 0
        assert run(True, 1.5) >= 1, "gate khoảng cách không chặn tâm lệch xa"
        # Factor quá rộng (2.5× đường chéo) thì không chặn nữa — knob có tác dụng
        assert run(True, 2.5) == 0

    def test_direction_gate_blocks_reversed_match(self):
        """Gate hướng: từ chối ghép detection ngược hẳn hướng xe đang đi."""
        def run(gate: bool):
            tracker = _tracker(
                gate_enabled=gate,
                minimum_matching_threshold=0.95,
                unconfirmed_match_threshold=0.95,
                gate_distance_factor=50.0,   # chỉ xét gate HƯỚNG
            )
            y = 200.0
            for i in range(6):
                y += 30.0
                tracker.update([_det(500, y, 40, 60, 0.9)], i, i * 333.0)
            # Detection dịch NGƯỢC hướng 30px trong khi Kalman dự đoán đi xuống
            tracker.update([_det(500, y - 30.0, 40, 60, 0.9)], 6, 6 * 333.0)
            return tracker._byte_track.gate_rejections

        assert run(False) == 0
        assert run(True) >= 1, "gate hướng không chặn cặp ngược chiều"

    def test_gate_keeps_normal_motion(self):
        """Gate không được chặn chuyển động bình thường (xe đi 24px/frame)."""
        tracker = _tracker(gate_enabled=True, gate_distance_factor=2.0)
        ids = []
        y = 200.0
        for i in range(12):
            y += 24.0
            tracks = tracker.update([_det(500, y, 40, 60, 0.9)], i, i * 333.0)
            ids.append([t.track_id for t in tracks])
        flat = [tid for row in ids[2:] for tid in row]
        assert flat
        main_id = flat[0]
        gaps = sum(1 for row in ids[3:] if main_id not in row)
        assert gaps == 0, f"gate làm đứt track bình thường: {ids}"
        assert tracker._byte_track.gate_rejections == 0


class TestTrackContractCompat:
    """Track mới phải tương thích ngược (field mới có default)."""

    def test_old_constructor_still_works(self):
        track = Track(
            track_id=1,
            bbox=BoundingBox(0, 0, 10, 10),
            label="car",
            confidence=0.5,
            age=1,
            hits=1,
            time_since_update=0,
        )
        assert track.motion.speed == 0.0
        assert track.trajectory == ()
        assert track.detection_confidence == 0.0
        assert track.metadata == {}

    def test_metadata_is_mapping(self):
        """Metadata của track phải JSON-safe (đưa vào payload outbox)."""
        import json

        tracker = _tracker()
        tracks: list[Track] = []
        for i in range(3):
            tracks = tracker.update([_det(500, 200.0 + i * 10, 40, 60, 0.9)],
                                    i, i * 333.0)
        assert tracks
        json.dumps(dict(tracks[0].metadata))
