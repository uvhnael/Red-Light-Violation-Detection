"""Tracker ByteTrack dựa trên thư viện *supervision* — tối ưu xe máy.

Thay thế tracker greedy-IoU đơn giản bằng cơ chế ghép nối hai giai đoạn
của ByteTrack (ưu tiên detection confidence cao trước, rồi đến các box
confidence thấp).

Bản fork nhẹ (``_TunedByteTrack``) chỉ mở một điểm supervision hard-code:
ngưỡng 0.7 (fused cost) cho khối đối chiếu track-chưa-xác-nhận — xe máy
chạy nhanh + confidence trung bình không bao giờ vượt qua frame thứ 2,
track không activate được và xe máy nhấp nháy/không hiện trên màn hình.

Cài đặt:  pip install supervision
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:
    import supervision as sv
    from supervision.tracker.byte_tracker import matching
    from supervision.tracker.byte_tracker.core import (
        joint_tracks,
        remove_duplicate_tracks,
        sub_tracks,
    )
    from supervision.tracker.byte_tracker.single_object_track import (
        STrack,
        TrackState,
    )
except ImportError as _exc:
    raise ImportError(
        "Cần package 'supervision' để chạy ByteTrack. "
        "Cài bằng lệnh: pip install supervision"
    ) from _exc

from edge_node.core.contracts import BoundingBox, Detection, Track

LOGGER = logging.getLogger(__name__)


def _real_bytrack_class():
    """Lấy class ByteTrack THẬT, bỏ qua deprecated proxy của supervision.

    Từ supervision 0.28 ``sv.ByteTrack`` là một ``_DeprecatedProxy`` quanh
    class gốc — subclass thẳng proxy sẽ chết lúc ``__init__`` (proxy nhận
    đối số riêng). Version cũ không có proxy thì trả về class luôn.
    """
    cls = sv.ByteTrack
    proxy_cfg = getattr(cls, "_DeprecatedProxy__config", None)
    if proxy_cfg is not None and getattr(proxy_cfg, "obj", None) is not None:
        return proxy_cfg.obj
    return cls


@dataclass(frozen=True)
class ByteTrackerConfig:
    """Tuning knobs cho ByteTrack — tối ưu xe máy, camera 3-10 fps.

    LƯU Ý SEMANTICS (dễ hiểu ngược!): association стадии 1 của supervision
    dùng ``fuse_score`` — cost = 1 - IoU * conf, chỉ match khi cost <=
    ``minimum_matching_threshold``. Nghĩa là threshold CAO = match LỎNG HƠN
    (chấp nhận cặp IoU * conf nhỏ hơn), ngược với直觉 "threshold cao = chặt".

    * ``frame_rate`` PHẢI là FPS thật của nguồn — ``lost_track_buffer``
      được ByteTrack quy đổi thành giây qua giá trị này (max_time_lost =
      frame_rate / 30 * lost_track_buffer frame). Hard-code 30 như trước
      khiến camera 3 fps giữ zombie track tận 10 giây → ID steal giữa xe.
    * ``track_activation_threshold`` 0.10 + ``min_detection_confidence``
      0.15: xe máy nhỏ/bị che có conf 0.2-0.4; supervision tạo track mới
      khi conf >= activation + 0.1, ngưỡng cao làm mất dấu chúng hoàn toàn.
    * ``minimum_matching_threshold`` 0.85: với fuse_score, chấp nhận cặp
      có IoU * conf >= 0.15 — xe máy chạy nhanh (IoU frame-qua-frame thấp
      do Kalman chưa học kịp vận tốc) vẫn giữ được track.
    * ``unconfirmed_match_threshold`` 0.85 (mặc định cũ hard-code 0.7):
      khối match track-chưa-xác-nhận — xe máy conf trung bình cần ngưỡng
      lỏng hơn để activate ở frame thứ 2.
    * ``lost_track_buffer`` 30: dung sai che khuất 1 GIÂY (quy đổi theo
      ``frame_rate``), giống nhau ở mọi camera.
    """

    track_activation_threshold: float = 0.10    # track mới cần conf >= activation + 0.1
    lost_track_buffer: int = 30                 # dung sai che khuất ~1s (scale theo frame_rate)
    minimum_matching_threshold: float = 0.85    # fused cost — CAO = lỏng (xem docstring)
    unconfirmed_match_threshold: float = 0.85    # khối unconfirmed (supervision hard-code 0.7)
    frame_rate: float = 30.0                     # FPS NGUỒN THẬT — caller phải truyền
    minimum_consecutive_frames: int = 1
    min_detection_confidence: float = 0.15       # không lọc mất detection yếu của xe máy


class _TunedByteTrack(_real_bytrack_class()):
    """ByteTrack với ngưỡng match unconfirmed-track cấu hình được.

    Fork toàn bộ ``update_with_tensors`` chỉ để thay MỘT hằng số
    (``thresh=0.7`` của khối unconfirmed) bằng ``unconfirmed_match_threshold``.
    Logic ByteTrack gốc giữ nguyên từng dòng. Khi nâng supervision (pin
    <0.29) cần đối chiếu lại method gốc với bản sao ở đây.
    """

    def __init__(
        self,
        *,
        unconfirmed_match_threshold: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.unconfirmed_match_threshold = unconfirmed_match_threshold

    def update_with_tensors(self, tensors: np.ndarray) -> list[STrack]:
        """Bản sao y hệt ``sv.ByteTrack.update_with_tensors`` trừ 1 ngưỡng."""
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = tensors[:, 4]
        bboxes = tensors[:, :4]

        remain_inds = scores > self.track_activation_threshold
        inds_low = scores > 0.1
        inds_high = scores < self.track_activation_threshold

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(
                    STrack.tlbr_to_tlwh(tlbr),
                    score_keep,
                    self.minimum_consecutive_frames,
                    self.shared_kalman,
                    self.internal_id_counter,
                    self.external_id_counter,
                )
                for (tlbr, score_keep) in zip(dets, scores_keep)
            ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks: list[STrack] = []

        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_tracks(tracked_stracks, self.lost_tracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool, self.shared_kalman)
        dists = matching.iou_distance(strack_pool, detections)

        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.minimum_matching_threshold
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(
                    STrack.tlbr_to_tlwh(tlbr),
                    score_second,
                    self.minimum_consecutive_frames,
                    self.shared_kalman,
                    self.internal_id_counter,
                    self.external_id_counter,
                )
                for (tlbr, score_second) in zip(dets_second, scores_second)
            ]
        else:
            detections_second = []
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _u_detection_second = matching.linear_assignment(
            dists, thresh=0.5
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.state = TrackState.Lost
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)

        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=self.unconfirmed_match_threshold  # ← điểm fork duy nhất
        )
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.state = TrackState.Removed
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_tracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.state = TrackState.Removed
                removed_stracks.append(track)

        self.tracked_tracks = [
            t for t in self.tracked_tracks if t.state == TrackState.Tracked
        ]
        self.tracked_tracks = joint_tracks(self.tracked_tracks, activated_starcks)
        self.tracked_tracks = joint_tracks(self.tracked_tracks, refind_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks = removed_stracks
        self.tracked_tracks, self.lost_tracks = remove_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks
        )
        output_stracks = [track for track in self.tracked_tracks if track.is_activated]

        return output_stracks


class SupervisionByteTracker:
    """MultiObjectTracker backed by (tuned) supervision.ByteTrack."""

    VEHICLE_LABELS: frozenset[str] = frozenset({
        "car", "truck", "bus", "motorcycle", "bicycle", "vehicle",
    })

    def __init__(self, config: ByteTrackerConfig | None = None) -> None:
        cfg = config or ByteTrackerConfig()
        self._min_confidence = cfg.min_detection_confidence
        self._byte_track = _TunedByteTrack(
            track_activation_threshold=cfg.track_activation_threshold,
            lost_track_buffer=cfg.lost_track_buffer,
            minimum_matching_threshold=cfg.minimum_matching_threshold,
            frame_rate=cfg.frame_rate,
            minimum_consecutive_frames=cfg.minimum_consecutive_frames,
            unconfirmed_match_threshold=cfg.unconfirmed_match_threshold,
        )
        # Internal bookkeeping – supervision resets tracker_id on each call,
        # so we keep a hits / age counter per ID ourselves.
        self._hits: dict[int, int] = {}
        self._ages: dict[int, int] = {}
        self._labels: dict[int, str] = {}
        self._frame_count = 0

    # ------------------------------------------------------------------
    # MultiObjectTracker protocol
    # ------------------------------------------------------------------
    def update(
        self,
        detections: Sequence[Detection],
        frame_index: int,
        timestamp_ms: float,
    ) -> list[Track]:
        del timestamp_ms  # unused
        self._frame_count = frame_index

        filtered = [
            d for d in detections
            if d.confidence >= self._min_confidence
        ]

        if filtered:
            xyxy = np.array(
                [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in filtered],
                dtype=np.float32,
            )
            confidence = np.array(
                [d.confidence for d in filtered], dtype=np.float32,
            )
            tensors = np.hstack((xyxy, confidence[:, np.newaxis]))
        else:
            tensors = np.zeros((0, 5), dtype=np.float32)

        # Gọi thẳng update_with_tensors thay vì update_with_detections:
        # hàm của supervision re-match track ↔ detection bằng IoU ngưỡng 0.5
        # CỨNG sau khi Kalman đã khớp xong, làm rơi mất track của target
        # nhanh (box Kalman trễ 1 nhịp so với detection) khỏi output —
        # biểu hiện là track nhấp nháy biến mất trên màn hình dù ByteTrack
        # vẫn còn giữ track đó bên trong.
        tracks = self._byte_track.update_with_tensors(tensors=tensors)

        labels_arr = [d.label for d in filtered]
        result: list[Track] = []

        for track_obj in tracks:
            tid = int(track_obj.external_track_id)
            self._hits[tid] = self._hits.get(tid, 0) + 1
            self._ages[tid] = self._ages.get(tid, 0) + 1

            box = track_obj.tlbr
            best_label = self._resolve_label(box, filtered, labels_arr)
            self._labels[tid] = best_label

            result.append(
                Track(
                    track_id=tid,
                    bbox=BoundingBox(
                        float(box[0]), float(box[1]),
                        float(box[2]), float(box[3]),
                    ),
                    label=best_label,
                    confidence=float(track_obj.score),
                    age=self._ages[tid],
                    hits=self._hits[tid],
                    time_since_update=0,
                )
            )

        # Age cho track không xuất hiện frame này (đã bị ByteTrack đánh Lost)
        seen_ids = {t.track_id for t in result}
        for tid in list(self._ages):
            if tid not in seen_ids:
                self._ages[tid] = self._ages.get(tid, 0) + 1

        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_label(
        box: np.ndarray,
        detections: Sequence[Detection],
        labels: list[str],
    ) -> str:
        """Find the original detection label closest to *box*."""
        best_iou = -1.0
        best_label = "vehicle"
        bx1, by1, bx2, by2 = box
        for det, label in zip(detections, labels):
            dx1, dy1, dx2, dy2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2
            ix1 = max(bx1, dx1)
            iy1 = max(by1, dy1)
            ix2 = min(bx2, dx2)
            iy2 = min(by2, dy2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            union = (bx2 - bx1) * (by2 - by1) + (dx2 - dx1) * (dy2 - dy1) - inter
            iou = inter / max(union, 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_label = label
        return best_label
