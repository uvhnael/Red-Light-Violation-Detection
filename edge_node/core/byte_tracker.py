"""Tracker ByteTrack dựa trên thư viện *supervision* — tối ưu xe máy.

Thay thế tracker greedy-IoU đơn giản bằng cơ chế ghép nối hai giai đoạn
của ByteTrack (ưu tiên detection confidence cao trước, rồi đến các box
confidence thấp).

Bản fork nhẹ (``_TunedByteTrack``) mở hai điểm so với supervision gốc:

1. Ngưỡng 0.7 (fused cost) cho khối đối chiếu track-chưa-xác-nhận — xe máy
   chạy nhanh + confidence trung bình không bao giờ vượt qua frame thứ 2,
   track không activate được và xe máy nhấp nháy/không hiện trên màn hình.
2. *Gating* theo khoảng cách + hướng di chuyển (``gate_enabled``, mặc định
   TẮT): từ chối ghép cặp track↔detection khi detection nằm quá xa vị trí
   Kalman dự đoán hoặc ngược hẳn hướng xe đang đi.

Ngoài association, ``SupervisionByteTracker`` bồi thêm một tầng **track
quality** mà ByteTrack không cung cấp (xem ``tracker_update.md``):

* quỹ đạo (``trajectory``) + vận tốc/hướng (``motion``) theo GIÂY,
* ``time_since_update`` thật (số frame track mất detection trước khi trở lại),
* ``confidence`` làm mượt EMA + ``detection_confidence`` thô,
* ``label`` bỏ phiếu có trọng số qua nhiều frame (chống lật motorcycle↔vehicle),
* log vòng đời track + cảnh báo association đáng ngờ (ID switch).

Cài đặt:  pip install supervision
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import hypot

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

from edge_node.core.contracts import (
    BoundingBox,
    Detection,
    Track,
    TrackSample,
)
from edge_node.core.motion import TrackMotion

LOGGER = logging.getLogger(__name__)

# Giá trị cost "cấm match" — lớn hơn mọi ngưỡng để linear_assignment loại cặp.
_GATE_REJECT_COST = 1e4
# Nhãn mặc định khi chưa có bằng chứng nào (chỉ dùng lúc track mới sinh).
_FALLBACK_LABEL = "vehicle"


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

    LƯU Ý SEMANTICS (dễ hiểu ngược!): association giai đoạn 1 của supervision
    dùng ``fuse_score`` — cost = 1 - IoU * conf, chỉ match khi cost <=
    ``minimum_matching_threshold``. Nghĩa là threshold CAO = match LỎNG HƠN
    (chấp nhận cặp IoU * conf nhỏ hơn), ngược với trực giác "threshold cao =
    chặt".

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

    Nhóm track quality (mới, 2026-09-12):

    * ``trajectory_max_samples``: số mẫu quỹ đạo giữ cho mỗi track (12 mẫu
      @ 3fps ≈ 4 giây — đủ cho evidence + hồi quy vận tốc).
    * ``confidence_ema_alpha``: trọng số detection mới trong EMA. 0.35 nghĩa
      là một frame conf tụt từ 0.90 xuống 0.31 chỉ kéo track confidence về
      ~0.69 thay vì 0.31 → downstream không hiểu nhầm "xe không đáng tin".
      Đặt 1.0 để tắt làm mượt (giữ hành vi cũ).
    * ``label_vote_window``: số frame bỏ phiếu nhãn (weighted theo conf ×
      IoU). Đặt 1 để tắt voting (dùng nhãn detection của frame hiện tại).
    * ``track_state_ttl_seconds``: giữ sổ tay track sau khi nó biến mất —
      đủ dài để track tìm lại còn quỹ đạo cũ, đủ ngắn để không rò rỉ RAM.
    * ``debug_associations``: log IoU/khoảng cách/conf của từng cặp match.
      Đây là dữ liệu để trả lời "0.85 thực sự gây lỗi ở đâu" trước khi nghĩ
      tới adaptive threshold.
    * ``id_switch_log_interval``: cứ N frame log WARNING tổng hợp (refind,
      missed frames, gate rejections). 0 = chỉ log theo sự kiện.
    """

    track_activation_threshold: float = 0.10    # track mới cần conf >= activation + 0.1
    lost_track_buffer: int = 30                 # dung sai che khuất ~1s (scale theo frame_rate)
    minimum_matching_threshold: float = 0.85    # fused cost — CAO = lỏng (xem docstring)
    unconfirmed_match_threshold: float = 0.85    # khối unconfirmed (supervision hard-code 0.7)
    frame_rate: float = 30.0                     # FPS NGUỒN THẬT — caller phải truyền
    minimum_consecutive_frames: int = 1
    min_detection_confidence: float = 0.15       # không lọc mất detection yếu của xe máy

    # ---- Track quality / motion ----
    trajectory_max_samples: int = 12
    confidence_ema_alpha: float = 0.35
    label_vote_window: int = 5
    track_state_ttl_seconds: float = 5.0

    # ---- Gating (mặc định TẮT — thay đổi hành vi association) ----
    gate_enabled: bool = False
    # Khoảng cách tối đa giữa tâm box Kalman và tâm detection, tính bằng hệ
    # số × đường chéo box dự đoán. Xa hơn → từ chối ghép, track thành Lost,
    # detection có thể sinh track mới.
    # ĐO THẬT (2026-09-12): hai box CÙNG cỡ mà IoU > 0 thì khoảng cách hai
    # tâm không bao giờ vượt đường chéo box (scan toàn bộ: tỉ lệ tối đa đúng
    # 1.0). Nên với factor >= 1.0 gate này KHÔNG chặn được cặp cùng cỡ nào —
    # nó chỉ có tác dụng khi detection là box lớn hơn nhiều (xe tải trùm lên
    # xe máy: IoU > 0 nhưng tâm lệch tới ~1.77× đường chéo box xe máy) hoặc
    # khi nới ``minimum_matching_threshold`` (adaptive matching, mục 1).
    # Gate HƯỚNG bên dưới thì vẫn có tác dụng ở ngưỡng mặc định, vì cặp
    # ngược chiều vẫn chồng lấn (IoU cao) nên matching chấp nhận.
    gate_distance_factor: float = 2.0
    # Cosine tối thiểu giữa vận tốc Kalman và vector dịch chuyển tới
    # detection; dưới ngưỡng (ngược hướng rõ rệt) thì từ chối. Chỉ xét khi
    # track đã có vận tốc >= gate_min_speed_px_per_frame.
    gate_direction_cosine: float = -0.3
    gate_min_speed_px_per_frame: float = 1.0

    # ---- Logging ----
    debug_associations: bool = False
    id_switch_log_interval: int = 0


@dataclass
class _TrackQuality:
    """Sổ tay chất lượng track, keyed by external_track_id."""

    first_frame: int
    last_frame: int
    bbox: BoundingBox
    hits: int = 1
    age: int = 1
    missed_frames: int = 0          # tổng frame mất detection (lifetime)
    refinds: int = 0                # số lần tìm lại sau khi biến mất
    gap_before: int = 0             # gap ngay trước lần xuất hiện hiện tại
    ema_confidence: float = 0.0
    detection_confidence: float = 0.0
    motion: TrackMotion = field(default_factory=TrackMotion)
    label_votes: deque[tuple[str, float]] = field(default_factory=deque)


class _TunedByteTrack(_real_bytrack_class()):
    """ByteTrack với ngưỡng unconfirmed cấu hình được + gating tuỳ chọn.

    Fork toàn bộ ``update_with_tensors`` chỉ để (a) thay MỘT hằng số
    (``thresh=0.7`` của khối unconfirmed) bằng ``unconfirmed_match_threshold``
    và (b) chèn bước gating vào ma trận cost giai đoạn 1. Logic ByteTrack gốc
    giữ nguyên từng dòng. Khi nâng supervision (pin <0.29) cần đối chiếu lại
    method gốc với bản sao ở đây.
    """

    def __init__(
        self,
        *,
        unconfirmed_match_threshold: float,
        gate_enabled: bool = False,
        gate_distance_factor: float = 2.0,
        gate_direction_cosine: float = -0.3,
        gate_min_speed_px_per_frame: float = 1.0,
        debug_associations: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.unconfirmed_match_threshold = unconfirmed_match_threshold
        self.gate_enabled = gate_enabled
        self.gate_distance_factor = gate_distance_factor
        self.gate_direction_cosine = gate_direction_cosine
        self.gate_min_speed_px_per_frame = gate_min_speed_px_per_frame
        self.debug_associations = debug_associations
        # Bộ đếm chẩn đoán — SupervisionByteTracker đọc để log tổng hợp.
        self.gate_rejections = 0
        self.suspicious_matches = 0

    # ------------------------------------------------------------------ #
    # Gating (mục 9 tài liệu nâng cấp) — mặc định TẮT
    # ------------------------------------------------------------------ #
    def _apply_gate(
        self,
        dists: np.ndarray,
        strack_pool: list[STrack],
        detections: list[STrack],
    ) -> np.ndarray:
        """Đặt cost = vô cực cho các cặp track↔detection vi phạm gate.

        Sửa MA TRẬN COST (không phải lọc danh sách match sau khi gán) để
        ``linear_assignment`` tự đẩy track/detection đó vào nhóm unmatched —
        track thành Lost, detection có thể sinh track mới: đúng semantics
        ByteTrack, không phá luật "mỗi detection một track".

        Hai điều kiện từ chối (chỉ áp dụng khi track đã có vận tốc):
        * khoảng cách tâm detection ↔ tâm box Kalman > factor × đường chéo box;
        * cosine(vận tốc Kalman, vector dịch chuyển) < ngưỡng (ngược hướng rõ).
        """
        if not self.gate_enabled or dists.size == 0:
            return dists
        for i, track in enumerate(strack_pool):
            mean = track.mean
            if mean is None:
                continue
            tlwh = track.tlwh
            tw, th = float(tlwh[2]), float(tlwh[3])
            if tw <= 0.0 or th <= 0.0:
                continue
            tcx, tcy = float(mean[0]), float(mean[1])
            max_dist = self.gate_distance_factor * hypot(tw, th)
            # mean[4:6] = vận tốc Kalman theo pixel/frame
            vx, vy = float(mean[4]), float(mean[5])
            speed = hypot(vx, vy)
            for j, det in enumerate(detections):
                dtlwh = det.tlwh
                dcx = float(dtlwh[0]) + float(dtlwh[2]) / 2.0
                dcy = float(dtlwh[1]) + float(dtlwh[3]) / 2.0
                dx, dy = dcx - tcx, dcy - tcy
                dist = hypot(dx, dy)
                reason = None
                if dist > max_dist:
                    reason = f"distance {dist:.1f}px > max {max_dist:.1f}px"
                elif speed >= self.gate_min_speed_px_per_frame:
                    cos = (vx * dx + vy * dy) / (speed * max(dist, 1e-6))
                    if cos < self.gate_direction_cosine:
                        reason = (
                            f"direction cos {cos:.2f} "
                            f"< {self.gate_direction_cosine:.2f}"
                        )
                if reason is None:
                    continue
                dists[i, j] = _GATE_REJECT_COST
                self.gate_rejections += 1
                LOGGER.debug(
                    "gate reject frame=%s track ext=%s int=%s det=(%.0f,%.0f): %s",
                    self.frame_id, track.external_track_id,
                    track.internal_track_id, dcx, dcy, reason,
                )
        return dists

    def _log_associations(
        self,
        matches: np.ndarray,
        strack_pool: list[STrack],
        detections: list[STrack],
        stage: str,
    ) -> None:
        """Log IoU/khoảng cách/conf từng cặp match — dữ liệu để tuning.

        Cặp match mà IoU thấp hoặc tâm dịch chuyển hơn nửa đường chéo box là
        dấu hiệu ID switch: nâng lên WARNING thay vì DEBUG để thấy ngay.
        """
        if not self.debug_associations or len(matches) == 0:
            return
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            tlbr = track.tlbr
            dtlbr = det.tlbr
            iou = _iou_xyxy(tlbr, dtlbr)
            dist = hypot(
                (dtlbr[0] + dtlbr[2]) / 2 - (tlbr[0] + tlbr[2]) / 2,
                (dtlbr[1] + dtlbr[3]) / 2 - (tlbr[1] + tlbr[3]) / 2,
            )
            diag = hypot(tlbr[2] - tlbr[0], tlbr[3] - tlbr[1])
            suspicious = iou < 0.2 or dist > 0.5 * diag
            if suspicious:
                self.suspicious_matches += 1
            log = LOGGER.warning if suspicious else LOGGER.debug
            log(
                "assoc[%s] frame=%s track ext=%s int=%s iou=%.3f dist=%.1fpx "
                "diag=%.1fpx det_conf=%.2f state=%s",
                stage, self.frame_id, track.external_track_id,
                track.internal_track_id, iou, dist, diag,
                float(det.score), track.state.name,
            )

    # ------------------------------------------------------------------ #
    def update_with_tensors(self, tensors: np.ndarray) -> list[STrack]:
        """Bản sao ``sv.ByteTrack.update_with_tensors`` + 2 điểm fork."""
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
        dists = self._apply_gate(dists, strack_pool, detections)   # ← fork (2)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.minimum_matching_threshold
        )
        self._log_associations(matches, strack_pool, detections, "high")

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
        self._log_associations(matches, r_tracked_stracks, detections_second, "low")
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
            dists, thresh=self.unconfirmed_match_threshold  # ← fork (1)
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


def _iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """IoU của hai box xyxy (numpy hoặc list)."""
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(union, 1e-6)


def _vote_label(votes: Sequence[tuple[str, float]]) -> str:
    """Bỏ phiếu nhãn có trọng số — chống lật motorcycle↔vehicle từng frame."""
    if not votes:
        return _FALLBACK_LABEL
    totals: dict[str, float] = {}
    for label, weight in votes:
        totals[label] = totals.get(label, 0.0) + weight
    best = max(totals.values())
    winners = [label for label, total in totals.items() if total >= best - 1e-9]
    if len(winners) == 1:
        return winners[0]
    # Hoà điểm → lấy nhãn của frame gần nhất trong nhóm hoà.
    for label, _weight in reversed(list(votes)):
        if label in winners:
            return label
    return votes[-1][0]


class SupervisionByteTracker:
    """MultiObjectTracker backed by (tuned) supervision.ByteTrack.

    Ngoài output ByteTrack, mỗi :class:`Track` mang thêm quỹ đạo, vận tốc,
    ``time_since_update`` thật, confidence đã làm mượt và nhãn đã vote.
    """

    VEHICLE_LABELS: frozenset[str] = frozenset({
        "car", "truck", "bus", "motorcycle", "bicycle", "vehicle",
    })

    def __init__(self, config: ByteTrackerConfig | None = None) -> None:
        cfg = config or ByteTrackerConfig()
        if not 0.0 < cfg.confidence_ema_alpha <= 1.0:
            raise ValueError("confidence_ema_alpha must be in (0, 1]")
        if cfg.trajectory_max_samples < 2:
            raise ValueError("trajectory_max_samples must be >= 2")
        if cfg.label_vote_window < 1:
            raise ValueError("label_vote_window must be >= 1")
        self._config = cfg
        self._min_confidence = cfg.min_detection_confidence
        self._byte_track = _TunedByteTrack(
            track_activation_threshold=cfg.track_activation_threshold,
            lost_track_buffer=cfg.lost_track_buffer,
            minimum_matching_threshold=cfg.minimum_matching_threshold,
            frame_rate=cfg.frame_rate,
            minimum_consecutive_frames=cfg.minimum_consecutive_frames,
            unconfirmed_match_threshold=cfg.unconfirmed_match_threshold,
            gate_enabled=cfg.gate_enabled,
            gate_distance_factor=cfg.gate_distance_factor,
            gate_direction_cosine=cfg.gate_direction_cosine,
            gate_min_speed_px_per_frame=cfg.gate_min_speed_px_per_frame,
            debug_associations=cfg.debug_associations,
        )
        # ByteTrack cấp external id mới theo call nên ta tự giữ sổ tay per ID.
        self._quality: dict[int, _TrackQuality] = {}
        self._frame_count = 0
        self._ttl_frames = max(
            5, int(round(cfg.frame_rate * cfg.track_state_ttl_seconds))
        )
        self._lost_logged_at: dict[int, int] = {}

    # ------------------------------------------------------------------
    # MultiObjectTracker protocol
    # ------------------------------------------------------------------
    def update(
        self,
        detections: Sequence[Detection],
        frame_index: int,
        timestamp_ms: float,
    ) -> list[Track]:
        # frame_index phải đơn điệu: gap (time_since_update) tính từ nó, và
        # pipeline có thể skip frame hỏng nên frame_index nhảy số là hợp lệ.
        self._frame_count = max(self._frame_count, frame_index)
        frame = self._frame_count

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
        stracks = self._byte_track.update_with_tensors(tensors=tensors)

        result: list[Track] = []
        for track_obj in stracks:
            tid = int(track_obj.external_track_id)
            if tid < 0:
                continue  # chưa được cấp external id (track unconfirmed) — bỏ qua
            bbox = self._safe_bbox(track_obj.tlbr)
            if bbox is None:
                continue
            det_conf = float(track_obj.score)
            quality = self._update_quality(
                tid=tid, bbox=bbox, det_conf=det_conf, frame=frame,
                timestamp_ms=timestamp_ms, detections=filtered,
            )
            result.append(self._build_track(tid, quality, det_conf))

        self._log_missing(result, frame)
        self._prune(frame)
        return result

    # ------------------------------------------------------------------
    # Track quality
    # ------------------------------------------------------------------
    def _update_quality(
        self,
        *,
        tid: int,
        bbox: BoundingBox,
        det_conf: float,
        frame: int,
        timestamp_ms: float,
        detections: Sequence[Detection],
    ) -> _TrackQuality:
        """Cập nhật sổ tay track (tạo mới nếu lần đầu thấy id này)."""
        cfg = self._config
        prior = self._quality.get(tid)
        sample = TrackSample(
            timestamp_ms=float(timestamp_ms),
            point=bbox.bottom_center,
            width=bbox.width,
            height=bbox.height,
            detection_confidence=det_conf,
            matched=True,
        )

        if prior is None:
            quality = _TrackQuality(
                first_frame=frame,
                last_frame=frame,
                bbox=bbox,
                ema_confidence=det_conf,
                detection_confidence=det_conf,
                motion=TrackMotion(cfg.trajectory_max_samples),
            )
            quality.motion.push(sample)
            self._quality[tid] = quality
            self._vote(quality, bbox, detections, det_conf)
            # DEBUG, không phải INFO: cảnh xe đông sinh hàng trăm track mỗi
            # video (240 dòng cho 384 frame ở 20221003-102556.mp4) làm chìm
            # log vận hành. Sự kiện ĐÁNG chú ý (refind đáng ngờ, summary định
            # kỳ) vẫn ở WARNING.
            LOGGER.debug(
                "track %s created frame=%s conf=%.2f box=(%.0f,%.0f,%.0f,%.0f) label=%s",
                tid, frame, det_conf, bbox.x1, bbox.y1, bbox.x2, bbox.y2,
                _vote_label(quality.label_votes),
            )
            return quality

        gap = max(0, frame - prior.last_frame - 1)
        prior.age += 1 + gap
        prior.hits += 1
        prior.gap_before = gap
        if gap > 0:
            prior.missed_frames += gap
            prior.refinds += 1
            # Log TRƯỚC khi xoá quỹ đạo (cần điểm neo cuối của đoạn cũ).
            self._log_refind(tid, prior, bbox, det_conf, frame, gap)
            # Quỹ đạo đứt: xoá lịch sử thay vì hồi quy xuyên qua gap — hai đoạn
            # cách nhau nhiều frame sẽ cho vận tốc ảo rất lớn.
            prior.motion.clear()

        # EMA: C_t = α * det_conf + (1-α) * C_(t-1)
        alpha = cfg.confidence_ema_alpha
        prior.ema_confidence = alpha * det_conf + (1.0 - alpha) * prior.ema_confidence
        prior.detection_confidence = det_conf
        prior.motion.push(sample)
        prior.bbox = bbox
        prior.last_frame = frame
        self._vote(prior, bbox, detections, det_conf)
        self._lost_logged_at.pop(tid, None)
        return prior

    def _vote(
        self,
        quality: _TrackQuality,
        bbox: BoundingBox,
        detections: Sequence[Detection],
        det_conf: float,
    ) -> None:
        """Thêm một phiếu nhãn (trọng số = conf × IoU với detection khớp nhất).

        Khi box Kalman không đè detection nào (trễ nhịp), KHÔNG bỏ phiếu: gán
        nhãn fallback sẽ làm loãng kết quả vote của các frame có bằng chứng.
        """
        best_iou = 0.0
        best_label = ""
        best_conf = 0.0
        for det in detections:
            iou = bbox.iou(det.bbox)
            if iou > best_iou:
                best_iou, best_label, best_conf = iou, det.label, det.confidence
        if best_iou <= 0.0:
            return
        votes = quality.label_votes
        votes.append((best_label, max(best_conf, det_conf) * best_iou))
        while len(votes) > self._config.label_vote_window:
            votes.popleft()

    def _log_refind(
        self,
        tid: int,
        prior: _TrackQuality,
        bbox: BoundingBox,
        det_conf: float,
        frame: int,
        gap: int,
    ) -> None:
        """Log track tìm lại sau gap — nguồn ID switch chính.

        Điểm neo cũ → mới dịch chuyển lớn bất thường (so với đường chéo box)
        nghĩa là association vừa ghép track này với một xe KHÁC: cảnh báo để
        rà soát thay vì đoán.
        """
        last_point = prior.motion.last_point
        new_point = bbox.bottom_center
        if last_point is None:
            LOGGER.debug(
                "track %s refind frame=%s gap=%d conf=%.2f (no trajectory)",
                tid, frame, gap, det_conf,
            )
            return
        dist = hypot(new_point.x - last_point.x, new_point.y - last_point.y)
        diag = hypot(bbox.width, bbox.height)
        ratio = dist / max(diag, 1e-6)
        message = (
            "track %s refind frame=%s gap=%d conf=%.2f dist=%.1fpx "
            "(%.2fx box diag) last=(%.0f,%.0f) now=(%.0f,%.0f)"
        )
        args = (
            tid, frame, gap, det_conf, dist, ratio,
            last_point.x, last_point.y, new_point.x, new_point.y,
        )
        if ratio > 2.0:
            LOGGER.warning("SUSPICIOUS id-switch? " + message, *args)
            self._byte_track.suspicious_matches += 1
        else:
            LOGGER.debug(message, *args)

    def _build_track(self, tid: int, q: _TrackQuality, det_conf: float) -> Track:
        """Dựng Track output từ sổ tay chất lượng."""
        motion = q.motion.motion()
        label = _vote_label(q.label_votes)
        metadata = {
            "velocity_px_per_s": round(motion.speed, 2),
            "vx_px_per_s": round(motion.vx, 2),
            "vy_px_per_s": round(motion.vy, 2),
            "direction": motion.direction,
            "heading_deg": round(motion.heading_deg, 1),
            "gap_frames": q.gap_before,
            "missed_frames_total": q.missed_frames,
            "refinds": q.refinds,
            "first_frame": q.first_frame,
            "label_votes": len(q.label_votes),
            "detection_confidence": round(det_conf, 4),
        }
        return Track(
            track_id=tid,
            bbox=q.bbox,
            label=label,
            confidence=round(q.ema_confidence, 4),
            age=q.age,
            hits=q.hits,
            # ByteTrack chỉ output track đã match trong frame nên 0 = có
            # detection thật ở frame này. > 0 = track vừa được tìm lại sau
            # gap q.gap_before frame (box Kalman có thể lệch) → downstream
            # nên thận trọng khi kết luận vi phạm ngay frame đó.
            time_since_update=q.gap_before,
            detection_confidence=det_conf,
            motion=motion,
            trajectory=q.motion.samples,
            metadata=metadata,
        )

    def _log_missing(self, result: Sequence[Track], frame: int) -> None:
        """Log track vừa biến mất khỏi output (Lost bên trong ByteTrack)."""
        seen = {t.track_id for t in result}
        for tid, quality in self._quality.items():
            if tid in seen or self._lost_logged_at.get(tid) == frame:
                continue
            if quality.last_frame < frame:
                self._lost_logged_at[tid] = frame
                LOGGER.debug(
                    "track %s lost at frame=%s (last seen %s, hits=%d)",
                    tid, frame, quality.last_frame, quality.hits,
                )
        interval = self._config.id_switch_log_interval
        if interval > 0 and frame > 0 and frame % interval == 0:
            bt = self._byte_track
            LOGGER.warning(
                "track quality @frame=%s active=%d refinds=%d missed_frames=%d "
                "gate_rejections=%d suspicious_assoc=%d",
                frame, len(result),
                sum(q.refinds for q in self._quality.values()),
                sum(q.missed_frames for q in self._quality.values()),
                bt.gate_rejections, bt.suspicious_matches,
            )

    def _prune(self, frame: int) -> None:
        """Xoá sổ tay track biến mất quá lâu (chống rò rỉ bộ nhớ)."""
        stale = [
            tid for tid, q in self._quality.items()
            if frame - q.last_frame > self._ttl_frames
        ]
        for tid in stale:
            self._quality.pop(tid, None)
            self._lost_logged_at.pop(tid, None)

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_bbox(box: np.ndarray) -> BoundingBox | None:
        """Kalman có thể sinh box thoái hoá / số không hữu hạn → bỏ qua."""
        x1, y1, x2, y2 = (float(v) for v in box)
        if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return BoundingBox(x1, y1, x2, y2)
