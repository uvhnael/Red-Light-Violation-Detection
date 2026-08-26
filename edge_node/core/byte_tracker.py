"""Tracker ByteTrack dựa trên thư viện *supervision*.

Thay thế tracker greedy-IoU đơn giản bằng cơ chế ghép nối hai giai đoạn
của ByteTrack (ưu tiên detection confidence cao trước, rồi đến các box
confidence thấp). Giúp loại bỏ gần hết hiện tượng nhảy ID so với tracker
IoU cũ.

Cài đặt:  pip install supervision
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    import supervision as sv
except ImportError as _exc:
    raise ImportError(
        "Cần package 'supervision' để chạy ByteTrack. "
        "Cài bằng lệnh: pip install supervision"
    ) from _exc

from edge_node.core.contracts import BoundingBox, Detection, Track

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ByteTrackerConfig:
    """Tuning knobs exposed by supervision's ByteTrack.

    Optimised for traffic-camera scenarios (moderate occlusion, 25-30 fps).
    """

    track_activation_threshold: float = 0.30    # slightly higher to reduce false tracks
    lost_track_buffer: int = 45                  # keep lost tracks longer (1.5 s @ 30 fps)
    minimum_matching_threshold: float = 0.70     # looser IoU for busy intersections
    frame_rate: int = 30
    minimum_consecutive_frames: int = 1
    min_detection_confidence: float = 0.25


class SupervisionByteTracker:
    """MultiObjectTracker backed by supervision.ByteTrack."""

    VEHICLE_LABELS: frozenset[str] = frozenset({
        "car", "truck", "bus", "motorcycle", "bicycle", "vehicle",
    })

    def __init__(self, config: ByteTrackerConfig | None = None) -> None:
        cfg = config or ByteTrackerConfig()
        self._min_confidence = cfg.min_detection_confidence
        self._byte_track = sv.ByteTrack(
            track_activation_threshold=cfg.track_activation_threshold,
            lost_track_buffer=cfg.lost_track_buffer,
            minimum_matching_threshold=cfg.minimum_matching_threshold,
            frame_rate=cfg.frame_rate,
            minimum_consecutive_frames=cfg.minimum_consecutive_frames,
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

        if not filtered:
            sv_dets = sv.Detections.empty()
        else:
            xyxy = np.array(
                [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in filtered],
                dtype=np.float32,
            )
            confidence = np.array(
                [d.confidence for d in filtered], dtype=np.float32,
            )
            class_id = np.zeros(len(filtered), dtype=int)
            sv_dets = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
            )

        tracked = self._byte_track.update_with_detections(sv_dets)

        labels_arr = [d.label for d in filtered]
        result: list[Track] = []
        seen_ids: set[int] = set()

        if tracked.tracker_id is not None:
            for idx in range(len(tracked)):
                tid = int(tracked.tracker_id[idx])
                seen_ids.add(tid)
                self._hits[tid] = self._hits.get(tid, 0) + 1
                self._ages[tid] = self._ages.get(tid, 0) + 1

                # Resolve label from the matched detection
                box = tracked.xyxy[idx]
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
                        confidence=float(tracked.confidence[idx]) if tracked.confidence is not None else 0.0,
                        age=self._ages[tid],
                        hits=self._hits[tid],
                        time_since_update=0,
                    )
                )

        # Increment age for unseen tracks (they'll be pruned by ByteTrack internally)
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
