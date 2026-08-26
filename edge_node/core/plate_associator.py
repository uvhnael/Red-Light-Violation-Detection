"""Gán biển số đã phát hiện với xe đang được track.

Giải bài "biển số nào của xe nào": mỗi detection biển số được ghép với
track xe có bounding box chứa nó, sau đó OCR chỉ chạy **một lần mỗi
track** rồi cache. Kết quả cache là *chốt* (biển không bị OCR lại trong
vòng đời track) khi confidence đạt từ ``ocr_confidence_threshold`` trở
lên **và** độ dài text nằm trong ``expected_plate_lengths``. Thấp hơn
ngưỡng đó, OCR thử lại nhưng bị giới hạn tần suất: mỗi
``retry_interval`` frame một lần.

Thứ tự pipeline (mỗi frame):
    phát hiện xe → track (ByteTrack) → phát hiện biển → gán

Cách dùng
---------
    associator = PlateAssociator()
    track_plates, unassigned = associator.update(
        frame, tracks, plate_detections, ocr, frame_index,
    )
    # track_plates: {track_id: PlateObservation}  → vẽ lên box xe
    # unassigned:   [(bbox, None), ...]           → biển không có xe chứa
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional, Sequence

from edge_node.core.contracts import (
    BoundingBox,
    Detection,
    PlateObservation,
    Track,
)

LOGGER = logging.getLogger(__name__)


def plate_text_length(text: str) -> int:
    """Number of meaningful characters in a normalised plate string.

    Counts letters and digits only — separators like ``-`` and ``.`` are
    layout noise, not part of the plate identity.  ``29H-123.45`` → 8,
    ``3012345`` → 7.
    """
    return sum(1 for ch in text if ch.isalnum())


def containment_ratio(plate: BoundingBox, vehicle: BoundingBox) -> float:
    """Fraction of the *plate* area that lies inside the *vehicle* box.

    ≈ 1.0 when the plate is fully inside the vehicle bounding box,
    which is the normal case for rear-plate detections.
    """
    inter_x1 = max(plate.x1, vehicle.x1)
    inter_y1 = max(plate.y1, vehicle.y1)
    inter_x2 = min(plate.x2, vehicle.x2)
    inter_y2 = min(plate.y2, vehicle.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if plate.area <= 0.0:
        return 0.0
    return intersection / plate.area


class PlateAssociator:
    """Matches plate detections to vehicle tracks and caches OCR results.

    Parameters
    ----------
    min_containment : minimum fraction of the plate area that must fall
        inside a vehicle box for the plate to be assigned to that track.
    ocr_confidence_threshold : a cached reading at or above this confidence
        AND with an expected plate length is final — the plate is not
        OCR'd again while the track lives.
    expected_plate_lengths : acceptable alphanumeric lengths for a valid
        plate (Vietnamese plates are 7–9 chars).  A reading is only
        considered final when its length is one of these values.
    retry_interval : while below the threshold, re-OCR at most once every
        N frames (the plate usually becomes more readable as the vehicle
        approaches, so we keep trying at a reduced rate).
    max_cache_age : frames after a track's last sighting before its cached
        plate is dropped (ByteTrack keeps IDs for ~45 frames while lost).
    """

    def __init__(
        self,
        min_containment: float = 0.5,
        ocr_confidence_threshold: float = 0.80,
        expected_plate_lengths: tuple[int, ...] = (8, 9),
        retry_interval: int = 5,
        max_cache_age: int = 90,
    ) -> None:
        self.min_containment = min_containment
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.expected_plate_lengths = tuple(expected_plate_lengths)
        self.retry_interval = retry_interval
        self.max_cache_age = max_cache_age

        self._cache: dict[int, PlateObservation] = {}   # track_id → best reading
        self._last_seen: dict[int, int] = {}            # track_id → frame index
        self._last_ocr: dict[int, int] = {}             # track_id → frame index

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_final(self, obs: PlateObservation) -> bool:
        """A reading is final when confidence is high AND length matches."""
        return (
            obs.confidence >= self.ocr_confidence_threshold
            and plate_text_length(obs.text) in self.expected_plate_lengths
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self,
        frame,
        tracks: Sequence[Track],
        plate_detections: Sequence[Detection],
        ocr,
        frame_index: int,
    ) -> tuple[dict[int, PlateObservation], list[tuple[BoundingBox, Optional[PlateObservation]]]]:
        """Associate plates with tracks and run (cached) OCR.

        Returns
        -------
        track_plates : mapping ``track_id → PlateObservation`` for every
            track that currently has a plate reading (fresh or cached).
        unassigned : ``(bbox, None)`` pairs for plates that matched no
            vehicle track — the visualizer draws bare boxes for these.
        """
        # ── 1. Match each plate to the vehicle track containing it ──
        assignments: dict[int, Detection] = {}
        unassigned_dets: list[Detection] = []

        for plate in plate_detections:
            best_tid: Optional[int] = None
            best_ratio = 0.0
            for track in tracks:
                ratio = containment_ratio(plate.bbox, track.bbox)
                if ratio > best_ratio:
                    best_ratio, best_tid = ratio, track.track_id
            if best_tid is not None and best_ratio >= self.min_containment:
                # Two plates on one track (BSD dual-plate trucks): keep the
                # higher-confidence detection.
                current = assignments.get(best_tid)
                if current is None or plate.confidence > current.confidence:
                    assignments[best_tid] = plate
            else:
                unassigned_dets.append(plate)

        # ── 2. OCR with per-track caching ──
        track_plates: dict[int, PlateObservation] = {}

        for tid, plate in assignments.items():
            self._last_seen[tid] = frame_index
            cached = self._cache.get(tid)

            if cached is not None and self._is_final(cached):
                # Final reading (high confidence + correct length) —
                # stop OCR'ing this plate entirely.
                track_plates[tid] = cached
                continue

            should_ocr = (
                ocr is not None
                and (
                    tid not in self._last_ocr
                    or frame_index - self._last_ocr[tid] >= self.retry_interval
                )
            )
            if should_ocr and ocr is not None:
                self._last_ocr[tid] = frame_index
                try:
                    obs = ocr.recognize_bbox(frame, plate.bbox, ref_id=tid)
                except Exception as exc:
                    LOGGER.debug("Plate OCR failed for track %s: %s", tid, exc)
                    obs = None
                if obs is not None and (
                    cached is None or obs.confidence > cached.confidence
                ):
                    self._cache[tid] = obs
                    LOGGER.debug(
                        "Plate track %s: %s (conf=%.2f, len=%d)",
                        tid, obs.text, obs.confidence, plate_text_length(obs.text),
                    )

            if tid in self._cache:
                track_plates[tid] = self._cache[tid]

        # ── 3. Sticky cache: keep showing the plate even on frames where
        #       the plate detector missed it (flicker prevention) ──
        for track in tracks:
            tid = track.track_id
            if tid in self._cache and tid not in track_plates:
                track_plates[tid] = self._cache[tid]
                self._last_seen[tid] = frame_index

        # ── 4. Prune cache entries for tracks gone too long ──
        for tid in list(self._cache):
            if frame_index - self._last_seen.get(tid, 0) > self.max_cache_age:
                del self._cache[tid]
                self._last_seen.pop(tid, None)
                self._last_ocr.pop(tid, None)

        unassigned: list[tuple[BoundingBox, Optional[PlateObservation]]] = [
            (det.bbox, None) for det in unassigned_dets
        ]
        return track_plates, unassigned

    def get_plate(self, track_id: int) -> Optional[PlateObservation]:
        """Return the cached plate reading for *track_id*, if any.

        Used to attach a plate to violation events without re-running OCR.
        """
        return self._cache.get(track_id)

    @property
    def cached_plates(self) -> Mapping[int, PlateObservation]:
        """Read-only view of the current plate cache."""
        return dict(self._cache)
