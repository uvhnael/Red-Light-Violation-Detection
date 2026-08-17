"""Core orchestration for red-light violation detection.

Key change from the original *rlvd* pipeline:
* When ``enable_queue`` is ``True``, each violation event is serialised to
  a plain ``dict`` payload and dispatched to the Celery task queue via
  ``push_violation_to_server.delay(payload)``.
* No JSON or image files are written to disk from the processing loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Optional

from edge_node.core.contracts import (
    FrameSource,
    MultiObjectTracker,
    ObjectDetector,
    PlateObservation,
    PlateRecognizer,
    TrafficLightClassifier,
    ViolationEvent,
)
from edge_node.core.violation_logic import RedLightStabilizer, ViolationDetector

LOGGER = logging.getLogger(__name__)

# Frames after which a remembered plate is forgotten if the track vanished
_PLATE_MEMORY_TTL_FRAMES = 300


def _event_to_payload(event: ViolationEvent) -> dict:
    """Serialise a ViolationEvent into a JSON-safe dictionary for the queue."""
    return {
        "event_id": event.event_id,
        "track_id": event.track_id,
        "frame_index": event.frame_index,
        "timestamp_ms": event.timestamp_ms,
        "crossing_point": {"x": event.crossing_point.x, "y": event.crossing_point.y},
        "previous_point": {"x": event.previous_point.x, "y": event.previous_point.y},
        "bbox_xyxy": list(event.bbox.as_xyxy()),
        "light_state": event.light_state.value,
        "light_confidence": event.light_confidence,
        "previous_side": event.previous_side,
        "current_side": event.current_side,
        "plate": (
            {"text": event.plate.text, "confidence": event.plate.confidence}
            if event.plate
            else None
        ),
        "metadata": dict(event.metadata),
    }


@dataclass(frozen=True)
class PipelineResult:
    """Summary returned after processing a frame source."""
    frames_processed: int
    violations: tuple[ViolationEvent, ...]


class RedLightViolationPipeline:
    """Model-agnostic red-light violation pipeline with optional queue output."""

    def __init__(
        self,
        detector: ObjectDetector,
        tracker: MultiObjectTracker,
        light_classifier: TrafficLightClassifier,
        stabilizer: RedLightStabilizer,
        violation_detector: ViolationDetector,
        ocr: Optional[PlateRecognizer] = None,
        enable_queue: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._detector = detector
        self._tracker = tracker
        self._light_classifier = light_classifier
        self._stabilizer = stabilizer
        self._violation_detector = violation_detector
        self._ocr = ocr
        self._enable_queue = enable_queue
        self._logger = logger or LOGGER
        # Best plate reading remembered per track so violation reports
        # still carry plate fields when OCR fails on the exact frame.
        self._plate_memory: dict[int, PlateObservation] = {}
        self._plate_last_seen: dict[int, int] = {}

    def process(
        self, source: FrameSource, max_frames: Optional[int] = None,
    ) -> PipelineResult:
        """Process frames and return violation events.

        When *enable_queue* is active each violation payload is dispatched
        to the Celery background worker via ``push_violation_to_server.delay``.
        No files are written to disk from this method.
        """
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be >= 1 when supplied")

        events: list[ViolationEvent] = []
        frames_processed = 0

        for packet in source:
            if max_frames is not None and frames_processed >= max_frames:
                break

            light = self._classify_light(packet)
            stable_signal = self._stabilizer.update(light, packet.frame_index)
            detections = self._detect_objects(packet)
            tracks = self._update_tracks(packet, detections)
            frame_events = self._violation_detector.update(
                tracks=tracks,
                signal=stable_signal,
                frame_index=packet.frame_index,
                timestamp_ms=packet.timestamp_ms,
            )

            if self._ocr is not None and frame_events:
                frame_events = self._attach_plates(
                    packet.image, tracks, frame_events,
                )

            # ---- Queue dispatch (requirement 2 from needed.md) ----
            if self._enable_queue and frame_events:
                self._dispatch_to_queue(frame_events)

            events.extend(frame_events)
            frames_processed += 1

        return PipelineResult(
            frames_processed=frames_processed,
            violations=tuple(events),
        )

    # ------------------------------------------------------------------ #
    # Queue dispatch                                                       #
    # ------------------------------------------------------------------ #
    def _dispatch_to_queue(self, frame_events: list[ViolationEvent]) -> None:
        """Push violation payloads to the Celery task queue."""
        try:
            from edge_node.worker.tasks import push_violation_to_server
        except ImportError:
            self._logger.warning(
                "Celery worker not available – skipping queue dispatch"
            )
            return

        for event in frame_events:
            payload = _event_to_payload(event)
            try:
                push_violation_to_server.delay(payload)
                self._logger.info(
                    "Queued violation %s for background push",
                    event.event_id,
                )
            except Exception as exc:
                self._logger.error(
                    "Failed to enqueue violation %s: %s",
                    event.event_id,
                    exc,
                )

    # ------------------------------------------------------------------ #
    # Internal helpers (unchanged from rlvd)                               #
    # ------------------------------------------------------------------ #
    def _classify_light(self, packet):
        try:
            return self._light_classifier.classify(
                packet.image, packet.frame_index, packet.timestamp_ms,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Traffic-light classification failed at frame {packet.frame_index}"
            ) from exc

    def _detect_objects(self, packet):
        try:
            return self._detector.detect(
                packet.image, packet.frame_index, packet.timestamp_ms,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Object detection failed at frame {packet.frame_index}"
            ) from exc

    def _update_tracks(self, packet, detections):
        try:
            return self._tracker.update(
                detections, packet.frame_index, packet.timestamp_ms,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Tracking failed at frame {packet.frame_index}"
            ) from exc

    def _attach_plates(
        self, frame, tracks, events: list[ViolationEvent],
    ) -> list[ViolationEvent]:
        track_by_id = {track.track_id: track for track in tracks}
        self._prune_plate_memory(tracks)
        enriched: list[ViolationEvent] = []
        for event in events:
            track = track_by_id.get(event.track_id)
            if track is None:
                plate = self._plate_memory.get(event.track_id)
                enriched.append(replace(event, plate=plate))
                continue
            plate = None
            if self._ocr is not None:
                try:
                    plate = self._ocr.recognize(frame, track)
                except Exception as exc:
                    self._logger.warning(
                        "OCR failed for track %s at frame %s: %s",
                        event.track_id, event.frame_index, exc,
                    )
                    plate = None
                plate = self._remember_plate(track, plate)
            else:
                # OCR disabled: reuse last known plate for this track
                plate = self._plate_memory.get(event.track_id)
            enriched.append(replace(event, plate=plate))
        return enriched

    def _remember_plate(self, track, plate: Optional[PlateObservation]) -> Optional[PlateObservation]:
        """Cache the best plate reading seen for *track*, returning it for the report."""
        self._plate_last_seen[track.track_id] = track.age
        if plate is None:
            return self._plate_memory.get(track.track_id)
        cached = self._plate_memory.get(track.track_id)
        if cached is None or plate.confidence > cached.confidence:
            self._plate_memory[track.track_id] = plate
        return self._plate_memory[track.track_id]

    def _prune_plate_memory(self, tracks) -> None:
        """Drop plates for tracks that have not been seen for a long time."""
        active_ids = {track.track_id for track in tracks}
        if not tracks:
            return
        newest_age = max(track.age for track in tracks)
        stale = [
            track_id
            for track_id, last_seen in self._plate_last_seen.items()
            if track_id not in active_ids and newest_age - last_seen > _PLATE_MEMORY_TTL_FRAMES
        ]
        for track_id in stale:
            self._plate_memory.pop(track_id, None)
            self._plate_last_seen.pop(track_id, None)
