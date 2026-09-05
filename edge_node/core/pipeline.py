"""Điều phối trung tâm của hệ phát hiện vi phạm vượt đèn đỏ.

Đường gửi event vi phạm đã xác nhận:
* ``outbox``: mỗi event + ảnh bằng chứng JPEG được ghi vào
  SQLite outbox bền vững ngay lúc phát hiện. Một
  :class:`edge_node.violation_sender.ViolationSender` chạy nền sẽ gom
  batch đẩy lên Central Server khi mạng cho phép — không mất dữ liệu khi
  mất kết nối hay restart. Đây là đường duy nhất hiện nay; đường Celery
  cũ đã bị loại bỏ.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Optional

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
from edge_node.metrics import (
    increment_errors,
    increment_frames,
    increment_violations,
)

if TYPE_CHECKING:
    from edge_node.outbox import ViolationOutbox

LOGGER = logging.getLogger(__name__)

# Số frame sau đó một biển số được ghi nhớ sẽ bị quên nếu track biến mất
_PLATE_MEMORY_TTL_FRAMES = 300

# Hook mỗi frame dùng bởi visualiser (run_pipeline.py live view). Nhận
# (packet, light, stable_signal, detections, tracks, frame_events) sau khi
# logic vi phạm đã chạy xong cho frame đó.
FrameCallback = Callable[..., None]


def _event_to_payload(event: ViolationEvent) -> dict:
    """Tuần tự hoá ViolationEvent thành dict an toàn JSON cho queue."""
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
        logger: Optional[logging.Logger] = None,
        frame_callback: Optional[FrameCallback] = None,
        outbox: Optional["ViolationOutbox"] = None,
        evidence_max_width: int = 1280,
        evidence_quality: int = 80,
    ) -> None:
        self._detector = detector
        self._tracker = tracker
        self._light_classifier = light_classifier
        self._stabilizer = stabilizer
        self._violation_detector = violation_detector
        self._ocr = ocr
        self._logger = logger or LOGGER
        self._frame_callback = frame_callback
        self._outbox = outbox
        self._evidence_max_width = evidence_max_width
        self._evidence_quality = evidence_quality
        # Best plate reading remembered per track so violation reports
        # still carry plate fields when OCR fails on the exact frame.
        self._plate_memory: dict[int, PlateObservation] = {}
        self._plate_last_seen: dict[int, int] = {}

    def process(
        self, source: FrameSource, max_frames: Optional[int] = None,
    ) -> PipelineResult:
        """Process frames and return violation events.

        Each violation is persisted to the durable outbox (when supplied) so
        the background sender can batch-deliver it. No files are written to
        disk from this method other than the outbox rows.
        """
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be >= 1 when supplied")

        events: list[ViolationEvent] = []
        frames_processed = 0

        for packet in source:
            if max_frames is not None and frames_processed >= max_frames:
                break

            # Một frame hỏng (corrupt, mất gói RTSP, lỗi model tạm thời)
            # KHÔNG được làm sập cả node: bỏ qua frame đó và đi tiếp.
            # Tracker/stabilizer đều chịu được khoảng trống ngắn (lost
            # buffer + debounce), nên skip một frame là an toàn.
            try:
                light = self._classify_light(packet)
                stable_signal = self._stabilizer.update(light, packet.frame_index)
                self._publish_light_state(packet, light, stable_signal)
                detections = self._detect_objects(packet)
                tracks = self._update_tracks(packet, detections)
                frame_events = self._violation_detector.update(
                    tracks=tracks,
                    signal=stable_signal,
                    frame_index=packet.frame_index,
                    timestamp_ms=packet.timestamp_ms,
                )
            except Exception as exc:
                self._logger.warning(
                    "Skipping frame %s due to processing error: %s",
                    packet.frame_index, exc,
                )
                increment_errors()
                continue

            if self._ocr is not None and frame_events:
                frame_events = self._attach_plates(
                    packet.image, tracks, frame_events,
                )

            # ---- Durable outbox dispatch (survives outage + restart) ----
            if self._outbox is not None and frame_events:
                self._dispatch_to_outbox(packet.image, frame_events)
                for event in frame_events:
                    increment_violations(event.timestamp_ms)

            if self._frame_callback is not None:
                try:
                    self._frame_callback(
                        packet, light, stable_signal,
                        detections, tracks, frame_events,
                    )
                except Exception as exc:
                    self._logger.warning(
                        "Frame callback failed at frame %s: %s",
                        packet.frame_index, exc,
                    )

            events.extend(frame_events)
            frames_processed += 1
            increment_frames(packet.timestamp_ms)

        return PipelineResult(
            frames_processed=frames_processed,
            violations=tuple(events),
        )

    # ------------------------------------------------------------------ #
    # Outbox dispatch (durable, network-independent)                       #
    # ------------------------------------------------------------------ #
    def _dispatch_to_outbox(self, frame, frame_events: list[ViolationEvent]) -> None:
        """Persist each violation + evidence image into the SQLite outbox."""
        outbox = self._outbox
        if outbox is None:
            return
        for event in frame_events:
            payload = _event_to_payload(event)
            image = self._render_evidence(frame, event)
            try:
                stored = outbox.enqueue(payload, image)
                if stored:
                    self._logger.info(
                        "Violation %s stored in outbox (pending=%d)",
                        event.event_id,
                        outbox.pending_count(),
                    )
            except Exception as exc:
                self._logger.error(
                    "Failed to store violation %s in outbox: %s",
                    event.event_id, exc,
                )

    def _render_evidence(self, frame, event: ViolationEvent) -> Optional[bytes]:
        """Annotate the violation frame (bbox + plate) and encode as JPEG.

        Returns None when OpenCV is unavailable so the JSON payload is
        still delivered without an image.
        """
        try:
            import cv2
        except ImportError:
            return None
        try:
            annotated = frame.copy()
            x1, y1, x2, y2 = (int(v) for v in event.bbox.as_xyxy())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"RED LIGHT | track {event.track_id}"
            if event.plate is not None:
                label += f" | {event.plate.text}"
            cv2.putText(
                annotated, label, (x1, max(0, y1 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
            )
            # Downscale huge frames (4000x3000) before encoding
            max_w = self._evidence_max_width
            if max_w > 0 and annotated.shape[1] > max_w:
                scale = max_w / annotated.shape[1]
                annotated = cv2.resize(
                    annotated,
                    (max_w, int(annotated.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            ok, jpeg = cv2.imencode(
                ".jpg", annotated,
                [cv2.IMWRITE_JPEG_QUALITY, self._evidence_quality],
            )
            return jpeg.tobytes() if ok else None
        except Exception as exc:
            self._logger.warning(
                "Evidence render failed for %s: %s", event.event_id, exc
            )
            return None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #
    def _publish_light_state(self, packet, light, stable_signal) -> None:
        """Expose the debounced light state to the control-plane API."""
        from edge_node.core.config import LightStateSnapshot, set_light_state

        set_light_state(
            LightStateSnapshot(
                state=stable_signal.state,
                confidence=stable_signal.confidence,
                stable=stable_signal.stable,
                frame_index=packet.frame_index,
                timestamp_ms=packet.timestamp_ms,
                source=light.source,
            )
        )

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
