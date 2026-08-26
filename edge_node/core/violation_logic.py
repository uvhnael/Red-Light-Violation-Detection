"""Logic ổn định trạng thái đèn giao thông và phát hiện xe cắt vạch đã calibrate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from edge_node.core.config import RedStabilizerConfig, TripwireConfig, ViolationConfig
from edge_node.core.contracts import (
    CrossingDirection,
    LightObservation,
    LightState,
    Point,
    StableSignal,
    Track,
    ViolationEvent,
)
from edge_node.core.geometry import segments_intersect, side_of_line


@dataclass
class _TrackCrossingState:
    last_point: Point
    last_side: int
    last_nonzero_side: Optional[int]
    last_seen_frame: int
    violation_event_id: Optional[str] = None


class RedLightStabilizer:
    """Debounces noisy per-frame traffic-light classifications.

    The output becomes stable only after the same confident state has been seen
    for ``required_consecutive_frames``. Once stable, switching to a different
    state requires ``switch_consecutive_frames`` consecutive confident
    observations (hysteresis), so scattered misclassifications cannot flip the
    signal. Low-confidence and unknown readings do not immediately clear a
    stable red signal, but they will move the stable state back to ``UNKNOWN``
    after ``unknown_tolerance_frames``.
    """

    def __init__(self, config: RedStabilizerConfig) -> None:
        if config.required_consecutive_frames < 1:
            raise ValueError("required_consecutive_frames must be >= 1")
        if config.switch_consecutive_frames < 1:
            raise ValueError("switch_consecutive_frames must be >= 1")
        if not 0.0 <= config.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if config.unknown_tolerance_frames < 0:
            raise ValueError("unknown_tolerance_frames must be >= 0")
        self._config = config
        self._candidate_state = LightState.UNKNOWN
        self._candidate_count = 0
        self._stable_state = LightState.UNKNOWN
        self._stable_confidence = 0.0
        self._stable_since_frame: Optional[int] = None
        self._unknown_count = 0

    def update(self, observation: LightObservation, frame_index: int) -> StableSignal:
        confident = (
            observation.state != LightState.UNKNOWN
            and observation.confidence >= self._config.min_confidence
        )

        if not confident:
            self._unknown_count += 1
            if self._unknown_count > self._config.unknown_tolerance_frames:
                self._stable_state = LightState.UNKNOWN
                self._stable_confidence = observation.confidence
                self._stable_since_frame = None
            return self._signal()

        self._unknown_count = 0
        if observation.state == self._candidate_state:
            self._candidate_count += 1
        else:
            self._candidate_state = observation.state
            self._candidate_count = 1

        # Hysteresis: locking in the first state (from UNKNOWN) is cheap,
        # but flipping between two stable states needs more consecutive
        # evidence so scattered misclassifications cannot cause flicker.
        if self._stable_state == LightState.UNKNOWN:
            threshold = self._config.required_consecutive_frames
        else:
            threshold = self._config.switch_consecutive_frames

        if self._candidate_count >= threshold:
            if self._stable_state != observation.state:
                start = frame_index - self._candidate_count + 1
                self._stable_since_frame = max(0, start)
            self._stable_state = observation.state
            self._stable_confidence = observation.confidence

        return self._signal()

    def _signal(self) -> StableSignal:
        return StableSignal(
            state=self._stable_state,
            confidence=self._stable_confidence,
            stable=self._stable_state != LightState.UNKNOWN,
            stable_since_frame=self._stable_since_frame,
            observations=self._candidate_count,
        )


class Tripwire:
    """A two-point virtual stop line with directional crossing semantics."""

    def __init__(self, config: TripwireConfig) -> None:
        if config.deadband_px < 0.0:
            raise ValueError("deadband_px must be >= 0")
        if config.start == config.end:
            raise ValueError("Tripwire start and end points must differ")
        self._config = config

    @property
    def start(self) -> Point:
        return self._config.start

    @property
    def end(self) -> Point:
        return self._config.end

    @property
    def direction(self) -> CrossingDirection:
        return self._config.direction

    def side(self, point: Point) -> int:
        return side_of_line(point, self.start, self.end, self._config.deadband_px)

    def is_allowed_transition(self, previous_side: int, current_side: int) -> bool:
        if previous_side == 0 or current_side == 0 or previous_side == current_side:
            return False
        if self.direction == CrossingDirection.ANY:
            return True
        if self.direction == CrossingDirection.POSITIVE_TO_NEGATIVE:
            return previous_side > 0 and current_side < 0
        if self.direction == CrossingDirection.NEGATIVE_TO_POSITIVE:
            return previous_side < 0 and current_side > 0
        return False

    def crossed(
        self,
        previous_point: Point,
        current_point: Point,
        previous_side: int,
        current_side: int,
    ) -> bool:
        if not self.is_allowed_transition(previous_side, current_side):
            return False
        return segments_intersect(previous_point, current_point, self.start, self.end)


class ViolationDetector:
    """Detects red-light violations from confirmed tracks and stable signal state."""

    def __init__(self, config: ViolationConfig) -> None:
        if config.min_track_hits < 1:
            raise ValueError("min_track_hits must be >= 1")
        if config.stale_track_frames < 1:
            raise ValueError("stale_track_frames must be >= 1")
        self._config = config
        # tripwire=None (camera not calibrated yet) is allowed: update()
        # then short-circuits and never emits events until a stop line is
        # set through the control-plane API.
        self._tripwire = Tripwire(config.tripwire) if config.tripwire else None
        self._states: dict[int, _TrackCrossingState] = {}
        self._event_counter = 0

    def update(
        self,
        tracks: Sequence[Track],
        signal: StableSignal,
        frame_index: int,
        timestamp_ms: float,
    ) -> list[ViolationEvent]:
        from edge_node.core.config import get_active_tripwire

        active_tw = get_active_tripwire()
        if active_tw is None:
            # No stop line configured yet (operator has not drawn one on the
            # web UI): red-light violations are undefined without a reference
            # line, so never flag anything.
            self._states.clear()
            return []

        if active_tw != getattr(self._tripwire, "_config", None):
            self._tripwire = Tripwire(active_tw)
            self._states.clear()

        events: list[ViolationEvent] = []

        for track in tracks:
            current_point = track.crossing_point
            current_side = self._tripwire.side(current_point)
            prior = self._states.get(track.track_id)

            if prior is not None:
                previous_side = prior.last_nonzero_side or prior.last_side
                can_evaluate = previous_side != 0 and current_side != 0
                already_flagged = prior.violation_event_id is not None
                crossed = can_evaluate and self._tripwire.crossed(
                    prior.last_point,
                    current_point,
                    previous_side,
                    current_side,
                )

                if (
                    crossed
                    and signal.is_red
                    and track.hits >= self._config.min_track_hits
                    and not already_flagged
                ):
                    event = self._build_event(
                        track=track,
                        signal=signal,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        crossing_point=current_point,
                        previous_point=prior.last_point,
                        previous_side=previous_side,
                        current_side=current_side,
                    )
                    prior.violation_event_id = event.event_id
                    events.append(event)

            last_nonzero = current_side if current_side != 0 else (
                prior.last_nonzero_side if prior else None
            )
            violation_event_id = prior.violation_event_id if prior else None
            self._states[track.track_id] = _TrackCrossingState(
                last_point=current_point,
                last_side=current_side,
                last_nonzero_side=last_nonzero,
                last_seen_frame=frame_index,
                violation_event_id=violation_event_id,
            )

        self._drop_stale_tracks(frame_index)
        return events

    def _build_event(
        self,
        track: Track,
        signal: StableSignal,
        frame_index: int,
        timestamp_ms: float,
        crossing_point: Point,
        previous_point: Point,
        previous_side: int,
        current_side: int,
    ) -> ViolationEvent:
        self._event_counter += 1
        event_id = f"{self._config.event_prefix}-{frame_index:08d}-{track.track_id}"
        return ViolationEvent(
            event_id=event_id,
            track_id=track.track_id,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            crossing_point=crossing_point,
            previous_point=previous_point,
            bbox=track.bbox,
            light_state=signal.state,
            light_confidence=signal.confidence,
            previous_side=previous_side,
            current_side=current_side,
            metadata={
                "stable_since_frame": signal.stable_since_frame,
                "track_hits": track.hits,
                "event_sequence": self._event_counter,
            },
        )

    def _drop_stale_tracks(self, frame_index: int) -> None:
        stale_ids = [
            track_id
            for track_id, state in self._states.items()
            if frame_index - state.last_seen_frame > self._config.stale_track_frames
        ]
        for track_id in stale_ids:
            del self._states[track_id]
