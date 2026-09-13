"""Test cho violation_logic: deadband anchor, no-tripwire gate, hysteresis.

Các bug đã fix và đang được cover:
* Deadband anchor (2026-08-28): track đi vào deadband trước khi qua vạch
  bị bỏ sót vì previous_point = deadband point (đã ở bên kia vạch).
  → Test `test_crossing_via_deadband_detected`.
* No-tripwire gate (2026-08-25): khi tripwire=None (chưa calibrate), detector
  phải trả về [] ngay cả khi signal RED stable.
  → Test `test_no_tripwire_means_no_events`.
* Crossing direction filter: positive_to_negative chỉ chấp nhận transition
  đúng chiều.
  → Test `test_direction_filter_rejects_wrong_way`.
* Each track chỉ được flag một lần cho đến khi state bị clear.
  → Test `test_each_track_flagged_once`.
* Red-light signal chưa stable → không flag.
  → Test `test_non_red_signal_no_event`.
"""

from __future__ import annotations

import pytest

from edge_node.core.config import (
    RedStabilizerConfig,
    TripwireConfig,
    ViolationConfig,
    set_active_tripwire,
)
from edge_node.core.contracts import (
    BoundingBox,
    CrossingDirection,
    LightObservation,
    LightState,
    Point,
    StableSignal,
    Track,
)
from edge_node.core.violation_logic import (
    RedLightStabilizer,
    Tripwire,
    ViolationDetector,
)


def _stable_red_signal(frame_index: int = 0) -> StableSignal:
    return StableSignal(
        state=LightState.RED,
        confidence=0.9,
        stable=True,
        stable_since_frame=frame_index,
        observations=5,
    )


def _track(track_id: int, cx: float, cy: float, label: str = "car") -> Track:
    """Tạo Track với ``crossing_point`` = (cx, cy).

    ``cy`` là toạ độ ĐIỂM NEO (bottom-center của bbox) — không phải tâm box:
    ``Track.crossing_point`` trả về ``bbox.bottom_center``. Vì vậy đáy box đặt
    tại cy và thân xe dựng ngược lên trên. hits=1 để pass min_track_hits=1.
    """
    half = 25.0
    height = 50.0
    return Track(
        track_id=track_id,
        bbox=BoundingBox(cx - half, cy - height, cx + half, cy),
        label=label,
        confidence=0.9,
        age=10,
        hits=1,
        time_since_update=0,
    )


class TestViolationDetectorCrossing:
    """Vạch ngang y=400.

    Image coords: y trỏ xuống.
    * y=600 → side=+1 (phía dưới trong ảnh)
    * y=200 → side=-1 (phía trên trong ảnh)
    * NEGATIVE_TO_POSITIVE = đi từ phía trên xuống phía dưới (y tăng).
    """

    def setup_method(self) -> None:
        self.tripwire = TripwireConfig(
            start=Point(0, 400),
            end=Point(1000, 400),
            direction=CrossingDirection.NEGATIVE_TO_POSITIVE,
            deadband_px=3.0,
        )
        set_active_tripwire(self.tripwire)
        self.detector = ViolationDetector(
            ViolationConfig(tripwire=self.tripwire, min_track_hits=1)
        )

    def test_direct_crossing_detected(self):
        # Frame 0: track ở phía trên vạch (y=200, side=-1)
        # Frame 1: track ở phía dưới vạch (y=600, side=+1) → CROSS
        track_above = _track(1, 500, 200)
        track_below = _track(1, 500, 600)

        assert self.detector.update([track_above], _stable_red_signal(), 0, 0.0) == []
        events = self.detector.update([track_below], _stable_red_signal(), 1, 33.0)
        assert len(events) == 1
        assert events[0].track_id == 1
        assert events[0].frame_index == 1

    def test_crossing_via_deadband_detected(self):
        """Regression test cho fix deadband 2026-08-28.

        Track đi từ phía trên vạch (y=200, side=-1) → vào deadband (y=397..403,
        side=0) → xuống phía dưới (y=600, side=+1).
        Nếu dùng điểm deadband làm previous_point, đoạn test chỉ dài ~3px và
        không giao vạch về hình học → bug. Fix: dùng last_nonzero_point.
        """
        track_above = _track(2, 500, 200)  # y=200, side=-1
        track_in_deadband = _track(2, 500, 402)  # y=402, side=0 (trong deadband ±3)
        track_below = _track(2, 500, 600)  # y=600, side=+1

        assert self.detector.update([track_above], _stable_red_signal(), 0, 0.0) == []
        # Track rơi vào deadband — KHÔNG có event
        assert (
            self.detector.update([track_in_deadband], _stable_red_signal(), 1, 33.0)
            == []
        )
        # Track đã xuống phía dưới — event PHẢI xuất hiện dù previous frame là deadband
        events = self.detector.update([track_below], _stable_red_signal(), 2, 66.0)
        assert len(events) == 1, f"deadband anchor regressed; got {events}"
        assert events[0].track_id == 2

    def test_each_track_flagged_once(self):
        """Track sau khi flag xong thì không flag lại cho đến khi clear state."""
        track_above = _track(3, 500, 200)
        track_below = _track(3, 500, 600)
        track_below_again = _track(3, 500, 650)  # vẫn phía dưới

        self.detector.update([track_above], _stable_red_signal(), 0, 0.0)
        first = self.detector.update([track_below], _stable_red_signal(), 1, 33.0)
        second = self.detector.update(
            [track_below_again], _stable_red_signal(), 2, 66.0
        )
        assert len(first) == 1
        assert len(second) == 0, f"track bị flag 2 lần: {second}"

    def test_stationary_in_deadband_no_event(self):
        """Track đứng yên trong deadband vẫn không được flag."""
        track_above = _track(4, 500, 200)
        track_in_deadband = _track(4, 500, 401)

        self.detector.update([track_above], _stable_red_signal(), 0, 0.0)
        events = self.detector.update(
            [track_in_deadband], _stable_red_signal(), 1, 33.0
        )
        assert events == []

    def test_non_red_signal_no_event(self):
        track_above = _track(5, 500, 200)
        track_below = _track(5, 500, 600)

        green_signal = StableSignal(
            state=LightState.GREEN,
            confidence=0.9,
            stable=True,
            stable_since_frame=0,
            observations=5,
        )
        self.detector.update([track_above], green_signal, 0, 0.0)
        events = self.detector.update([track_below], green_signal, 1, 33.0)
        assert events == []

    def test_direction_filter_rejects_wrong_way(self):
        """Vạch NEGATIVE_TO_POSITIVE: đi từ dưới lên (y=600 → y=200) bị từ chối.

        NEGATIVE_TO_POSITIVE chỉ chấp nhận previous_side< 0 → current_side > 0.
        Image coords: y=200 (phía trên) → side=-1; y=600 (phía dưới) → side=+1.
        Nên đi từ dưới lên trên (+1 → -1) bị từ chối.
        """
        # Direction cho phép đi từ trên xuống dưới (-1 → +1)
        tw = TripwireConfig(
            start=Point(0, 400),
            end=Point(1000, 400),
            direction=CrossingDirection.NEGATIVE_TO_POSITIVE,
            deadband_px=3.0,
        )
        set_active_tripwire(tw)
        det = ViolationDetector(ViolationConfig(tripwire=tw, min_track_hits=1))

        track_below = _track(6, 500, 600)  # side=+1
        track_above = _track(
            6, 500, 200
        )  # side=-1 — đi từ dưới lên (SAI CHIỀU NEGATIVE_TO_POSITIVE)

        det.update([track_below], _stable_red_signal(), 0, 0.0)
        events = det.update([track_above], _stable_red_signal(), 1, 33.0)
        assert events == [], f"sai chiều nhưng vẫn flag: {events}"

    def test_direction_filter_accepts_correct_way(self):
        """Smoke test: NEGATIVE_TO_POSITIVE cho phép đi từ trên xuống (-1 → +1)."""
        tw = TripwireConfig(
            start=Point(0, 400),
            end=Point(1000, 400),
            direction=CrossingDirection.NEGATIVE_TO_POSITIVE,
            deadband_px=3.0,
        )
        set_active_tripwire(tw)
        det = ViolationDetector(ViolationConfig(tripwire=tw, min_track_hits=1))

        track_above = _track(6, 500, 200)  # side=-1
        track_below = _track(6, 500, 600)  # side=+1 — đi từ trên xuống (ĐÚNG CHIỀU)

        det.update([track_above], _stable_red_signal(), 0, 0.0)
        events = det.update([track_below], _stable_red_signal(), 1, 33.0)
        assert len(events) == 1

    def test_no_tripwire_means_no_events(self):
        """Khi tripwire=None (chưa calibrate), detector phải trả về [] ngay cả khi RED stable."""
        set_active_tripwire(None)
        det = ViolationDetector(ViolationConfig(tripwire=None, min_track_hits=1))

        track_above = _track(7, 500, 200)
        track_below = _track(7, 500, 600)

        assert det.update([track_above], _stable_red_signal(), 0, 0.0) == []
        events = det.update([track_below], _stable_red_signal(), 1, 33.0)
        assert events == [], "no-tripwire gate regressed"

    def test_min_track_hits_filter(self):
        """Track mới xuất hiện (hits < min_track_hits) bị bỏ qua để tránh noise."""
        tw = TripwireConfig(
            start=Point(0, 400),
            end=Point(1000, 400),
            direction=CrossingDirection.NEGATIVE_TO_POSITIVE,
        )
        set_active_tripwire(tw)
        det = ViolationDetector(ViolationConfig(tripwire=tw, min_track_hits=3))

        # hits=2 — chưa đạt ngưỡng. Box dựng sao cho BOTTOM-CENTER = (500, 200)
        # (Track.crossing_point = bbox.bottom_center).
        young_track_above = Track(
            track_id=8,
            bbox=BoundingBox(475, 150, 525, 200),
            label="car",
            confidence=0.9,
            age=2,
            hits=2,
            time_since_update=0,
        )
        # hits=3 — đạt ngưỡng
        mature_track_above = Track(
            track_id=8,
            bbox=BoundingBox(475, 150, 525, 200),
            label="car",
            confidence=0.9,
            age=3,
            hits=3,
            time_since_update=0,
        )
        mature_track_below = Track(
            track_id=8,
            bbox=BoundingBox(475, 550, 525, 600),
            label="car",
            confidence=0.9,
            age=4,
            hits=4,
            time_since_update=0,
        )

        det.update([young_track_above], _stable_red_signal(), 0, 0.0)
        events = det.update([mature_track_above], _stable_red_signal(), 1, 33.0)
        assert events == [], "min_track_hits filter không hoạt động"
        events = det.update([mature_track_below], _stable_red_signal(), 2, 66.0)
        assert len(events) == 1


class TestRedLightStabilizer:
    """Debounce + hysteresis cho traffic light."""

    def _confident_red(self) -> LightObservation:
        return LightObservation(state=LightState.RED, confidence=0.9, source="test")

    def _confident_green(self) -> LightObservation:
        return LightObservation(state=LightState.GREEN, confidence=0.9, source="test")

    def _unknown(self, confidence: float = 0.0) -> LightObservation:
        return LightObservation(
            state=LightState.UNKNOWN, confidence=confidence, source="test"
        )

    def test_first_state_locks_after_required_frames(self):
        cfg = RedStabilizerConfig(
            required_consecutive_frames=3,
            switch_consecutive_frames=5,
            min_confidence=0.5,
            unknown_tolerance_frames=2,
        )
        stab = RedLightStabilizer(cfg)

        # Frame 0-1: chưa đủ → UNKNOWN
        for i in range(2):
            sig = stab.update(self._confident_red(), i)
            assert not sig.stable

        # Frame 2: đạt ngưỡng 3 → RED stable
        sig = stab.update(self._confident_red(), 2)
        assert sig.stable
        assert sig.state == LightState.RED

    def test_low_confidence_does_not_lock(self):
        cfg = RedStabilizerConfig(
            required_consecutive_frames=2,
            switch_consecutive_frames=5,
            min_confidence=0.7,
            unknown_tolerance_frames=2,
        )
        stab = RedLightStabilizer(cfg)

        weak = LightObservation(state=LightState.RED, confidence=0.5, source="test")
        for i in range(5):
            sig = stab.update(weak, i)
        assert not sig.stable

    def test_hysteresis_blocks_scattered_misclassification(self):
        """RED stable → 1 frame GREEN lẻ loi KHÔNG đảo signal."""
        cfg = RedStabilizerConfig(
            required_consecutive_frames=2,
            switch_consecutive_frames=5,
            min_confidence=0.5,
            unknown_tolerance_frames=3,
        )
        stab = RedLightStabilizer(cfg)

        # Lock RED ở frame 0-1
        stab.update(self._confident_red(), 0)
        stab.update(self._confident_red(), 1)
        assert stab.update(self._confident_red(), 2).stable

        # Một frame GREEN lẻ
        sig = stab.update(self._confident_green(), 3)
        assert sig.stable, "hysteresis không hoạt động — 1 frame GREEN đã đảo RED"
        assert sig.state == LightState.RED

    def test_unknown_after_tolerance_returns_to_unknown(self):
        cfg = RedStabilizerConfig(
            required_consecutive_frames=2,
            switch_consecutive_frames=5,
            min_confidence=0.5,
            unknown_tolerance_frames=2,
        )
        stab = RedLightStabilizer(cfg)

        stab.update(self._confident_red(), 0)
        stab.update(self._confident_red(), 1)
        assert stab.update(self._confident_red(), 2).stable

        # unknown_count > unknown_tolerance_frames (2) mới reset → cần 3 frame unknown
        stab.update(self._unknown(), 3)
        assert stab.update(self._unknown(), 4).state == LightState.RED
        sig = stab.update(self._unknown(), 5)
        assert sig.state == LightState.UNKNOWN

    def test_constructor_validates(self):
        with pytest.raises(ValueError):
            RedLightStabilizer(RedStabilizerConfig(required_consecutive_frames=0))
        with pytest.raises(ValueError):
            RedLightStabilizer(RedStabilizerConfig(switch_consecutive_frames=0))
        with pytest.raises(ValueError):
            RedLightStabilizer(RedStabilizerConfig(min_confidence=1.5))
        with pytest.raises(ValueError):
            RedLightStabilizer(RedStabilizerConfig(unknown_tolerance_frames=-1))


class TestTripwire:
    def test_endpoints_must_differ(self):
        with pytest.raises(ValueError, match="differ"):
            Tripwire(TripwireConfig(start=Point(0, 0), end=Point(0, 0)))

    def test_negative_deadband_rejected(self):
        with pytest.raises(ValueError, match="deadband"):
            Tripwire(
                TripwireConfig(
                    start=Point(0, 0),
                    end=Point(100, 0),
                    deadband_px=-1.0,
                )
            )

    def test_crossed_true_for_simple_transition(self):
        # Tripwire y=400, NEGATIVE_TO_POSITIVE: phía trên (y=200) → phía dưới (y=600)
        tw = Tripwire(
            TripwireConfig(
                start=Point(0, 400),
                end=Point(1000, 400),
                direction=CrossingDirection.NEGATIVE_TO_POSITIVE,
            )
        )
        # y=200 → side=-1; y=600 → side=+1
        assert tw.crossed(Point(500, 200), Point(500, 600), -1, +1)
        # Cùng side → không cross
        assert not tw.crossed(Point(500, 600), Point(600, 650), +1, +1)

    def test_crossed_false_when_only_one_side_moves(self):
        tw = Tripwire(
            TripwireConfig(
                start=Point(0, 400),
                end=Point(1000, 400),
                direction=CrossingDirection.ANY,
            )
        )
        # Cùng side, không phải crossing
        assert not tw.crossed(Point(500, 600), Point(600, 650), +1, +1)
