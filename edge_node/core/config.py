"""Cấu hình runtime của pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import threading

from edge_node.core.contracts import CrossingDirection, LightState, Point



@dataclass(frozen=True)
class TripwireConfig:
    """Calibrated stop-line definition."""

    start: Point
    end: Point
    direction: CrossingDirection = CrossingDirection.ANY
    deadband_px: float = 3.0


@dataclass(frozen=True)
class RedStabilizerConfig:
    """Debounce settings for traffic-light state.

    ``required_consecutive_frames`` locks in the first stable state (from
    UNKNOWN).  ``switch_consecutive_frames`` is the hysteresis threshold to
    leave an already-stable state for a different one — set it higher than
    ``required_consecutive_frames`` so scattered misclassifications cannot
    flip a stable red/green signal.
    """

    required_consecutive_frames: int = 3
    switch_consecutive_frames: int = 8
    min_confidence: float = 0.70
    unknown_tolerance_frames: int = 5


@dataclass(frozen=True)
class ViolationConfig:
    """Violation rule settings.

    ``tripwire`` may be ``None`` while the camera is not calibrated yet —
    the violation detector then stays disabled (no events are produced).
    """

    tripwire: Optional[TripwireConfig]
    min_track_hits: int = 2
    stale_track_frames: int = 60
    event_prefix: str = "rlv"


# Active tripwire override — None cho tới khi operator kẻ vạch trên web UI.
# The running pipeline re-reads this every frame so calibration takes effect live.
_ACTIVE_TRIPWIRE: Optional[TripwireConfig] = None
_TRIPWIRE_LOCK = threading.Lock()

def set_active_tripwire(config: TripwireConfig) -> None:
    global _ACTIVE_TRIPWIRE
    with _TRIPWIRE_LOCK:
        _ACTIVE_TRIPWIRE = config

def get_active_tripwire() -> Optional[TripwireConfig]:
    with _TRIPWIRE_LOCK:
        return _ACTIVE_TRIPWIRE


# Active traffic-light ROI override as (x, y, w, h). When set, the runtime
# classifier prefers this over its own ROI so the web dashboard can re-calibrate.
_ACTIVE_LIGHT_ROI: Optional[tuple[int, int, int, int]] = None
_LIGHT_ROI_LOCK = threading.Lock()

def set_active_light_roi(roi: Optional[tuple[int, int, int, int]]) -> None:
    global _ACTIVE_LIGHT_ROI
    with _LIGHT_ROI_LOCK:
        _ACTIVE_LIGHT_ROI = roi

def get_active_light_roi() -> Optional[tuple[int, int, int, int]]:
    with _LIGHT_ROI_LOCK:
        return _ACTIVE_LIGHT_ROI


# Latest traffic-light state published by the running pipeline each frame.
# The control-plane API reads this so the web dashboard can show the live
# signal without re-running the classifier.
@dataclass(frozen=True)
class LightStateSnapshot:
    """One frame's debounced traffic-light state."""

    state: LightState
    confidence: float
    stable: bool
    frame_index: int
    timestamp_ms: float
    source: str = "unknown"


_LIGHT_STATE: Optional[LightStateSnapshot] = None
_LIGHT_STATE_LOCK = threading.Lock()

def set_light_state(snapshot: LightStateSnapshot) -> None:
    global _LIGHT_STATE
    with _LIGHT_STATE_LOCK:
        _LIGHT_STATE = snapshot

def get_light_state() -> Optional[LightStateSnapshot]:
    with _LIGHT_STATE_LOCK:
        return _LIGHT_STATE
