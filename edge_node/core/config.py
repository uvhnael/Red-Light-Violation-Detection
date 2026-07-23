"""Runtime configuration for the clean pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import threading

from edge_node.core.contracts import CrossingDirection, Point



@dataclass(frozen=True)
class TripwireConfig:
    """Calibrated stop-line definition."""

    start: Point
    end: Point
    direction: CrossingDirection = CrossingDirection.ANY
    deadband_px: float = 3.0


@dataclass(frozen=True)
class RedStabilizerConfig:
    """Debounce settings for traffic-light state."""

    required_consecutive_frames: int = 3
    min_confidence: float = 0.70
    unknown_tolerance_frames: int = 5


@dataclass(frozen=True)
class TrackingConfig:
    """IoU tracker settings."""

    iou_threshold: float = 0.30
    max_age_frames: int = 20
    min_hits: int = 2
    min_detection_confidence: float = 0.30


@dataclass(frozen=True)
class ViolationConfig:
    """Violation rule settings."""

    tripwire: TripwireConfig
    min_track_hits: int = 2
    stale_track_frames: int = 60
    event_prefix: str = "rlv"


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level processing configuration."""

    input_path: Optional[Path]
    events_output_path: Path
    max_frames: Optional[int] = None
    export_format: str = "json"
    enable_ocr: bool = False


_ACTIVE_TRIPWIRE: Optional[TripwireConfig] = None
_TRIPWIRE_LOCK = threading.Lock()

def set_active_tripwire(config: TripwireConfig) -> None:
    global _ACTIVE_TRIPWIRE
    with _TRIPWIRE_LOCK:
        _ACTIVE_TRIPWIRE = config

def get_active_tripwire() -> Optional[TripwireConfig]:
    with _TRIPWIRE_LOCK:
        return _ACTIVE_TRIPWIRE
