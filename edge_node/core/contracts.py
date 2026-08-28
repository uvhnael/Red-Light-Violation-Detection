"""Các contract chung (data class + protocol) của pipeline phát hiện."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence


Frame = Any
Metadata = Mapping[str, Any]


class LightState(str, Enum):
    """Traffic light states used by classifiers and stabilization logic."""

    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"


class CrossingDirection(str, Enum):
    """Allowed direction of travel across the oriented stop-line segment."""

    ANY = "any"
    POSITIVE_TO_NEGATIVE = "positive_to_negative"
    NEGATIVE_TO_POSITIVE = "negative_to_positive"


@dataclass(frozen=True)
class Point:
    """Image-space point in pixels."""

    x: float
    y: float

    def __post_init__(self) -> None:
        if not isfinite(self.x) or not isfinite(self.y):
            raise ValueError(f"Point coordinates must be finite: {self!r}")


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in xyxy pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(isfinite(value) for value in values):
            raise ValueError(f"Bounding box coordinates must be finite: {self!r}")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"Invalid bounding box geometry: {self!r}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return Point((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Point:
        return Point((self.x1 + self.x2) / 2.0, self.y2)

    def iou(self, other: "BoundingBox") -> float:
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h
        union = self.area + other.area - intersection
        return 0.0 if union <= 0.0 else intersection / union

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True)
class Detection:
    """Object detector output."""

    bbox: BoundingBox
    label: str
    confidence: float
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Detection confidence must be in [0, 1]: {self.confidence}")
        if not self.label:
            raise ValueError("Detection label must be non-empty")


@dataclass(frozen=True)
class Track:
    """Tracked object state exposed to downstream logic."""

    track_id: int
    bbox: BoundingBox
    label: str
    confidence: float
    age: int
    hits: int
    time_since_update: int
    metadata: Metadata = field(default_factory=dict)

    @property
    def crossing_point(self) -> Point:
        """Point used for tripwire tests.

        Dùng tâm box (center) thay vì bottom-center: với camera góc nghiêng,
        bottom-center là điểm đuôi xe chạm đất — xe có thể đã thò đầu qua vạch
        mà điểm này vẫn chưa qua, gây bỏ sót. Tâm box đại diện cho thân xe và
        qua vạch sớm hơn, khớp với cảm nhận "xe đã vượt" hơn.
        """

        return self.bbox.center


@dataclass(frozen=True)
class LightObservation:
    """Raw traffic-light classifier output for one frame."""

    state: LightState
    confidence: float
    source: str = "unknown"
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Light confidence must be in [0, 1]: {self.confidence}")


@dataclass(frozen=True)
class PlateObservation:
    """Optional OCR result associated with a track."""

    text: str
    confidence: float
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class FramePacket:
    """One frame and its timing metadata."""

    frame_index: int
    timestamp_ms: float
    image: Frame


@dataclass(frozen=True)
class StableSignal:
    """Debounced traffic-light state."""

    state: LightState
    confidence: float
    stable: bool
    stable_since_frame: Optional[int]
    observations: int

    @property
    def is_red(self) -> bool:
        return self.stable and self.state == LightState.RED


@dataclass(frozen=True)
class ViolationEvent:
    """Evidence record for a confirmed tripwire crossing under stabilized red."""

    event_id: str
    track_id: int
    frame_index: int
    timestamp_ms: float
    crossing_point: Point
    previous_point: Point
    bbox: BoundingBox
    light_state: LightState
    light_confidence: float
    previous_side: int
    current_side: int
    plate: Optional[PlateObservation] = None
    metadata: Metadata = field(default_factory=dict)


class ObjectDetector(Protocol):
    """Interface for YOLO or any other object detector."""

    def detect(self, frame: Frame, frame_index: int, timestamp_ms: float) -> Sequence[Detection]:
        """Return object detections for the frame."""


class MultiObjectTracker(Protocol):
    """Interface for a vehicle tracker."""

    def update(
        self,
        detections: Sequence[Detection],
        frame_index: int,
        timestamp_ms: float,
    ) -> Sequence[Track]:
        """Update tracks and return currently visible tracks."""


class TrafficLightClassifier(Protocol):
    """Interface for traffic-light state classification."""

    def classify(self, frame: Frame, frame_index: int, timestamp_ms: float) -> LightObservation:
        """Return the current traffic-light observation."""


class PlateRecognizer(Protocol):
    """Optional OCR interface."""

    def recognize(self, frame: Frame, track: Track) -> Optional[PlateObservation]:
        """Return a plate reading for the supplied track, if available."""


class FrameSource(Protocol):
    """Iterable source of frames."""

    def __iter__(self) -> Iterable[FramePacket]:
        """Yield frame packets."""
