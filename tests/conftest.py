"""Shared fixtures for edge_node tests."""
from __future__ import annotations

import pytest

from edge_node.core.config import (
    RedStabilizerConfig,
    TripwireConfig,
    ViolationConfig,
    set_active_tripwire,
)


@pytest.fixture(autouse=True)
def _reset_active_tripwire():
    """Reset global active tripwire giữa các test.

    ViolationDetector.update() đọc active tripwire qua get_active_tripwire()
    mỗi frame; nếu không reset, test trước có thể leak trạng thái vào test sau.
    """
    set_active_tripwire(None)
    yield
    set_active_tripwire(None)


@pytest.fixture
def horizontal_tripwire() -> TripwireConfig:
    """Vạch ngang ở y=400.

    NEGATIVE_TO_POSITIVE theo image coords (y trỏ xuống):
    side=-1 ↔ điểm có y< 400 (phía trên trong ảnh)
    side=+1 ↔ điểm có y > 400 (phía dưới trong ảnh)
    Nên NEGATIVE_TO_POSITIVE = đi từ phía trên xuống phía dưới.
    """
    from edge_node.core.contracts import Point, CrossingDirection

    return TripwireConfig(
        start=Point(100, 400),
        end=Point(800, 400),
        direction=CrossingDirection.NEGATIVE_TO_POSITIVE,
        deadband_px=3.0,
    )


@pytest.fixture
def violation_config(horizontal_tripwire: TripwireConfig) -> ViolationConfig:
    return ViolationConfig(
        tripwire=horizontal_tripwire,
        min_track_hits=1,
        stale_track_frames=60,
    )


@pytest.fixture
def stabilizer_config() -> RedStabilizerConfig:
    return RedStabilizerConfig(
        required_consecutive_frames=3,
        switch_consecutive_frames=5,
        min_confidence=0.5,
        unknown_tolerance_frames=3,
    )