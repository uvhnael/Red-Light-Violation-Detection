"""Tests cho edge_node.metrics — counter + thread safety."""
from __future__ import annotations

import threading

import pytest

from edge_node import metrics


@pytest.fixture(autouse=True)
def _reset_metrics():
    metrics.reset_for_test()
    yield
    metrics.reset_for_test()


def test_initial_state_is_zero():
    snap = metrics.get_metrics()
    assert snap.frames_processed == 0
    assert snap.violations_detected == 0
    assert snap.errors_skipped == 0
    assert snap.last_frame_ts_ms == 0.0


def test_increment_frames_updates_counter_and_timestamp():
    metrics.increment_frames(1234.5)
    metrics.increment_frames(1267.0)
    snap = metrics.get_metrics()
    assert snap.frames_processed == 2
    assert snap.last_frame_ts_ms == 1267.0  # lấy giá trị cuối


def test_increment_violations():
    metrics.increment_violations(1000.0)
    metrics.increment_violations(2000.0)
    metrics.increment_violations(3000.0)
    snap = metrics.get_metrics()
    assert snap.violations_detected == 3
    assert snap.last_violation_ts_ms == 3000.0


def test_increment_errors():
    metrics.increment_errors()
    metrics.increment_errors()
    snap = metrics.get_metrics()
    assert snap.errors_skipped == 2


def test_snapshot_is_independent_copy():
    """Mutating snapshot không được ảnh hưởng singleton."""
    metrics.increment_frames(100.0)
    snap1 = metrics.get_metrics()
    snap1.frames_processed = 999  # type: ignore[misc]
    snap2 = metrics.get_metrics()
    assert snap2.frames_processed == 1  # singleton không đổi


def test_concurrent_increments_are_thread_safe():
    N_THREADS = 10
    INCREMENTS_PER_THREAD = 1000

    def worker():
        for _ in range(INCREMENTS_PER_THREAD):
            metrics.increment_frames(1.0)

    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = metrics.get_metrics()
    expected = N_THREADS * INCREMENTS_PER_THREAD
    assert snap.frames_processed == expected, (
        f"thread-safe counter bị miss: got {snap.frames_processed}, want {expected}"
    )


def test_reset_clears_state():
    metrics.increment_frames(500.0)
    metrics.increment_violations(600.0)
    assert metrics.get_metrics().frames_processed == 1

    metrics.reset_for_test()
    snap = metrics.get_metrics()
    assert snap.frames_processed == 0
    assert snap.last_frame_ts_ms == 0.0


def test_started_at_is_set_on_construction():
    metrics.reset_for_test()
    snap = metrics.get_metrics()
    assert snap.started_at_ms > 0
    # Không reset khi increment — uptime tracking phải liên tục.
    metrics.increment_frames(1.0)
    snap2 = metrics.get_metrics()
    assert snap2.started_at_ms == snap.started_at_ms