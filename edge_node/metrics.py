"""Lightweight in-process metrics cho edge pipeline.

Đơn giản cố ý: chỉ 4 counter + 1 timestamp, đủ cho /health endpoint
và để central dashboard nhìn thấy trạng thái runtime của node biên.

KHÔNG phải Prometheus/OTel:
* edge node chỉ chạy 1 instance → không cần pull model.
* Central thu thập qua /api/health (đã có sẵn) — push-style không cần.
* Triển khai Prometheus tăng phụ thuộc (client + scrape config) mà không
  có lợi tương xứng ở quy mô hiện tại.

Khi scale lên nhiều node, có thể thay bằng OTel exporter — interface
đã giữ tối giản nên việc thay không ảnh hưởng caller.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class PipelineMetrics:
    """Counter + timestamp cho một process edge node.

    Mọi field chỉ tăng/tick — không reset trong suốt uptime. Reset
    chỉ khi restart pipeline (hiếm) → central dashboard thấy "violations
    tăng lên" là signal thật.
    """

    frames_processed: int = 0
    violations_detected: int = 0
    errors_skipped: int = 0  # Frame bị skip do exception (xem pipeline.process try/except)
    last_frame_ts_ms: float = 0.0  # timestamp_ms của frame cuối
    last_violation_ts_ms: float = 0.0
    started_at_ms: float = field(default_factory=lambda: time.time() * 1000.0)


# Module-level singleton — các thread-safe nhờ _lock.
_metrics = PipelineMetrics()
_lock = threading.Lock()


def get_metrics() -> PipelineMetrics:
    """Trả về snapshot — copy dataclass (immutable) để caller không mutate state."""
    with _lock:
        # dataclass replace() tạo bản sao — atomic từ quan điểm caller.
        from dataclasses import replace
        return replace(_metrics)


def increment_frames(timestamp_ms: float) -> None:
    """Đếm thêm 1 frame đã xử lý và cập nhật timestamp cuối."""
    with _lock:
        _metrics.frames_processed += 1
        _metrics.last_frame_ts_ms = timestamp_ms


def increment_violations(timestamp_ms: float) -> None:
    """Đếm thêm 1 violation đã ghi nhận (chưa gửi central)."""
    with _lock:
        _metrics.violations_detected += 1
        _metrics.last_violation_ts_ms = timestamp_ms


def increment_errors() -> None:
    """Đếm frame bị skip do exception — tăng cao bất thường = model/corrupt frame issue."""
    with _lock:
        _metrics.errors_skipped += 1


def reset_for_test() -> None:
    """Chỉ test mới gọi — production pipeline KHÔNG nên reset giữa uptime."""
    global _metrics
    with _lock:
        _metrics = PipelineMetrics()