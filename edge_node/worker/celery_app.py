"""Ứng dụng Celery kết nối tới Redis broker cục bộ.

Chạy worker::

    celery -A edge_node.worker.celery_app worker --loglevel=info --concurrency=2
"""

from __future__ import annotations

from celery import Celery

from edge_node.settings import get_settings

_settings = get_settings()

app = Celery(
    "edge_node",
    broker=_settings.redis_url,
    backend=_settings.celery_result_backend,
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Ho_Chi_Minh",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
)

# Auto-discover tasks in edge_node.worker.tasks
app.autodiscover_tasks(["edge_node.worker"])
