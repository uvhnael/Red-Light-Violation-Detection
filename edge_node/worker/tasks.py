"""Các Celery task do worker nền thực thi.

Nhiệm vụ duy nhất của module này: nhận payload violation an toàn JSON
từ queue Redis rồi chuyển tiếp tới Central Server qua HTTP POST. Retry
do cơ chế retry sẵn có của Celery đảm nhiệm nên pipeline camera không
bao giờ bị chặn vì lỗi mạng.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import requests

from edge_node.worker.celery_app import app
from edge_node.settings import get_settings

LOGGER = logging.getLogger(__name__)


@app.task(
    bind=True,
    name="edge_node.worker.tasks.push_violation_to_server",
    max_retries=None,  # controlled by settings
    acks_late=True,
)
def push_violation_to_server(
    self,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """POST a violation payload to the Central Server.

    On transient failures the task is automatically re-queued with
    exponential back-off so the edge pipeline is never stalled.
    """
    settings = get_settings()
    url = settings.central_server_url
    timeout = settings.push_timeout_seconds
    max_retries = settings.push_max_retries
    retry_delay = settings.push_retry_delay

    event_id = payload.get("event_id", "unknown")
    LOGGER.info("Pushing violation %s to %s", event_id, url)

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "X-Node-ID": settings.node_id,
            },
        )
        response.raise_for_status()
        LOGGER.info(
            "Violation %s pushed successfully (HTTP %s)",
            event_id,
            response.status_code,
        )
        return {
            "status": "ok",
            "event_id": event_id,
            "http_status": response.status_code,
        }

    except requests.exceptions.RequestException as exc:
        retries_left = max_retries - self.request.retries
        LOGGER.warning(
            "Push failed for %s (%s). Retries left: %s",
            event_id,
            exc,
            retries_left,
        )
        if self.request.retries < max_retries:
            raise self.retry(
                exc=exc,
                countdown=retry_delay * (2 ** self.request.retries),
                max_retries=max_retries,
            )
        LOGGER.error(
            "Permanently failed to push violation %s after %s retries",
            event_id,
            max_retries,
        )
        return {
            "status": "failed",
            "event_id": event_id,
            "error": str(exc),
        }
