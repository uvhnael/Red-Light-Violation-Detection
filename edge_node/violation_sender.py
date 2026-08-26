"""Sender chạy nền rút SQLite outbox đẩy lên Central Server.

Hành vi:
* Poll outbox mỗi ``outbox_flush_interval`` giây.
* Gửi các violation pending thành một POST batch tới
  ``/api/violations/batch`` (nếu server từ chối endpoint batch thì
  fallback sang POST đơn lẻ tới ``/api/violations``).
* Mất mạng thì các dòng giữ nguyên trạng thái ``pending`` — không mất
  dữ liệu, tick sau thử lại với exponential back-off.
* Sau khi JSON batch thành công, ảnh bằng chứng được upload từng cái
  lên media endpoint rồi đánh dấu ``media_sent``.

Sender hoàn toàn tuỳ chọn: khi ``enable_queue`` tắt thì pipeline vẫn
chạy, chỉ là không có đường giao hàng nào.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import requests

from edge_node.outbox import OutboxItem, ViolationOutbox
from edge_node.settings import EdgeNodeSettings

LOGGER = logging.getLogger(__name__)


def _media_upload_url(central_url: str, event_id: str) -> str:
    """Derive the media upload URL from the violations endpoint URL."""
    if "/api/violations" in central_url:
        base = central_url.rstrip("/").replace("/api/violations", "")
        return f"{base}/api/v1/violations/{event_id}/media"
    base = central_url.rstrip("/").rsplit("/", 1)[0]
    return f"{base}/v1/violations/{event_id}/media"


def _batch_url(central_url: str) -> str:
    return central_url.rstrip("/") + "/batch"


class ViolationSender:
    """Drains the outbox in batches while the network allows."""

    def __init__(self, outbox: ViolationOutbox, settings: EdgeNodeSettings) -> None:
        self._outbox = outbox
        self._settings = settings
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._consecutive_failures = 0

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="violation-sender"
        )
        self._thread.start()
        LOGGER.info(
            "Violation sender started (batch=%d, interval=%ds, url=%s)",
            self._settings.outbox_batch_size,
            self._settings.outbox_flush_interval,
            self._settings.central_server_url,
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._flush_once()
            except Exception as exc:  # never let the loop die
                LOGGER.error("Sender loop error: %s", exc)

            # Back off after repeated failures (capped at 60s)
            if self._consecutive_failures > 0:
                delay = min(60.0, self._settings.outbox_flush_interval
                            * (2 ** min(self._consecutive_failures, 4)))
            else:
                delay = float(self._settings.outbox_flush_interval)
            self._stop_event.wait(delay)

    def _flush_once(self) -> None:
        pending = self._outbox.fetch_pending(self._settings.outbox_batch_size)
        if pending:
            if self._send_batch(pending):
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
                self._outbox.bump_attempts([item.id for item in pending])
                return  # offline — retry later, skip media pass

        # Upload evidence images for already-sent violations
        if self._settings.outbox_media_upload:
            media = self._outbox.fetch_media_pending(5)
            for item in media:
                if self._stop_event.is_set():
                    return
                if self._upload_media(item):
                    self._outbox.mark_media_sent(item.event_id)
                else:
                    self._consecutive_failures += 1
                    return

    # ------------------------------------------------------------------ #
    # Batch delivery                                                       #
    # ------------------------------------------------------------------ #
    def _send_batch(self, items: List[OutboxItem]) -> bool:
        payloads = [item.payload for item in items]
        url = _batch_url(self._settings.central_server_url)
        try:
            response = requests.post(
                url,
                json={"violations": payloads},
                timeout=self._settings.push_timeout_seconds,
                headers={
                    "Content-Type": "application/json",
                    "X-Node-ID": self._settings.node_id,
                },
            )
            if response.status_code == 404:
                # Older server without the batch endpoint — fall back
                return self._send_individually(items)
            response.raise_for_status()
            body = response.json() if response.content else {}
            accepted = body.get("accepted", len(payloads))
            duplicates = body.get("duplicates", 0)
            self._outbox.mark_sent([item.id for item in items])
            LOGGER.info(
                "Batch sent: %d violation(s) -> %s (accepted=%s, duplicates=%s)",
                len(payloads), url, accepted, duplicates,
            )
            return True
        except requests.exceptions.RequestException as exc:
            LOGGER.warning(
                "Batch push failed (%d pending): %s", len(payloads), exc
            )
            return False

    def _send_individually(self, items: List[OutboxItem]) -> bool:
        """Fallback for servers without /api/violations/batch."""
        url = self._settings.central_server_url
        all_ok = True
        for item in items:
            try:
                response = requests.post(
                    url,
                    json=item.payload,
                    timeout=self._settings.push_timeout_seconds,
                    headers={
                        "Content-Type": "application/json",
                        "X-Node-ID": self._settings.node_id,
                    },
                )
                if response.status_code in (200, 201, 409):
                    self._outbox.mark_sent([item.id])
                else:
                    response.raise_for_status()
            except requests.exceptions.RequestException as exc:
                LOGGER.warning("Push failed for %s: %s", item.event_id, exc)
                all_ok = False
        return all_ok

    # ------------------------------------------------------------------ #
    # Media delivery                                                       #
    # ------------------------------------------------------------------ #
    def _upload_media(self, item: OutboxItem) -> bool:
        if not item.image:
            return True
        url = _media_upload_url(self._settings.central_server_url, item.event_id)
        try:
            response = requests.post(
                url,
                files={
                    "file": (
                        f"{item.event_id}.jpg",
                        io.BytesIO(item.image),
                        "image/jpeg",
                    )
                },
                timeout=self._settings.push_timeout_seconds,
            )
            if response.status_code == 404:
                # Violation record not on server yet (e.g. it was a
                # duplicate that got rejected) — mark done to avoid retrying
                LOGGER.debug(
                    "Media upload skipped for %s: violation not found on server",
                    item.event_id,
                )
                return True
            response.raise_for_status()
            LOGGER.info("Evidence image uploaded for %s", item.event_id)
            return True
        except requests.exceptions.RequestException as exc:
            LOGGER.warning("Media upload failed for %s: %s", item.event_id, exc)
            return False
