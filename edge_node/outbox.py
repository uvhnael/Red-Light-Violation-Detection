"""Outbox SQLite bền vững cho các event vi phạm.

Violation được ghi vào đây ngay lúc phát hiện, *trước* mọi thao tác mạng.
Một :class:`edge_node.violation_sender.ViolationSender` chạy nền sẽ rút
outbox theo batch khi Central Server kết nối lại được.

Việc tách detection khỏi delivery đảm bảo:
* mất mạng không làm mất vi phạm (vẫn nằm trong SQLite),
* restart edge node sẽ gửi lại mọi bản còn pending,
* gửi theo batch hiệu quả thay vì một POST cho mỗi event.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    TEXT NOT NULL UNIQUE,
    payload     TEXT NOT NULL,
    image       BLOB,
    status      TEXT NOT NULL DEFAULT 'pending',
    media_sent  INTEGER NOT NULL DEFAULT 0,
    attempts    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    sent_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_outbox_status ON outbox (status, id);
"""


@dataclass(frozen=True)
class OutboxItem:
    """One pending violation row pulled from the outbox."""

    id: int
    event_id: str
    payload: Dict[str, Any]
    image: Optional[bytes]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ViolationOutbox:
    """Thread-safe SQLite store bridging detection and delivery."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # check_same_thread=False is safe because every access is guarded
        # by ``self._lock``.
        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        LOGGER.info("Violation outbox ready at %s", self._db_path)

    # ------------------------------------------------------------------ #
    # Write path (called from the detection loop)                          #
    # ------------------------------------------------------------------ #
    def enqueue(
        self,
        payload: Dict[str, Any],
        image: Optional[bytes] = None,
    ) -> bool:
        """Persist a violation. Returns False if the event_id already exists."""
        event_id = payload.get("event_id")
        if not event_id:
            LOGGER.warning("Refusing to enqueue payload without event_id")
            return False
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO outbox (event_id, payload, image, status, created_at)
                    VALUES (?, ?, ?, 'pending', ?)
                    """,
                    (
                        event_id,
                        json.dumps(payload, ensure_ascii=False),
                        image,
                        _utcnow(),
                    ),
                )
                self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            LOGGER.debug("Event %s already in outbox — skipped", event_id)
            return False
        except sqlite3.Error as exc:
            LOGGER.error("Failed to enqueue %s: %s", event_id, exc)
            return False

    # ------------------------------------------------------------------ #
    # Read path (called from the sender thread)                            #
    # ------------------------------------------------------------------ #
    def fetch_pending(self, limit: int = 20) -> List[OutboxItem]:
        """Pending violations for the JSON batch pass.

        KHÔNG đọc cột ``image`` (BLOB) ở đây: batch JSON chỉ cần payload,
        còn ảnh bằng chứng đi qua :meth:`fetch_media_pending` riêng. Tránh
        kéo cả blob vào bộ nhớ khi batch lớn.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, event_id, payload
                FROM outbox
                WHERE status = 'pending'
                ORDER BY id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            OutboxItem(
                id=row["id"],
                event_id=row["event_id"],
                payload=json.loads(row["payload"]),
                image=None,
            )
            for row in rows
        ]

    def mark_sent(self, ids: List[int]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            self._conn.execute(
                f"""
                UPDATE outbox
                SET status = 'sent', sent_at = ?
                WHERE id IN ({placeholders})
                """,
                [_utcnow(), *ids],
            )
            self._conn.commit()

    def bump_attempts(self, ids: List[int]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            self._conn.execute(
                f"""
                UPDATE outbox SET attempts = attempts + 1
                WHERE id IN ({placeholders})
                """,
                ids,
            )
            self._conn.commit()

    def mark_media_sent(self, event_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE outbox SET media_sent = 1 WHERE event_id = ?",
                (event_id,),
            )
            self._conn.commit()

    def fetch_media_pending(self, limit: int = 10) -> List[OutboxItem]:
        """Sent violations whose evidence image has not been uploaded yet."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, event_id, payload, image
                FROM outbox
                WHERE status = 'sent' AND media_sent = 0 AND image IS NOT NULL
                ORDER BY id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            OutboxItem(
                id=row["id"],
                event_id=row["event_id"],
                payload=json.loads(row["payload"]),
                image=row["image"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    # Introspection                                                        #
    # ------------------------------------------------------------------ #
    def pending_count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM outbox WHERE status = 'pending'"
            ).fetchone()
        return int(row["n"]) if row else 0

    def sent_count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM outbox WHERE status = 'sent'"
            ).fetchone()
        return int(row["n"]) if row else 0

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
