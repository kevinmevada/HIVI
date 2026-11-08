"""Async retention queue — survives interruptions via SQLite."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from hivi.storage.sqlite_db import connect

from .models import MemoryStatus, MemoryType, RetainItem


@dataclass
class QueueItem:
    id: int
    event_id: str
    project_id: str
    payload: dict[str, Any]
    status: str
    attempts: int


class RetentionQueue:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self._conn = connect(db_path)

    def enqueue(
        self,
        *,
        event_id: str,
        project_id: str,
        content: str,
        memory_type: MemoryType | None = None,
        status: MemoryStatus = MemoryStatus.UNVERIFIED,
        source: str | None = None,
        metadata: dict[str, str] | None = None,
        department: str | None = None,
        agent_id: str | None = None,
        task_id: str | None = None,
        wait: bool = False,
    ) -> int:
        payload = {
            "content": content,
            "memory_type": memory_type.value if memory_type else None,
            "status": status.value,
            "source": source or event_id,
            "metadata": metadata or {},
            "department": department,
            "agent_id": agent_id,
            "task_id": task_id,
            "wait": wait,
        }
        cur = self._conn.execute(
            """
            INSERT INTO retention_queue (event_id, project_id, payload_json, status)
            VALUES (?, ?, ?, 'pending')
            """,
            (event_id, project_id, json.dumps(payload)),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def pending(self, *, limit: int = 50) -> list[QueueItem]:
        rows = self._conn.execute(
            """
            SELECT * FROM retention_queue
            WHERE status = 'pending'
            ORDER BY id ASC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._row(r) for r in rows]

    def mark_done(self, item_id: int) -> None:
        self._conn.execute(
            """
            UPDATE retention_queue
            SET status = 'done', updated_at = datetime('now')
            WHERE id = ?
            """,
            (item_id,),
        )
        self._conn.commit()

    def mark_failed(self, item_id: int, error: str) -> None:
        self._conn.execute(
            """
            UPDATE retention_queue
            SET status = 'pending',
                attempts = attempts + 1,
                last_error = ?,
                updated_at = datetime('now')
            WHERE id = ?
            """,
            (error[:500], item_id),
        )
        self._conn.commit()

    def drain(
        self,
        retain_fn: Callable[..., Any],
        *,
        limit: int = 50,
        max_attempts: int = 5,
    ) -> dict[str, int]:
        """Process pending items via retain_fn(project_id=..., items=..., **kwargs)."""
        stats = {"ok": 0, "fail": 0, "skipped": 0}
        for item in self.pending(limit=limit):
            if item.attempts >= max_attempts:
                self._conn.execute(
                    "UPDATE retention_queue SET status='dead' WHERE id=?",
                    (item.id,),
                )
                self._conn.commit()
                stats["skipped"] += 1
                continue
            p = item.payload
            mt = MemoryType(p["memory_type"]) if p.get("memory_type") else None
            st = MemoryStatus(p.get("status") or "unverified")
            try:
                retain_fn(
                    project_id=item.project_id,
                    items=[
                        RetainItem(
                            content=p["content"],
                            memory_type=mt,
                            status=st,
                            source=p.get("source"),
                            metadata=dict(p.get("metadata") or {}),
                        )
                    ],
                    department=p.get("department"),
                    agent_id=p.get("agent_id"),
                    task_id=p.get("task_id"),
                    wait=bool(p.get("wait", False)),
                )
                self.mark_done(item.id)
                stats["ok"] += 1
            except Exception as e:
                self.mark_failed(item.id, str(e))
                stats["fail"] += 1
        return stats

    @staticmethod
    def _row(r: Any) -> QueueItem:
        return QueueItem(
            id=int(r["id"]),
            event_id=r["event_id"],
            project_id=r["project_id"],
            payload=json.loads(r["payload_json"]),
            status=r["status"],
            attempts=int(r["attempts"]),
        )
