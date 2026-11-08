"""Project event ledger — durable before async Hindsight retain."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sqlite_db import connect


@dataclass
class LedgerEvent:
    id: str
    project_id: str
    event_type: str
    content: str
    actor_role: str | None = None
    agent_id: str | None = None
    task_id: str | None = None
    evidence_path: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: str | None = None


class EventLedger:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = db_path
        self._conn = connect(db_path)

    def append(
        self,
        *,
        project_id: str,
        event_type: str,
        content: str,
        actor_role: str | None = None,
        agent_id: str | None = None,
        task_id: str | None = None,
        evidence_path: str | None = None,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
    ) -> LedgerEvent:
        eid = event_id or f"evt_{uuid.uuid4().hex[:12]}"
        meta = metadata or {}
        self._conn.execute(
            """
            INSERT INTO events (
                id, project_id, event_type, actor_role, agent_id, task_id,
                content, evidence_path, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                eid,
                project_id,
                event_type,
                actor_role,
                agent_id,
                task_id,
                content,
                evidence_path,
                json.dumps(meta),
            ),
        )
        self._conn.commit()
        return LedgerEvent(
            id=eid,
            project_id=project_id,
            event_type=event_type,
            content=content,
            actor_role=actor_role,
            agent_id=agent_id,
            task_id=task_id,
            evidence_path=evidence_path,
            metadata=meta,
        )

    def get(self, event_id: str) -> LedgerEvent | None:
        row = self._conn.execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_event(row)

    def list_for_project(self, project_id: str, *, limit: int = 100) -> list[LedgerEvent]:
        rows = self._conn.execute(
            """
            SELECT * FROM events WHERE project_id = ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (project_id, limit),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    @staticmethod
    def _row_to_event(row: Any) -> LedgerEvent:
        meta = {}
        try:
            meta = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            meta = {}
        return LedgerEvent(
            id=row["id"],
            project_id=row["project_id"],
            event_type=row["event_type"],
            content=row["content"],
            actor_role=row["actor_role"],
            agent_id=row["agent_id"],
            task_id=row["task_id"],
            evidence_path=row["evidence_path"],
            metadata=meta,
            created_at=row["created_at"],
        )

    def close(self) -> None:
        self._conn.close()
