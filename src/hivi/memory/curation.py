"""Corrections, supersession, and invalidation (HIVI governance layer)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from hivi.storage.sqlite_db import connect

from .models import MemoryStatus, RecalledMemory
from .policies import ActorRole, assert_can_correct


@dataclass
class CorrectionRecord:
    id: str
    project_id: str
    target_memory_key: str
    action: str  # supersede | invalidate | dispute
    actor_role: str
    reason: str | None = None
    new_content: str | None = None


class CorrectionStore:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self._conn = connect(db_path)

    def apply(
        self,
        *,
        project_id: str,
        target_memory_key: str,
        action: str,
        actor_role: ActorRole | str,
        reason: str | None = None,
        new_content: str | None = None,
    ) -> CorrectionRecord:
        assert_can_correct(actor_role)
        if action not in {"supersede", "invalidate", "dispute"}:
            raise ValueError(f"unsupported action: {action}")
        cid = f"corr_{uuid.uuid4().hex[:12]}"
        self._conn.execute(
            """
            INSERT INTO memory_corrections (
                id, project_id, target_memory_key, action, reason, actor_role, new_content
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cid,
                project_id,
                target_memory_key,
                action,
                reason,
                str(actor_role),
                new_content,
            ),
        )
        self._conn.commit()
        return CorrectionRecord(
            id=cid,
            project_id=project_id,
            target_memory_key=target_memory_key,
            action=action,
            actor_role=str(actor_role),
            reason=reason,
            new_content=new_content,
        )

    def for_project(self, project_id: str) -> list[CorrectionRecord]:
        rows = self._conn.execute(
            "SELECT * FROM memory_corrections WHERE project_id = ? ORDER BY created_at",
            (project_id,),
        ).fetchall()
        return [
            CorrectionRecord(
                id=r["id"],
                project_id=r["project_id"],
                target_memory_key=r["target_memory_key"],
                action=r["action"],
                actor_role=r["actor_role"],
                reason=r["reason"],
                new_content=r["new_content"],
            )
            for r in rows
        ]

    def blocked_keys(self, project_id: str) -> dict[str, str]:
        """memory_key → effective status after corrections."""
        out: dict[str, str] = {}
        for c in self.for_project(project_id):
            if c.action == "supersede":
                out[c.target_memory_key] = MemoryStatus.SUPERSEDED.value
            elif c.action == "invalidate":
                out[c.target_memory_key] = MemoryStatus.INVALIDATED.value
            elif c.action == "dispute":
                out[c.target_memory_key] = MemoryStatus.DISPUTED.value
        return out


def filter_current_memories(
    results: list[RecalledMemory],
    *,
    blocked: dict[str, str] | None = None,
    drop_statuses: set[str] | None = None,
) -> list[RecalledMemory]:
    """Drop superseded/invalidated (and optionally disputed) for agent context."""
    drop = drop_statuses or {
        MemoryStatus.SUPERSEDED.value,
        MemoryStatus.INVALIDATED.value,
    }
    blocked = blocked or {}
    kept: list[RecalledMemory] = []
    for m in results:
        key = m.id or _stable_key(m)
        if key in blocked and blocked[key] in drop:
            continue
        status = str((m.metadata or {}).get("status") or "").lower()
        if status in drop:
            continue
        # also honor blocked overlay without matching id
        text_key = _stable_key(m)
        if text_key in blocked and blocked[text_key] in drop:
            continue
        kept.append(m)
    return kept


def _stable_key(m: RecalledMemory) -> str:
    return f"text:{hash(m.text) & 0xFFFFFFFF:08x}"


