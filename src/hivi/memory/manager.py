"""HIVI Memory Gateway — single entry point for agents and workflows."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from hivi.storage.ledger import EventLedger

from .context_builder import BuiltContext, ContextBudget, build_context
from .curation import CorrectionStore
from .event_selector import select_event
from .interfaces import MemoryProvider
from .models import (
    CurateOperation,
    CurateRequest,
    CurateResult,
    HealthStatus,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    RecallRequest,
    RecallResult,
    ReflectRequest,
    ReflectResult,
    RetainItem,
    RetainRequest,
    RetainResult,
)
from .policies import ActorRole, PolicyDenied, assert_can_retain, assert_project_access
from .provenance import Provenance, ProvenanceIndex
from .providers.hindsight import HindsightProvider
from .retention_queue import RetentionQueue


class MemoryManager:
    """Coordinates memory ops through a pluggable provider (default: Hindsight)."""

    def __init__(
        self,
        provider: MemoryProvider | None = None,
        *,
        db_path: Path | str | None = None,
        actor_project_id: str | None = None,
    ) -> None:
        self._provider = provider or default_hindsight_provider()
        self._db_path = db_path
        self._ledger = EventLedger(db_path)
        self._queue = RetentionQueue(db_path)
        self._corrections = CorrectionStore(db_path)
        self._provenance = ProvenanceIndex()
        # When set, enforces bank isolation for this manager instance
        self.actor_project_id = actor_project_id

    @property
    def provider_name(self) -> str:
        return getattr(self._provider, "name", type(self._provider).__name__)

    @property
    def ledger(self) -> EventLedger:
        return self._ledger

    @property
    def queue(self) -> RetentionQueue:
        return self._queue

    @property
    def corrections(self) -> CorrectionStore:
        return self._corrections

    def health(self) -> HealthStatus:
        return self._provider.health_check()

    def _guard_project(self, project_id: str, role: ActorRole | str = ActorRole.SYSTEM) -> None:
        if self.actor_project_id is not None:
            assert_project_access(role, self.actor_project_id, project_id)

    def ensure_project_bank(self, project_id: str) -> CurateResult:
        self._guard_project(project_id)
        return self._provider.curate(
            CurateRequest(
                operation=CurateOperation.CREATE_BANK,
                scope=MemoryScope(project_id=project_id),
            )
        )

    def retain(
        self,
        *,
        project_id: str,
        items: list[RetainItem],
        department: str | None = None,
        agent_id: str | None = None,
        task_id: str | None = None,
        wait: bool = False,
        role: ActorRole | str = ActorRole.SYSTEM,
    ) -> RetainResult:
        self._guard_project(project_id, role)
        for it in items:
            assert_can_retain(role, memory_type=it.memory_type, status=it.status)
        return self._provider.retain(
            RetainRequest(
                scope=MemoryScope(
                    project_id=project_id,
                    department=department,
                    agent_id=agent_id,
                    task_id=task_id,
                ),
                items=items,
                wait=wait,
            )
        )

    def recall(
        self,
        *,
        project_id: str,
        query: str,
        department: str | None = None,
        agent_id: str | None = None,
        task_id: str | None = None,
        max_tokens: int = 1024,
        role: ActorRole | str = ActorRole.SYSTEM,
        apply_governance: bool = True,
    ) -> RecallResult:
        self._guard_project(project_id, role)
        result = self._provider.recall(
            RecallRequest(
                scope=MemoryScope(
                    project_id=project_id,
                    department=department,
                    agent_id=agent_id,
                    task_id=task_id,
                ),
                query=query,
                max_tokens=max_tokens,
            )
        )
        if apply_governance:
            from .curation import filter_current_memories

            blocked = self._corrections.blocked_keys(project_id)
            result.results = filter_current_memories(result.results, blocked=blocked)
        return result

    def reflect(
        self,
        *,
        project_id: str,
        query: str,
        department: str | None = None,
        agent_id: str | None = None,
        task_id: str | None = None,
        role: ActorRole | str = ActorRole.SYSTEM,
    ) -> ReflectResult:
        self._guard_project(project_id, role)
        return self._provider.reflect(
            ReflectRequest(
                scope=MemoryScope(
                    project_id=project_id,
                    department=department,
                    agent_id=agent_id,
                    task_id=task_id,
                ),
                query=query,
            )
        )

    def build_context(
        self,
        *,
        project_id: str,
        query: str,
        budget: ContextBudget | None = None,
        role: ActorRole | str = ActorRole.SYSTEM,
    ) -> BuiltContext:
        recall = self.recall(project_id=project_id, query=query, role=role)
        return build_context(recall, budget=budget, corrections=self._corrections)

    def record_event(
        self,
        *,
        project_id: str,
        event_type: str,
        content: str,
        role: ActorRole | str = ActorRole.SYSTEM,
        agent_id: str | None = None,
        task_id: str | None = None,
        evidence_path: str | None = None,
        enqueue: bool = True,
        sync_retain: bool = False,
        wait: bool = False,
    ):
        """Append to ledger; optionally queue or sync-retain based on event selector."""
        self._guard_project(project_id, role)
        evt = self._ledger.append(
            project_id=project_id,
            event_type=event_type,
            content=content,
            actor_role=str(role),
            agent_id=agent_id,
            task_id=task_id,
            evidence_path=evidence_path,
        )
        sel = select_event(event_type)
        if not sel.retain:
            return evt

        status = (
            MemoryStatus.VERIFIED
            if str(role) in {ActorRole.PROJECT_OWNER.value, ActorRole.SYSTEM.value}
            else MemoryStatus.UNVERIFIED
        )
        prov = Provenance(
            event_id=evt.id,
            evidence_path=evidence_path,
            source=evt.id,
            actor_role=str(role),
            agent_id=agent_id,
        )
        self._provenance.register(evt.id, prov)
        item = RetainItem(
            content=content,
            memory_type=sel.memory_type,
            status=status,
            source=evt.id,
            metadata=prov.to_metadata(),
        )
        if sync_retain:
            self._retain_with_retry(
                project_id=project_id,
                items=[item],
                agent_id=agent_id,
                task_id=task_id,
                wait=wait,
                role=role,
            )
        elif enqueue:
            self._queue.enqueue(
                event_id=evt.id,
                project_id=project_id,
                content=content,
                memory_type=sel.memory_type,
                status=status,
                source=evt.id,
                metadata=prov.to_metadata(),
                agent_id=agent_id,
                task_id=task_id,
            )
        return evt

    def drain_retention_queue(self, *, limit: int = 50) -> dict[str, int]:
        return self._queue.drain(self._retain_with_retry, limit=limit)

    def _retain_with_retry(
        self,
        *,
        project_id: str,
        items: list[RetainItem],
        department: str | None = None,
        agent_id: str | None = None,
        task_id: str | None = None,
        wait: bool = False,
        role: ActorRole | str = ActorRole.SYSTEM,
        attempts: int = 3,
    ) -> RetainResult:
        import time

        last: Exception | None = None
        for i in range(attempts):
            try:
                return self.retain(
                    project_id=project_id,
                    items=items,
                    department=department,
                    agent_id=agent_id,
                    task_id=task_id,
                    wait=wait,
                    role=role,
                )
            except Exception as e:
                last = e
                if i + 1 >= attempts:
                    break
                time.sleep(2.0 * (i + 1))
        assert last is not None
        raise last

    def close(self) -> None:
        """Release SQLite handles (needed on Windows before deleting db files)."""
        try:
            self._ledger.close()
        except Exception:
            pass
        for holder in (self._queue, self._corrections):
            conn = getattr(holder, "_conn", None)
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def correct_memory(
        self,
        *,
        project_id: str,
        target_memory_key: str,
        action: str,
        role: ActorRole | str,
        reason: str | None = None,
        new_content: str | None = None,
        sync_retain_replacement: bool = True,
    ):
        self._guard_project(project_id, role)
        rec = self._corrections.apply(
            project_id=project_id,
            target_memory_key=target_memory_key,
            action=action,
            actor_role=role,
            reason=reason,
            new_content=new_content,
        )
        if action == "supersede" and new_content and sync_retain_replacement:
            self.record_event(
                project_id=project_id,
                event_type="requirement_change",
                content=new_content,
                role=role,
                sync_retain=True,
                enqueue=False,
                wait=True,
            )
        return rec


def default_hindsight_provider() -> HindsightProvider:
    base = os.environ.get("HIVI_HINDSIGHT_URL", "http://127.0.0.1:8888")
    token = os.environ.get("HIVI_HINDSIGHT_TOKEN") or None
    return HindsightProvider(base_url=base, token=token)


@lru_cache(maxsize=1)
def get_memory_manager() -> MemoryManager:
    """Process-wide gateway (swap provider in tests via MemoryManager(provider=...))."""
    return MemoryManager()
