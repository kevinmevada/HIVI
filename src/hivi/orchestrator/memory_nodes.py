"""Reusable LangGraph node functions for HIVI memory."""
from __future__ import annotations

from typing import Any, TypedDict

from hivi.memory.context_builder import ContextBudget
from hivi.memory.manager import MemoryManager
from hivi.memory.policies import ActorRole


class MemoryGraphState(TypedDict, total=False):
    project_id: str
    query: str
    task_id: str | None
    agent_id: str | None
    role: str
    context_text: str
    context_truncated: bool
    last_event_type: str
    last_event_content: str
    evidence_path: str | None
    event_id: str | None
    reflect_answer: str
    artifact_path: str | None
    deliverable_ok: bool
    verify_detail: str | None
    error: str | None


def make_memory_nodes(mm: MemoryManager):
    """Bind a MemoryManager into LangGraph-compatible node callables."""

    def load_project_context(state: MemoryGraphState) -> dict[str, Any]:
        project_id = state["project_id"]
        query = state.get("query") or "project status and decisions"
        role = state.get("role") or ActorRole.SYSTEM.value
        budget = ContextBudget(max_chars=3500, max_items=10)
        built = mm.build_context(
            project_id=project_id, query=query, budget=budget, role=role
        )
        return {
            "context_text": built.text,
            "context_truncated": built.truncated,
            "error": None,
        }

    def store_execution_memory(state: MemoryGraphState) -> dict[str, Any]:
        event_type = state.get("last_event_type") or "agent_chat"
        content = state.get("last_event_content") or ""
        if not content:
            return {"event_id": None}
        role = state.get("role") or ActorRole.WORKER.value
        evt = mm.record_event(
            project_id=state["project_id"],
            event_type=event_type,
            content=content,
            role=role,
            agent_id=state.get("agent_id"),
            task_id=state.get("task_id"),
            evidence_path=state.get("evidence_path"),
            enqueue=True,
            sync_retain=False,
        )
        # Drain immediately for small graphs; production can run a worker
        mm.drain_retention_queue(limit=20)
        return {"event_id": evt.id}

    def review_project_memory(state: MemoryGraphState) -> dict[str, Any]:
        project_id = state["project_id"]
        query = state.get("query") or "Summarize current project constraints and failures"
        role = state.get("role") or ActorRole.PROJECT_DIRECTOR.value
        ref = mm.reflect(project_id=project_id, query=query, role=role)
        return {"reflect_answer": ref.answer}

    return {
        "load_project_context": load_project_context,
        "store_execution_memory": store_execution_memory,
        "review_project_memory": review_project_memory,
    }
