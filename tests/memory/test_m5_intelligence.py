"""M5 — selective retention, budgets, dedupe, reflection."""
from __future__ import annotations

from pathlib import Path

from hivi.memory.context_builder import ContextBudget, dedupe_memories
from hivi.memory.event_selector import EventKind, RetainPriority, select_event, should_retain
from hivi.memory.manager import MemoryManager
from hivi.memory.models import RecalledMemory
from hivi.memory.providers.memory_store import InMemoryProvider


def test_event_selector_discards_chat():
    assert not should_retain(EventKind.AGENT_CHAT)
    assert select_event(EventKind.ARCHITECTURE_DECISION).priority == RetainPriority.IMMEDIATE


def test_context_budget_and_dedupe(tmp_path: Path):
    mm = MemoryManager(provider=InMemoryProvider(), db_path=tmp_path / "hivi.sqlite")
    for i in range(20):
        mm.record_event(
            project_id="001",
            event_type="architecture_decision",
            content=f"Decision number {i}: use module m{i}",
            sync_retain=True,
            enqueue=False,
        )
    # duplicate
    mm.record_event(
        project_id="001",
        event_type="architecture_decision",
        content="Decision number 0: use module m0",
        sync_retain=True,
        enqueue=False,
    )
    built = mm.build_context(
        project_id="001",
        query="module decisions",
        budget=ContextBudget(max_chars=500, max_items=5),
    )
    assert len(built.items) <= 5
    assert len(built.text) <= 600  # some overhead for labels
    assert built.truncated or len(built.items) == 5


def test_dedupe_helper():
    rows = [
        RecalledMemory(text="Same fact", metadata={"status": "verified"}, score=0.9),
        RecalledMemory(text="Same   fact", metadata={"status": "verified"}, score=0.8),
        RecalledMemory(text="Other", metadata={"status": "verified"}, score=0.7),
    ]
    assert len(dedupe_memories(rows)) == 2


def test_reflect_uses_provider(tmp_path: Path):
    mm = MemoryManager(provider=InMemoryProvider(), db_path=tmp_path / "hivi.sqlite")
    mm.record_event(
        project_id="001",
        event_type="test_failure",
        content="Parser crashed on empty input",
        sync_retain=True,
        enqueue=False,
    )
    ref = mm.reflect(project_id="001", query="failures")
    assert "empty input" in ref.answer.lower() or "Parser" in ref.answer


def test_build_context_drops_superseded(tmp_path: Path):
    mm = MemoryManager(provider=InMemoryProvider(), db_path=tmp_path / "hivi.sqlite")
    mm.record_event(
        project_id="001",
        event_type="architecture_decision",
        content="Display is CLI only",
        sync_retain=True,
        enqueue=False,
    )
    old = mm.recall(project_id="001", query="Display", apply_governance=False).results[0]
    mm.correct_memory(
        project_id="001",
        target_memory_key=old.id or "x",
        action="supersede",
        role="project_owner",
        new_content="Display is GUI with PySide6",
        sync_retain_replacement=True,
    )
    built = mm.build_context(project_id="001", query="Display")
    assert "PySide6" in built.text
    assert "CLI only" not in built.text
