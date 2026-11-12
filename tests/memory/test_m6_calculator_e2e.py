"""M6 — Calculator project end-to-end memory acceptance (in-memory provider)."""
from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from hivi.memory.manager import MemoryManager
from hivi.memory.policies import ActorRole
from hivi.memory.providers.memory_store import InMemoryProvider
from hivi.orchestrator.graphs import build_memory_agent_graph, run_turn


def test_calculator_project_memory_lifecycle(tmp_path: Path):
    """
    Acceptance:
    - decisions retained
    - stop/restart (new manager, same DB + provider state via re-seed from queue/ledger)
    - requirement change supersedes old decision
    - failure recalled
    - agent handover sees current context
    - bank isolation vs other project
    """
    db = tmp_path / "calc.sqlite"
    provider = InMemoryProvider()
    mm = MemoryManager(provider=provider, db_path=db)

    # 1. Architecture decision
    mm.record_event(
        project_id="001",
        event_type="architecture_decision",
        content="Calculator desktop UI uses PySide6",
        role=ActorRole.PROJECT_DIRECTOR,
        evidence_path="projects/calculator/ADR-001.md",
        sync_retain=True,
        enqueue=False,
    )
    # 2. Failure
    mm.record_event(
        project_id="001",
        event_type="test_failure",
        content="Calculator failed: division by zero on empty input",
        role=ActorRole.WORKER,
        agent_id="tester_01",
        sync_retain=True,
        enqueue=False,
    )

    # 3. Simulate restart: new MemoryManager, same provider+db
    mm2 = MemoryManager(provider=provider, db_path=db)
    ctx = mm2.build_context(project_id="001", query="UI toolkit and known failures")
    assert "PySide6" in ctx.text
    assert "division by zero" in ctx.text.lower()
    assert mm2.ledger.list_for_project("001")

    # 4. Requirement change — supersede UI decision
    old = mm2.recall(project_id="001", query="PySide6", apply_governance=False).results
    pyside = next(r for r in old if "PySide6" in r.text)
    mm2.correct_memory(
        project_id="001",
        target_memory_key=pyside.id or "",
        action="supersede",
        role=ActorRole.PROJECT_OWNER,
        reason="Owner wants web UI",
        new_content="Calculator UI is web-based",
    )

    # 5. New agent handover
    graph = build_memory_agent_graph(mm2, checkpointer=MemorySaver())
    out = run_turn(
        graph,
        thread_id="calc-handover",
        project_id="001",
        query="What is the current UI requirement? Any past failures?",
        role=ActorRole.WORKER.value,
        agent_id="dev_02",
    )
    ctx = (out.get("context_text") or "").lower()
    text = ctx + " " + (out.get("reflect_answer") or "").lower()
    assert "web-based" in text or "web" in text
    # superseded original decision must not appear as current context
    assert "desktop ui uses pyside6" not in ctx
    assert "division by zero" in text or "empty input" in text

    # 6. Isolation — project 002 cannot see calculator secrets
    mm2.record_event(
        project_id="002",
        event_type="architecture_decision",
        content="Unrelated weather app uses Qt",
        sync_retain=True,
        enqueue=False,
    )
    other = mm2.recall(project_id="002", query="Calculator PySide6 division")
    blob = " ".join(r.text for r in other.results).lower()
    assert "division by zero" not in blob
    assert "web-based" not in blob


def test_queue_recovery_after_interrupt(tmp_path: Path):
    db = tmp_path / "q.sqlite"
    p = InMemoryProvider()
    mm = MemoryManager(provider=p, db_path=db)
    mm.record_event(
        project_id="001",
        event_type="final_deliverable",
        content="Shipped calculator v0.1",
        enqueue=True,
        sync_retain=False,
    )
    assert mm.queue.pending()
    # crash before drain — reopen
    mm_b = MemoryManager(provider=p, db_path=db)
    assert mm_b.queue.pending()
    assert mm_b.drain_retention_queue()["ok"] == 1
    rec = mm_b.recall(project_id="001", query="calculator v0.1")
    assert rec.results
