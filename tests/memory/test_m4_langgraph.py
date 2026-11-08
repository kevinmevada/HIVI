"""M4 — LangGraph memory nodes, queue, checkpoint resume."""
from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from hivi.memory.manager import MemoryManager
from hivi.memory.providers.memory_store import InMemoryProvider
from hivi.orchestrator.graphs import build_memory_agent_graph, run_turn


def test_graph_loads_context_stores_and_reviews(tmp_path: Path):
    mm = MemoryManager(provider=InMemoryProvider(), db_path=tmp_path / "hivi.sqlite")
    mm.record_event(
        project_id="001",
        event_type="architecture_decision",
        content="Calculator UI uses PySide6",
        sync_retain=True,
        enqueue=False,
    )
    graph = build_memory_agent_graph(mm, checkpointer=MemorySaver())
    out = run_turn(
        graph,
        thread_id="t1",
        project_id="001",
        query="What UI toolkit?",
        event_type="test_failure",
        event_content="divide by zero when input empty",
        role="worker",
        agent_id="dev_01",
        task_id="calc_div",
    )
    assert "PySide6" in (out.get("context_text") or "")
    assert out.get("event_id")
    # failure retained via queue drain
    fail = mm.recall(project_id="001", query="divide by zero")
    assert fail.results


def test_checkpoint_resume_same_thread(tmp_path: Path):
    mm = MemoryManager(provider=InMemoryProvider(), db_path=tmp_path / "hivi.sqlite")
    saver = MemorySaver()
    graph = build_memory_agent_graph(mm, checkpointer=saver)
    run_turn(
        graph,
        thread_id="resume-1",
        project_id="001",
        query="start",
        event_type="architecture_decision",
        event_content="Use integer arithmetic for money",
        role="project_director",
    )
    # Second turn same thread — checkpoint exists; new invoke still works
    out2 = run_turn(
        graph,
        thread_id="resume-1",
        project_id="001",
        query="arithmetic policy",
        role="worker",
    )
    assert out2.get("context_text") is not None
    snap = graph.get_state({"configurable": {"thread_id": "resume-1"}})
    assert snap is not None


def test_queue_survives_reopen(tmp_path: Path):
    db = tmp_path / "hivi.sqlite"
    mm1 = MemoryManager(provider=InMemoryProvider(), db_path=db)
    mm1.record_event(
        project_id="001",
        event_type="requirement_change",
        content="Support scientific mode",
        enqueue=True,
        sync_retain=False,
    )
    pending = mm1.queue.pending()
    assert len(pending) == 1

    # New manager instance, same DB — pending still there
    mm2 = MemoryManager(provider=InMemoryProvider(), db_path=db)
    assert len(mm2.queue.pending()) == 1
    stats = mm2.drain_retention_queue()
    assert stats["ok"] == 1
    assert mm2.queue.pending() == []
