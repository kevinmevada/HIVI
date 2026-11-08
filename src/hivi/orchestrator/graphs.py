"""Minimal HIVI memory-aware LangGraph workflow with checkpointing."""
from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from hivi.memory.manager import MemoryManager

from .memory_nodes import MemoryGraphState, make_memory_nodes

# Project-room graph (load → execute → store → verify) lives in project_room.py


def build_memory_agent_graph(
    mm: MemoryManager,
    *,
    checkpointer: Any | None = None,
):
    """
    Graph: load context → (optional) store event → optional review → end.

    Checkpoints are LangGraph's job; Hindsight is not used as a checkpoint store.
    For the full project-room E2E graph, use ProjectRoom.
    """
    nodes = make_memory_nodes(mm)
    g = StateGraph(MemoryGraphState)
    g.add_node("load_project_context", nodes["load_project_context"])
    g.add_node("store_execution_memory", nodes["store_execution_memory"])
    g.add_node("review_project_memory", nodes["review_project_memory"])

    g.add_edge(START, "load_project_context")
    g.add_edge("load_project_context", "store_execution_memory")
    g.add_edge("store_execution_memory", "review_project_memory")
    g.add_edge("review_project_memory", END)

    saver = checkpointer if checkpointer is not None else MemorySaver()
    return g.compile(checkpointer=saver)


def run_turn(
    graph,
    *,
    thread_id: str,
    project_id: str,
    query: str,
    event_type: str | None = None,
    event_content: str | None = None,
    role: str = "system",
    agent_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    payload: MemoryGraphState = {
        "project_id": project_id,
        "query": query,
        "role": role,
        "agent_id": agent_id,
        "task_id": task_id,
    }
    if event_type:
        payload["last_event_type"] = event_type
    if event_content:
        payload["last_event_content"] = event_content
    return graph.invoke(payload, config)
