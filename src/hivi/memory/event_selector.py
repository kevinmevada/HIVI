"""Decide which execution events are worth retaining."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .models import MemoryType


class EventKind(StrEnum):
    REQUIREMENT_CHANGE = "requirement_change"
    ARCHITECTURE_DECISION = "architecture_decision"
    TASK_COMPLETED = "task_completed"
    TEST_FAILURE = "test_failure"
    USER_APPROVAL = "user_approval"
    USER_REJECTION = "user_rejection"
    FINAL_DELIVERABLE = "final_deliverable"
    AGENT_CHAT = "agent_chat"
    TOOL_OUTPUT = "tool_output"
    CORRECTION = "correction"


class RetainPriority(StrEnum):
    IMMEDIATE = "immediate"
    AFTER_VERIFY = "after_verify"
    DISCARD = "discard"
    DEDUPE = "dedupe"


@dataclass
class Selection:
    retain: bool
    priority: RetainPriority
    memory_type: MemoryType | None
    reason: str


_POLICY: dict[EventKind, Selection] = {
    EventKind.REQUIREMENT_CHANGE: Selection(
        True, RetainPriority.IMMEDIATE, MemoryType.CONSTRAINT, "requirement"
    ),
    EventKind.ARCHITECTURE_DECISION: Selection(
        True, RetainPriority.IMMEDIATE, MemoryType.DECISION, "architecture"
    ),
    EventKind.TASK_COMPLETED: Selection(
        True, RetainPriority.AFTER_VERIFY, MemoryType.EPISODIC, "task done"
    ),
    EventKind.TEST_FAILURE: Selection(
        True, RetainPriority.AFTER_VERIFY, MemoryType.FAILURE, "test failure"
    ),
    EventKind.USER_APPROVAL: Selection(
        True, RetainPriority.IMMEDIATE, MemoryType.VERIFICATION, "approval"
    ),
    EventKind.USER_REJECTION: Selection(
        True, RetainPriority.IMMEDIATE, MemoryType.CONSTRAINT, "rejection"
    ),
    EventKind.FINAL_DELIVERABLE: Selection(
        True, RetainPriority.IMMEDIATE, MemoryType.SEMANTIC, "deliverable"
    ),
    EventKind.CORRECTION: Selection(
        True, RetainPriority.IMMEDIATE, MemoryType.CONSTRAINT, "correction"
    ),
    EventKind.AGENT_CHAT: Selection(
        False, RetainPriority.DISCARD, None, "noise chat"
    ),
    EventKind.TOOL_OUTPUT: Selection(
        False, RetainPriority.DEDUPE, None, "tool noise"
    ),
}


def select_event(event_type: str | EventKind) -> Selection:
    try:
        kind = EventKind(event_type) if not isinstance(event_type, EventKind) else event_type
    except ValueError:
        return Selection(False, RetainPriority.DISCARD, None, f"unknown:{event_type}")
    return _POLICY[kind]


def should_retain(event_type: str | EventKind) -> bool:
    return select_event(event_type).retain
