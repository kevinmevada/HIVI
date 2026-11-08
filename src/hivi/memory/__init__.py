"""HIVI memory gateway — persistent project knowledge (Hindsight-backed)."""

from .context_builder import BuiltContext, ContextBudget, build_context
from .event_selector import EventKind, select_event, should_retain
from .manager import MemoryManager, get_memory_manager
from .models import (
    HealthStatus,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    RecallBudget,
    RecallRequest,
    RecallResult,
    RecalledMemory,
    ReflectRequest,
    ReflectResult,
    RetainItem,
    RetainRequest,
    RetainResult,
)
from .policies import ActorRole, PolicyDenied
from .scopes import bank_id_for_project, normalize_project_id

__all__ = [
    "MemoryManager",
    "get_memory_manager",
    "HealthStatus",
    "MemoryScope",
    "MemoryStatus",
    "MemoryType",
    "RecallBudget",
    "RecallRequest",
    "RecallResult",
    "RecalledMemory",
    "ReflectRequest",
    "ReflectResult",
    "RetainItem",
    "RetainRequest",
    "RetainResult",
    "ActorRole",
    "PolicyDenied",
    "BuiltContext",
    "ContextBudget",
    "build_context",
    "EventKind",
    "select_event",
    "should_retain",
    "bank_id_for_project",
    "normalize_project_id",
]
