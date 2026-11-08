"""Memory provider contract — agents never depend on a specific backend SDK."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    CurateRequest,
    CurateResult,
    HealthStatus,
    RecallRequest,
    RecallResult,
    ReflectRequest,
    ReflectResult,
    RetainRequest,
    RetainResult,
)


@runtime_checkable
class MemoryProvider(Protocol):
    """Backend adapter (Hindsight today, Graphiti later)."""

    name: str

    def health_check(self) -> HealthStatus: ...

    def retain(self, request: RetainRequest) -> RetainResult: ...

    def recall(self, request: RecallRequest) -> RecallResult: ...

    def reflect(self, request: ReflectRequest) -> ReflectResult: ...

    def curate(self, request: CurateRequest) -> CurateResult: ...
