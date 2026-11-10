"""Typed memory records for the HIVI memory gateway (provider-agnostic)."""
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    DECISION = "decision"
    FAILURE = "failure"
    VERIFICATION = "verification"
    CONSTRAINT = "constraint"
    OPEN_QUESTION = "open_question"


class MemoryStatus(StrEnum):
    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    VERIFIED = "verified"
    SUPERSEDED = "superseded"
    DISPUTED = "disputed"
    INVALIDATED = "invalidated"


class RecallBudget(StrEnum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class MemoryScope(BaseModel):
    """Logical HIVI identity mapped to a Hindsight bank + tags."""

    project_id: str = Field(..., description="e.g. 001")
    department: str | None = None
    agent_id: str | None = None
    task_id: str | None = None

    def bank_id(self) -> str:
        from .scopes import bank_id_for_project

        return bank_id_for_project(self.project_id)

    def default_tags(self) -> list[str]:
        tags = [f"project:{self.project_id}"]
        if self.department:
            tags.append(f"department:{self.department}")
        if self.agent_id:
            tags.append(f"agent:{self.agent_id}")
        if self.task_id:
            tags.append(f"task:{self.task_id}")
        return tags


class RetainItem(BaseModel):
    content: str
    context: str | None = None
    memory_type: MemoryType | None = None
    status: MemoryStatus = MemoryStatus.UNVERIFIED
    source: str | None = Field(
        default=None, description="Evidence id / artifact path / event id"
    )
    metadata: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class RetainRequest(BaseModel):
    scope: MemoryScope
    items: list[RetainItem]
    wait: bool = Field(
        default=False,
        description="If True, block until Hindsight finishes extraction (slow).",
    )


class RetainResult(BaseModel):
    success: bool
    bank_id: str
    items_count: int
    async_: bool = Field(default=True, alias="async")
    raw: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class RecallRequest(BaseModel):
    scope: MemoryScope
    query: str
    budget: RecallBudget = RecallBudget.MID
    max_tokens: int = 1024


class RecalledMemory(BaseModel):
    id: str | None = None
    text: str
    type: str | None = None
    context: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    score: float | None = None


class RecallResult(BaseModel):
    bank_id: str
    query: str
    results: list[RecalledMemory]
    raw: dict[str, Any] = Field(default_factory=dict)


class ReflectRequest(BaseModel):
    scope: MemoryScope
    query: str


class ReflectResult(BaseModel):
    bank_id: str
    query: str
    answer: str
    raw: dict[str, Any] = Field(default_factory=dict)


class CurateOperation(StrEnum):
    CREATE_BANK = "create_bank"
    LIST_BANKS = "list_banks"


class CurateRequest(BaseModel):
    operation: CurateOperation
    scope: MemoryScope | None = None


class CurateResult(BaseModel):
    success: bool
    detail: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    ok: bool
    provider: str
    detail: str = ""
    base_url: str | None = None
