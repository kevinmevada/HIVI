"""Link memories to original evidence (ledger event / filesystem artifact)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Provenance(BaseModel):
    event_id: str | None = None
    evidence_path: str | None = None
    source: str | None = None
    actor_role: str | None = None
    agent_id: str | None = None
    notes: str | None = None

    def to_metadata(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.event_id:
            out["event_id"] = self.event_id
        if self.evidence_path:
            out["evidence_path"] = self.evidence_path
        if self.source:
            out["source"] = self.source
        if self.actor_role:
            out["actor_role"] = self.actor_role
        if self.agent_id:
            out["agent_id"] = self.agent_id
        if self.notes:
            out["provenance_notes"] = self.notes
        return out


class ProvenanceIndex(BaseModel):
    """In-process index of memory_key → provenance (also mirrored in retain metadata)."""

    records: dict[str, Provenance] = Field(default_factory=dict)

    def register(self, memory_key: str, provenance: Provenance) -> None:
        self.records[memory_key] = provenance

    def get(self, memory_key: str) -> Provenance | None:
        return self.records.get(memory_key)

    def from_metadata(self, memory_key: str, metadata: dict[str, Any]) -> Provenance:
        p = Provenance(
            event_id=metadata.get("event_id"),
            evidence_path=metadata.get("evidence_path"),
            source=metadata.get("source"),
            actor_role=metadata.get("actor_role"),
            agent_id=metadata.get("agent_id"),
            notes=metadata.get("provenance_notes"),
        )
        self.register(memory_key, p)
        return p
