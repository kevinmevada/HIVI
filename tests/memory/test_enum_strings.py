"""Regression: (str, Enum) members stringified as 'ActorRole.X' on Python 3.11+."""
from __future__ import annotations

from pathlib import Path

from hivi.memory.manager import MemoryManager
from hivi.memory.models import MemoryStatus, MemoryType
from hivi.memory.policies import ActorRole
from hivi.memory.providers.memory_store import InMemoryProvider


def test_enum_members_stringify_to_values():
    assert str(ActorRole.PROJECT_OWNER) == "project_owner"
    assert f"{ActorRole.SYSTEM}" == "system"
    assert str(MemoryStatus.VERIFIED) == MemoryStatus.VERIFIED.value
    assert str(MemoryType.DECISION) == MemoryType.DECISION.value


def test_owner_event_is_recorded_with_plain_role_and_verified(tmp_path: Path):
    provider = InMemoryProvider()
    mm = MemoryManager(provider=provider, db_path=tmp_path / "hivi.sqlite")
    evt = mm.record_event(
        project_id="001",
        event_type="architecture_decision",
        content="Calculator UI is web-based",
        role=ActorRole.PROJECT_OWNER,  # enum, as ProjectRoom passes it
        sync_retain=True,
        enqueue=False,
    )
    assert mm.ledger.get(evt.id).actor_role == "project_owner"
    stored = [m for bank in provider.banks.values() for m in bank]
    assert stored and stored[0]["metadata"]["status"] == MemoryStatus.VERIFIED.value
