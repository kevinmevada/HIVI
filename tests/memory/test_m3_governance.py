"""M3 — provenance, policies, corrections, bank isolation."""
from __future__ import annotations

from pathlib import Path

import pytest

from hivi.memory.manager import MemoryManager
from hivi.memory.models import MemoryStatus, MemoryType, RetainItem
from hivi.memory.policies import ActorRole, PolicyDenied
from hivi.memory.providers.memory_store import InMemoryProvider


@pytest.fixture()
def mm(tmp_path: Path) -> MemoryManager:
    db = tmp_path / "hivi.sqlite"
    return MemoryManager(provider=InMemoryProvider(), db_path=db)


def test_worker_cannot_retain_verified(mm: MemoryManager):
    with pytest.raises(PolicyDenied):
        mm.retain(
            project_id="001",
            items=[
                RetainItem(
                    content="secret",
                    memory_type=MemoryType.DECISION,
                    status=MemoryStatus.VERIFIED,
                )
            ],
            role=ActorRole.WORKER,
        )


def test_bank_isolation(mm: MemoryManager):
    mm.retain(
        project_id="001",
        items=[RetainItem(content="Project A uses PySide6", status=MemoryStatus.VERIFIED)],
        role=ActorRole.SYSTEM,
    )
    mm.retain(
        project_id="002",
        items=[RetainItem(content="Project B uses web UI", status=MemoryStatus.VERIFIED)],
        role=ActorRole.SYSTEM,
    )
    a = mm.recall(project_id="001", query="UI toolkit")
    b = mm.recall(project_id="002", query="UI toolkit")
    assert any("PySide6" in r.text for r in a.results)
    assert not any("PySide6" in r.text for r in b.results)
    assert any("web UI" in r.text for r in b.results)


def test_actor_project_guard(tmp_path: Path):
    mm = MemoryManager(
        provider=InMemoryProvider(),
        db_path=tmp_path / "t.sqlite",
        actor_project_id="001",
    )
    try:
        with pytest.raises(PolicyDenied):
            mm.recall(project_id="002", query="leak")
    finally:
        mm.close()


def test_supersede_hides_old_decision(mm: MemoryManager):
    ret = mm.retain(
        project_id="001",
        items=[
            RetainItem(
                content="UI toolkit is PySide6",
                memory_type=MemoryType.DECISION,
                status=MemoryStatus.VERIFIED,
                source="dec1",
            )
        ],
        role=ActorRole.PROJECT_OWNER,
    )
    assert ret.success
    # Capture memory id from recall
    old = mm.recall(project_id="001", query="UI toolkit", apply_governance=False)
    assert old.results
    mid = old.results[0].id
    assert mid
    mm.correct_memory(
        project_id="001",
        target_memory_key=mid,
        action="supersede",
        role=ActorRole.PROJECT_OWNER,
        reason="owner switched to web",
        new_content="UI toolkit is web-based",
    )
    cur = mm.recall(project_id="001", query="UI toolkit")
    texts = " ".join(r.text for r in cur.results)
    assert "web-based" in texts
    # old id should be filtered
    assert not any(r.id == mid for r in cur.results)


def test_provenance_on_record_event(mm: MemoryManager):
    evt = mm.record_event(
        project_id="001",
        event_type="architecture_decision",
        content="Use SQLite for app ledger",
        role=ActorRole.PROJECT_DIRECTOR,
        evidence_path="docs/arch.md",
        sync_retain=True,
        enqueue=False,
    )
    assert evt.id
    assert mm.ledger.get(evt.id) is not None
    rec = mm.recall(project_id="001", query="SQLite ledger")
    assert rec.results
    assert rec.results[0].metadata.get("event_id") == evt.id
    assert rec.results[0].metadata.get("evidence_path") == "docs/arch.md"
