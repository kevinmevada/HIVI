"""Live integration against Hindsight on :8888 (skipped if unhealthy)."""
from __future__ import annotations

import time
import uuid

import pytest

from hivi.memory.manager import MemoryManager
from hivi.memory.models import MemoryStatus, MemoryType, RetainItem
from hivi.memory.providers.hindsight import HindsightProvider


@pytest.fixture(scope="module")
def provider() -> HindsightProvider:
    p = HindsightProvider(base_url="http://127.0.0.1:8888")
    health = p.health_check()
    if not health.ok:
        pytest.skip(f"Hindsight not healthy: {health.detail}")
    return p


@pytest.fixture(scope="module")
def mm(provider: HindsightProvider) -> MemoryManager:
    return MemoryManager(provider=provider)


def test_health(provider: HindsightProvider):
    h = provider.health_check()
    assert h.ok
    assert h.provider == "hindsight"


def test_ensure_bank(mm: MemoryManager):
    out = mm.ensure_project_bank("001")
    assert out.success
    assert "hivi_project_001" in out.detail


def test_retain_and_recall(mm: MemoryManager):
    marker = f"M2_LIVE_{uuid.uuid4().hex[:8]}"
    content = (
        f"{marker}: HIVI desktop UI uses PySide6. "
        "This is an M2 adapter live acceptance fact."
    )
    ret = mm.retain(
        project_id="001",
        items=[
            RetainItem(
                content=content,
                memory_type=MemoryType.DECISION,
                status=MemoryStatus.VERIFIED,
                source="m2_live_test",
            )
        ],
        department="architecture",
        wait=True,
    )
    assert ret.success
    assert ret.bank_id == "hivi_project_001"
    assert ret.items_count >= 1

    # Give index a moment even after sync retain
    time.sleep(1.0)

    rec = mm.recall(
        project_id="001",
        query=f"What UI toolkit does HIVI use? Look for {marker}",
        max_tokens=512,
    )
    blob = " ".join(r.text for r in rec.results).lower()
    assert rec.results, f"empty recall raw={rec.raw}"
    assert "pyside6" in blob or marker.lower() in blob
