"""Unit tests with a fake HTTP layer (no live Hindsight required)."""
from __future__ import annotations

from typing import Any

import pytest

from hivi.memory.manager import MemoryManager
from hivi.memory.models import (
    CurateOperation,
    CurateRequest,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    RecallRequest,
    ReflectRequest,
    RetainItem,
    RetainRequest,
)
from hivi.memory.providers.hindsight import HindsightProvider


class FakeHindsight(HindsightProvider):
    def __init__(self) -> None:
        super().__init__(base_url="http://fake")
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.banks: set[str] = set()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any] | list[Any] | None:
        self.calls.append((method, path, json_body))
        if method == "PUT" and "/banks/" in path:
            bank = path.rstrip("/").split("/")[-1]
            self.banks.add(bank)
            return {"bank_id": bank}
        if method == "POST" and path.endswith("/memories"):
            return {
                "success": True,
                "bank_id": "hivi_project_001",
                "items_count": len((json_body or {}).get("items") or []),
                "async": (json_body or {}).get("async", True),
            }
        if method == "POST" and path.endswith("/recall"):
            return {
                "results": [
                    {
                        "id": "m1",
                        "text": "Chose PySide6 for desktop UI",
                        "type": "world",
                        "metadata": {"type": "decision"},
                        "tags": [],
                        "scores": {"final": 0.9},
                    }
                ]
            }
        if method == "POST" and path.endswith("/reflect"):
            return {"answer": "The project uses PySide6."}
        if method == "GET" and path.endswith("/banks"):
            return {"banks": [{"bank_id": b} for b in sorted(self.banks)], "total": len(self.banks)}
        return {}


def test_manager_retain_recall_reflect():
    fake = FakeHindsight()
    mm = MemoryManager(provider=fake)

    ret = mm.retain(
        project_id="001",
        items=[
            RetainItem(
                content="Use PySide6 for the UI.",
                memory_type=MemoryType.DECISION,
                status=MemoryStatus.VERIFIED,
                source="m2_unit",
            )
        ],
        department="architecture",
        wait=False,
    )
    assert ret.success and ret.items_count == 1
    assert "hivi_project_001" in fake.banks

    rec = mm.recall(project_id="001", query="Which UI toolkit?")
    assert rec.results and "PySide6" in rec.results[0].text

    ref = mm.reflect(project_id="1", query="Summarize UI choice")
    assert "PySide6" in ref.answer


def test_curate_create_bank():
    fake = FakeHindsight()
    out = fake.curate(
        CurateRequest(
            operation=CurateOperation.CREATE_BANK,
            scope=MemoryScope(project_id="001"),
        )
    )
    assert out.success
    assert "hivi_project_001" in fake.banks


def test_provider_builds_metadata_tags():
    fake = FakeHindsight()
    req = RetainRequest(
        scope=MemoryScope(project_id="001", department="testing", agent_id="test_01"),
        items=[
            RetainItem(
                content="18 tests passed",
                memory_type=MemoryType.VERIFICATION,
                status=MemoryStatus.VERIFIED,
                source="report.json",
            )
        ],
        wait=False,
    )
    fake.retain(req)
    # last POST body
    post = [c for c in fake.calls if c[0] == "POST" and c[1].endswith("/memories")][-1]
    item = post[2]["items"][0]
    assert item["metadata"]["type"] == "verification"
    assert item["metadata"]["status"] == "verified"
    assert "project:001" in item["metadata"]["hivi_tags"]
