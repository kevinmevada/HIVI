"""In-process MemoryProvider for tests and offline governance validation."""
from __future__ import annotations

import itertools
from typing import Any

from hivi.memory.models import (
    CurateOperation,
    CurateRequest,
    CurateResult,
    HealthStatus,
    RecallRequest,
    RecallResult,
    RecalledMemory,
    ReflectRequest,
    ReflectResult,
    RetainRequest,
    RetainResult,
)


class InMemoryProvider:
    name = "memory_store"

    def __init__(self) -> None:
        self.banks: dict[str, list[dict[str, Any]]] = {}
        self._ids = itertools.count(1)

    def health_check(self) -> HealthStatus:
        return HealthStatus(ok=True, provider=self.name, detail="in-memory", base_url=None)

    def ensure_bank(self, bank_id: str) -> dict[str, Any]:
        self.banks.setdefault(bank_id, [])
        return {"bank_id": bank_id}

    def retain(self, request: RetainRequest) -> RetainResult:
        bank_id = request.scope.bank_id()
        self.ensure_bank(bank_id)
        for it in request.items:
            meta = dict(it.metadata)
            if it.memory_type is not None:
                meta.setdefault("type", it.memory_type.value)
            meta.setdefault("status", it.status.value)
            if it.source:
                meta.setdefault("source", it.source)
            self.banks[bank_id].append(
                {
                    "id": f"m{next(self._ids)}",
                    "text": it.content,
                    "type": meta.get("type"),
                    "metadata": meta,
                    "tags": list(request.scope.default_tags()) + list(it.tags),
                    "score": 1.0,
                }
            )
        return RetainResult(
            success=True,
            bank_id=bank_id,
            items_count=len(request.items),
            async_=not request.wait,
            raw={"ok": True},
        )

    def recall(self, request: RecallRequest) -> RecallResult:
        bank_id = request.scope.bank_id()
        rows = self.banks.get(bank_id, [])
        q = request.query.lower()
        hits = []
        for row in rows:
            text = str(row["text"])
            score = 1.0 if any(tok in text.lower() for tok in q.split() if len(tok) > 2) else 0.2
            hits.append(
                RecalledMemory(
                    id=row["id"],
                    text=text,
                    type=row.get("type"),
                    metadata=dict(row.get("metadata") or {}),
                    tags=list(row.get("tags") or []),
                    score=score,
                )
            )
        hits.sort(key=lambda m: m.score or 0, reverse=True)
        return RecallResult(bank_id=bank_id, query=request.query, results=hits, raw={})

    def reflect(self, request: ReflectRequest) -> ReflectResult:
        rec = self.recall(
            RecallRequest(scope=request.scope, query=request.query, max_tokens=512)
        )
        answer = " ".join(m.text for m in rec.results[:5]) or "No memories."
        return ReflectResult(
            bank_id=rec.bank_id, query=request.query, answer=answer, raw={}
        )

    def curate(self, request: CurateRequest) -> CurateResult:
        if request.operation == CurateOperation.CREATE_BANK:
            if request.scope is None:
                return CurateResult(success=False, detail="scope required")
            bank_id = request.scope.bank_id()
            self.ensure_bank(bank_id)
            return CurateResult(success=True, detail=f"bank ensured: {bank_id}", raw={})
        if request.operation == CurateOperation.LIST_BANKS:
            return CurateResult(
                success=True,
                detail="ok",
                raw={"banks": list(self.banks.keys())},
            )
        return CurateResult(success=False, detail=f"unsupported: {request.operation}")
