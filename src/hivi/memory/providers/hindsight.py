"""Hindsight REST adapter — uses httpx only (no hindsight SDK in HIVI core)."""
from __future__ import annotations

from typing import Any

import httpx

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
    RetainItem,
    RetainRequest,
    RetainResult,
)


class HindsightError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class HindsightProvider:
    """Talks to a running hindsight-api (default http://127.0.0.1:8888)."""

    name = "hindsight"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8888",
        tenant: str = "default",
        timeout_s: float = 120.0,
        token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.tenant = tenant
        self.timeout_s = timeout_s
        self.token = token

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any] | list[Any] | None:
        timeout = timeout_s if timeout_s is not None else self.timeout_s
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.request(
                    method,
                    self._url(path),
                    headers=self._headers(),
                    json=json_body,
                )
        except httpx.HTTPError as e:
            raise HindsightError(f"Hindsight request failed: {e}") from e
        if r.status_code >= 400:
            raise HindsightError(
                f"Hindsight HTTP {r.status_code} for {path}",
                status_code=r.status_code,
                body=r.text[:500],
            )
        if not r.content:
            return None
        try:
            return r.json()
        except Exception:
            return {"raw": r.text}

    def health_check(self) -> HealthStatus:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(self._url("/health"), headers=self._headers())
            ok = r.status_code == 200
            return HealthStatus(
                ok=ok,
                provider=self.name,
                detail=f"HTTP {r.status_code}",
                base_url=self.base_url,
            )
        except Exception as e:
            return HealthStatus(
                ok=False,
                provider=self.name,
                detail=str(e),
                base_url=self.base_url,
            )

    def ensure_bank(self, bank_id: str) -> dict[str, Any]:
        data = self._request(
            "PUT",
            f"/v1/{self.tenant}/banks/{bank_id}",
            json_body={},
            timeout_s=30.0,
        )
        return data if isinstance(data, dict) else {"bank_id": bank_id}

    def retain(self, request: RetainRequest) -> RetainResult:
        bank_id = request.scope.bank_id()
        self.ensure_bank(bank_id)
        items = [self._to_hindsight_item(it, request.scope.default_tags()) for it in request.items]
        # async=true is Hindsight default; wait=True → async false
        body = {"items": items, "async": not request.wait}
        timeout = 600.0 if request.wait else 60.0
        data = self._request(
            "POST",
            f"/v1/{self.tenant}/banks/{bank_id}/memories",
            json_body=body,
            timeout_s=timeout,
        )
        raw = data if isinstance(data, dict) else {}
        return RetainResult(
            success=bool(raw.get("success", True)),
            bank_id=str(raw.get("bank_id") or bank_id),
            items_count=int(raw.get("items_count") or len(items)),
            async_=bool(raw.get("async", not request.wait)),
            raw=raw,
        )

    def recall(self, request: RecallRequest) -> RecallResult:
        bank_id = request.scope.bank_id()
        body = {
            "query": request.query,
            "budget": request.budget.value,
            "max_tokens": request.max_tokens,
        }
        data = self._request(
            "POST",
            f"/v1/{self.tenant}/banks/{bank_id}/memories/recall",
            json_body=body,
            timeout_s=120.0,
        )
        raw = data if isinstance(data, dict) else {}
        results: list[RecalledMemory] = []
        for row in raw.get("results") or []:
            if not isinstance(row, dict):
                continue
            scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
            results.append(
                RecalledMemory(
                    id=row.get("id"),
                    text=str(row.get("text") or ""),
                    type=row.get("type"),
                    context=row.get("context"),
                    metadata=dict(row.get("metadata") or {}),
                    tags=list(row.get("tags") or []),
                    score=(
                        float(scores["final"])
                        if scores.get("final") is not None
                        else None
                    ),
                )
            )
        return RecallResult(
            bank_id=bank_id, query=request.query, results=results, raw=raw
        )

    def reflect(self, request: ReflectRequest) -> ReflectResult:
        bank_id = request.scope.bank_id()
        data = self._request(
            "POST",
            f"/v1/{self.tenant}/banks/{bank_id}/reflect",
            json_body={"query": request.query},
            timeout_s=600.0,
        )
        raw = data if isinstance(data, dict) else {}
        answer = (
            raw.get("answer")
            or raw.get("text")
            or raw.get("reflection")
            or raw.get("content")
            or ""
        )
        if not answer and isinstance(raw.get("result"), str):
            answer = raw["result"]
        return ReflectResult(
            bank_id=bank_id,
            query=request.query,
            answer=str(answer),
            raw=raw,
        )

    def curate(self, request: CurateRequest) -> CurateResult:
        if request.operation == CurateOperation.CREATE_BANK:
            if request.scope is None:
                return CurateResult(success=False, detail="scope required")
            bank_id = request.scope.bank_id()
            raw = self.ensure_bank(bank_id)
            return CurateResult(
                success=True, detail=f"bank ensured: {bank_id}", raw=raw
            )
        if request.operation == CurateOperation.LIST_BANKS:
            data = self._request(
                "GET", f"/v1/{self.tenant}/banks", timeout_s=30.0
            )
            raw = data if isinstance(data, dict) else {"banks": data}
            return CurateResult(success=True, detail="ok", raw=raw)
        return CurateResult(success=False, detail=f"unsupported: {request.operation}")

    @staticmethod
    def _to_hindsight_item(item: RetainItem, scope_tags: list[str]) -> dict[str, Any]:
        meta = dict(item.metadata)
        if item.memory_type is not None:
            meta.setdefault("type", item.memory_type.value)
        meta.setdefault("status", item.status.value)
        if item.source:
            meta.setdefault("source", item.source)
        # Prefer structured metadata; tags also carried for future filter support
        tags = list(dict.fromkeys([*scope_tags, *item.tags]))
        if item.memory_type is not None:
            tags.append(f"type:{item.memory_type.value}")
        tags.append(f"status:{item.status.value}")
        if item.source:
            tags.append(f"source:{item.source}")
        out: dict[str, Any] = {"content": item.content, "metadata": meta}
        if item.context:
            out["context"] = item.context
        # Hindsight retain items accept metadata; tags may be ignored by API — keep in content footer if needed
        if tags:
            out["metadata"]["hivi_tags"] = ",".join(tags)
        return out
