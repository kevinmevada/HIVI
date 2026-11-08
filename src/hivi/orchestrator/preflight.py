"""Live preflight for HIVI project-room E2E (inference + Hindsight)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from hivi.memory.manager import MemoryManager
from hivi.memory.providers.hindsight import HindsightProvider


@dataclass
class PreflightResult:
    ok: bool
    inference_url: str
    hindsight_url: str
    model_id: str | None = None
    hindsight_ok: bool = False
    inference_models_ok: bool = False
    inference_chat_ok: bool = False
    bank_ok: bool = False
    detail: str = ""
    errors: list[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise RuntimeError(
                "E2E preflight failed: " + "; ".join(self.errors) or self.detail
            )


def resolve_inference_url() -> str:
    return os.environ.get("HIVI_INFERENCE_URL", "http://127.0.0.1:8080/v1").rstrip("/")


def resolve_hindsight_url() -> str:
    return os.environ.get("HIVI_HINDSIGHT_URL", "http://127.0.0.1:8888").rstrip("/")


def fetch_model_id(inference_url: str, *, timeout_s: float = 10.0) -> str:
    r = httpx.get(f"{inference_url}/models", timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    if isinstance(data.get("data"), list) and data["data"]:
        mid = data["data"][0].get("id")
        if mid:
            return str(mid)
    if isinstance(data.get("models"), list) and data["models"]:
        name = data["models"][0].get("name") or data["models"][0].get("id")
        if name:
            return str(name)
    raise RuntimeError("No models advertised on inference server")


def chat_ping(inference_url: str, model_id: str, *, timeout_s: float = 60.0) -> dict[str, Any]:
    r = httpx.post(
        f"{inference_url}/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with OK only."}],
            "max_tokens": 8,
            "temperature": 0,
        },
        timeout=timeout_s,
    )
    r.raise_for_status()
    return r.json()


def run_preflight(
    *,
    project_id: str = "001",
    ensure_bank: bool = True,
    chat_check: bool = True,
) -> PreflightResult:
    inference_url = resolve_inference_url()
    hindsight_url = resolve_hindsight_url()
    errors: list[str] = []
    model_id: str | None = None
    hindsight_ok = False
    models_ok = False
    chat_ok = False
    bank_ok = False

    provider = HindsightProvider(base_url=hindsight_url)
    health = provider.health_check()
    hindsight_ok = health.ok
    if not hindsight_ok:
        errors.append(f"hindsight unhealthy: {health.detail}")

    try:
        model_id = fetch_model_id(inference_url)
        models_ok = True
    except Exception as e:
        errors.append(f"inference /models: {e}")

    if models_ok and chat_check and model_id:
        try:
            chat_ping(inference_url, model_id)
            chat_ok = True
        except Exception as e:
            errors.append(f"inference chat: {e}")

    if hindsight_ok and ensure_bank:
        try:
            mm = MemoryManager(provider=provider)
            out = mm.ensure_project_bank(project_id)
            bank_ok = bool(out.success)
            if not bank_ok:
                errors.append(f"bank ensure failed: {out.detail}")
            mm.close()
        except Exception as e:
            errors.append(f"bank ensure: {e}")

    ok = hindsight_ok and models_ok and (chat_ok if chat_check else True) and (
        bank_ok if ensure_bank else True
    )
    return PreflightResult(
        ok=ok,
        inference_url=inference_url,
        hindsight_url=hindsight_url,
        model_id=model_id,
        hindsight_ok=hindsight_ok,
        inference_models_ok=models_ok,
        inference_chat_ok=chat_ok,
        bank_ok=bank_ok,
        detail="ok" if ok else "failed",
        errors=errors,
    )
