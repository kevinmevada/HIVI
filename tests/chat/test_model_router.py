"""Retrieval-model registration, role/alias routing, auto-swap, and side services."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import aux_services
import httpx
import model_router
import pytest
import registry
import server
from fastapi.responses import Response

from routing import select_model as route_select


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * 16)
    return p


@pytest.fixture()
def library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A model folder shaped like the real HIVI library."""
    root = tmp_path / "Models"
    monkeypatch.setenv("HIVI_MODELS_ROOT", str(root))
    for rel in (
        "Embedding/Qwen3-Embedding-4B/Qwen3-Embedding-4B-Q4_K_M.gguf",
        "Reranker/Qwen3-Reranker-4B/Qwen3-Reranker-4B.Q4_K_M.gguf",
        "Coding/qwen2.5-coder-1.5b-fim/qwen2.5-coder-1.5b-q8_0.gguf",
        "General/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "General/Qwen3.6-35B-A3B/mmproj-F16.gguf",
        "General/heretic/Qwen3.6-35B-A3B-uncensored-heretic-Q4_K_M.gguf",
        "General/Bolt/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf",
        "Chat/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf",
        "General/Qwen3.5-9B/Qwen3.5-9B-Q4_K_M.gguf",
    ):
        _touch(root / rel)
    return root


def _by_kind(kind: str) -> list[registry.ModelEntry]:
    return [m for m in registry.scan_models() if m.kind == kind]


# ---- registry ---------------------------------------------------------------

def test_retrieval_models_are_classified_and_use_llama(library):
    (emb,) = _by_kind(registry.EMBEDDING)
    (rer,) = _by_kind(registry.RERANKER)
    assert emb.engine == "llama" and rer.engine == "llama"  # never Colibrì
    assert emb.name == "Qwen3 Embedding 4B"
    assert rer.name == "Qwen3 Reranker 4B" and rer.quant == "Q4_K_M"


def test_chat_dropdown_hides_retrieval_models(library):
    names = [m["name"] for m in registry.models_as_dicts()]
    assert not any("Embedding" in n or "Reranker" in n for n in names)
    every = registry.models_as_dicts(None)
    assert {m["kind"] for m in every} == {"chat", "embedding", "reranker", "fim"}


def test_base_coder_is_autocomplete_only(library):
    (coder,) = _by_kind(registry.FIM)
    assert "1.5b" in coder.path.lower() and coder.engine == "llama"
    assert coder.id not in {m["id"] for m in registry.models_as_dicts()}
    assert model_router.resolve(coder.id, config=CFG) is None


def test_generic_moe_params(library):
    bolt = next(m for m in registry.scan_models() if "74b" in m.path.lower())
    assert bolt.params == "74B MoE (~9B active)"


def test_prompt_router_never_picks_retrieval_models(library):
    decision = route_select("embed and rerank these documents for search")
    assert decision.model is not None and decision.model.kind == "chat"


# ---- role / alias resolution -----------------------------------------------

CFG = model_router.parse_config({
    "aliases": {
        "hivi-heavy": {"match": ["qwen3.6-35b-a3b"], "exclude": ["heretic"]},
        "hivi-review": {"match": ["bolt", "74b"]},
        "hivi-fast": {"match": ["qwen3.5-9b"]},
    },
    "roles": {"main-chef": "hivi-heavy", "security-auditor": "hivi-review",
              "researcher": "hivi-fast"},
})


def test_roles_resolve_through_aliases(library):
    chef = model_router.resolve("main-chef", config=CFG)
    assert chef and "heretic" not in chef.model.path.lower()
    assert chef.model.name == "Qwen3.6-35B-A3B" and chef.via == "role:main-chef->hivi-heavy"
    auditor = model_router.resolve("Security-Auditor", config=CFG)  # case-insensitive
    assert auditor and "74b" in auditor.model.path.lower()


def test_exact_registry_id_and_unknown_names(library):
    fast = model_router.resolve("researcher", config=CFG)
    assert model_router.resolve(fast.model.id, config=CFG).via == "id"
    assert model_router.resolve("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", config=CFG) is None
    assert model_router.resolve(None, config=CFG) is None


def test_retrieval_models_are_not_routable(library):
    emb = _by_kind(registry.EMBEDDING)[0]
    assert model_router.resolve(emb.id, config=CFG) is None


def test_shipped_config_covers_every_role_in_the_pipeline():
    cfg = model_router.load_config()
    for role in ("main-chef", "developer", "researcher", "security-auditor",
                 "integration-reviewer", "devops", "documentation", "acceptance"):
        assert cfg.roles[role] in cfg.aliases


# ---- side-service command lines ---------------------------------------------

def test_aux_args(library, monkeypatch):
    monkeypatch.delenv("HIVI_AUX_GPU_LAYERS", raising=False)
    emb = _by_kind(registry.EMBEDDING)[0]
    rer = _by_kind(registry.RERANKER)[0]
    llama = Path("llama-server")
    e = aux_services.build_aux_args("embedding", emb, llama=llama, port=8081, threads=4)
    assert "--embedding" in e and e[e.index("--pooling") + 1] == "last"
    assert e[e.index("--port") + 1] == "8081" and e[e.index("-ngl") + 1] == "0"
    r = aux_services.build_aux_args("reranker", rer, llama=llama, port=8082, threads=4)
    assert "--reranking" in r and "--embedding" not in r


def test_aux_idle_stop_respects_inflight(monkeypatch):
    svc = aux_services.AuxService("embedding")
    stopped: list[bool] = []
    monkeypatch.setattr(svc, "_stop_locked", lambda: stopped.append(True))
    svc.status, svc.last_used = "ready", 0.0
    svc.enter()
    svc.last_used = 0.0
    assert svc.stop_if_idle(now=10_000) is False  # a request is running
    svc.leave()
    svc.last_used = 0.0
    assert svc.stop_if_idle(now=10_000) is True and stopped == [True]


# ---- server: auto-swap -----------------------------------------------------

def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://hivi")


@pytest.fixture()
def router_env(library, monkeypatch):
    """Fake engine switch + fake backend; records what happened, in order."""
    events: list[str] = []
    seen: list[dict] = []

    def fake_select(model):
        events.append(f"switch:{model.name}")
        time.sleep(0.3)
        server.STATE.model_id, server.STATE.status = model.id, "ready"
        server.STATE.detail = "ready"
        return server.STATE

    async def fake_proxy(
        request, path, body=None, extra_headers=None, *, base=server.BACKEND_BASE
    ):
        payload = json.loads(body) if body else {}
        seen.append({"path": path, "body": payload, "base": base, "headers": extra_headers or {}})
        events.append(f"proxy:{payload.get('tag', '')}:start")
        await asyncio.sleep(float(payload.get("hold", 0)))
        events.append(f"proxy:{payload.get('tag', '')}:end")
        return Response(content=b"{}", media_type="application/json",
                        headers=extra_headers or {})

    monkeypatch.setattr(server, "select_model", fake_select)
    monkeypatch.setattr(server, "_proxy_buffered", fake_proxy)
    monkeypatch.setattr(server, "_current_backend_model", lambda: "backend-own-id")
    monkeypatch.setattr(server, "_adapt_chat_body", lambda raw: (raw, {}))
    monkeypatch.setattr(model_router, "load_config", lambda path=None: CFG)
    server.STATE.model_id, server.STATE.status = None, "idle"
    yield events, seen
    server.STATE.model_id, server.STATE.status = None, "idle"


async def test_role_request_loads_model_once_then_reuses_it(router_env):
    events, seen = router_env
    async with _client() as c:
        r1 = await c.post("/v1/chat/completions", json={"model": "main-chef", "messages": []})
        r2 = await c.post("/v1/chat/completions", json={"model": "hivi-heavy", "messages": []})
    assert r1.status_code == 200 and r2.status_code == 200
    assert [e for e in events if e.startswith("switch")] == ["switch:Qwen3.6-35B-A3B"]
    assert seen[0]["body"]["model"] == "backend-own-id"  # Colibrì-safe rewrite
    assert r1.headers["X-HIVI-Model-Via"] == "role:main-chef->hivi-heavy"


async def test_switch_waits_for_other_agents_inflight_reply(router_env):
    events, _ = router_env
    async with _client() as c:
        await c.post("/v1/chat/completions", json={"model": "main-chef", "messages": []})
        events.clear()
        dev = asyncio.create_task(c.post(
            "/v1/chat/completions",
            json={"model": "main-chef", "messages": [], "tag": "dev", "hold": 0.6}))
        await asyncio.sleep(0.1)
        audit = await c.post(
            "/v1/chat/completions",
            json={"model": "security-auditor", "messages": [], "tag": "audit"})
        assert (await dev).status_code == 200 and audit.status_code == 200
    review_switch = events.index("switch:NVIDIA Nemotron 3.5 Bolt 74B A9B 24GB")
    assert events.index("proxy:dev:end") < review_switch
    assert events[-1] == "proxy:audit:end"


async def test_unknown_model_name_never_switches(router_env):
    events, _ = router_env
    async with _client() as c:
        r = await c.post("/v1/chat/completions",
                         json={"model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", "messages": []})
    assert r.status_code == 200
    assert not any(e.startswith("switch") for e in events)


async def test_failed_load_is_reported(router_env, monkeypatch):
    def broken(model):
        server.STATE.status, server.STATE.detail = "error", "out of memory"
        return server.STATE

    monkeypatch.setattr(server, "select_model", broken)
    async with _client() as c:
        r = await c.post("/v1/chat/completions", json={"model": "devops-x", "messages": []})
        assert r.status_code == 200  # unknown role → no switch attempted
        r = await c.post("/v1/chat/completions", json={"model": "security-auditor", "messages": []})
    assert r.status_code == 502 and "out of memory" in r.json()["detail"]
    assert server.GATE.inflight == 0 and server.GATE.is_open


async def test_embeddings_use_side_service_not_chat_backend(router_env, monkeypatch):
    events, seen = router_env
    svc = aux_services.SERVICES["embedding"]
    monkeypatch.setattr(svc, "ensure", lambda: None)
    async with _client() as c:
        r = await c.post("/v1/embeddings", json={"model": "x", "input": ["hello"]})
    assert r.status_code == 200
    assert seen[-1]["base"] == svc.base and seen[-1]["path"] == "v1/embeddings"
    assert not any(e.startswith("switch") for e in events)


async def test_roles_endpoint(router_env):
    async with _client() as c:
        data = (await c.get("/api/roles")).json()
    assert data["roles"]["researcher"]["model"]["name"] == "Qwen3.5-9B"