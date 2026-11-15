"""Control server: event loop stays free during switches; memory gateway coordination."""
from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
from fastapi import HTTPException
from fastapi.responses import Response

import server


@pytest.fixture()
def fake_switch(monkeypatch):
    """Replace the real engine switch with a slow, blocking stand-in."""
    calls: list[str] = []

    def slow_select(model):
        calls.append(model.id)
        time.sleep(1.0)  # blocking, like waiting for llama-server to load
        server.STATE.status = "ready"
        return server.STATE

    entry = type("E", (), {"id": "m1", "name": "M1", "engine": "llama"})()
    monkeypatch.setattr(server, "select_model", slow_select)
    monkeypatch.setattr(server, "get_model", lambda mid: entry if mid == "m1" else None)
    monkeypatch.setattr(server, "status_dict", lambda: {"status": server.STATE.status})
    monkeypatch.setattr(server, "_MEMORY_DRAIN_S", 2.0)
    return calls


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://hivi")


async def test_status_answers_while_model_loads(fake_switch):
    async with _client() as c:
        select = asyncio.create_task(c.post("/api/select", json={"id": "m1"}))
        await asyncio.sleep(0.1)
        status = await c.get("/api/status")
        # If the switch blocked the event loop, it would finish before status could answer.
        assert not select.done(), "/api/status only answered after the model switch finished"
        assert status.status_code == 200
        assert (await select).status_code == 200


async def test_second_switch_is_rejected(fake_switch):
    async with _client() as c:
        first = asyncio.create_task(c.post("/api/select", json={"id": "m1"}))
        await asyncio.sleep(0.1)
        second = await c.post("/api/select", json={"id": "m1"})
        assert second.status_code == 409
        assert (await first).status_code == 200
    assert fake_switch == ["m1"]


async def test_memory_request_waits_for_switch_then_targets_new_model(fake_switch, monkeypatch):
    seen: dict = {}

    async def fake_proxy(request, path, body=None, extra_headers=None):
        seen["path"] = path
        seen["body"] = json.loads(body)
        seen["at"] = time.perf_counter()
        return Response(content=b"{}", media_type="application/json")

    monkeypatch.setattr(server, "_proxy_buffered", fake_proxy)
    monkeypatch.setattr(server, "_current_backend_model", lambda: "new-model")
    async with _client() as c:
        select = asyncio.create_task(c.post("/api/select", json={"id": "m1"}))
        await asyncio.sleep(0.1)
        t_sent = time.perf_counter()
        mem = await c.post("/memory/v1/chat/completions",
                           json={"model": "stale-model", "messages": []})
        sel = await select
    assert mem.status_code == 200 and sel.status_code == 200
    assert seen["at"] - t_sent > 0.5, "memory call should have waited for the switch"
    assert seen["body"]["model"] == "new-model"
    assert seen["path"] == "v1/chat/completions"


async def test_switch_does_not_wait_for_memory_and_memory_is_retried(fake_switch, monkeypatch):
    """A click on Load must not sit behind Hindsight's memory extraction."""
    order: list[str] = []
    calls: list[dict] = []

    async def proxy(request, path, body=None, extra_headers=None):
        calls.append(json.loads(body))
        gen = server.GATE.generation
        order.append(f"memory-start-{len(calls)}")
        await asyncio.sleep(0.6)
        if server.GATE.generation != gen:  # the switch killed the backend under us
            order.append("memory-cut")
            raise HTTPException(502, "backend unreachable")
        order.append("memory-end")
        return Response(content=b"{}", media_type="application/json")

    real_select = server.select_model

    def tracking_select(model):
        order.append("switch")
        server._current_backend_model = lambda: "new-model"
        return real_select(model)

    monkeypatch.setattr(server, "_proxy_buffered", proxy)
    monkeypatch.setattr(server, "_current_backend_model", lambda: "old-model")
    monkeypatch.setattr(server, "select_model", tracking_select)
    async with _client() as c:
        mem = asyncio.create_task(c.post("/memory/v1/chat/completions",
                                         json={"model": "x", "messages": []}))
        await asyncio.sleep(0.1)
        t = time.perf_counter()
        sel = await c.post("/api/select", json={"id": "m1"})
        switch_took = time.perf_counter() - t
        mem_resp = await mem
    assert sel.status_code == 200 and mem_resp.status_code == 200
    assert order.index("switch") < order.index("memory-cut")  # did not wait
    assert switch_took < 1.5  # fake load is 1.0 s; old behaviour added the 0.6 s memory call
    assert [c["model"] for c in calls] == ["old-model", "new-model"]  # retried on new model
    assert server.GATE.inflight == 0


async def test_memory_error_without_a_switch_is_not_retried(monkeypatch):
    calls = 0

    async def broken(request, path, body=None, extra_headers=None):
        nonlocal calls
        calls += 1
        raise HTTPException(502, "backend unreachable")

    monkeypatch.setattr(server, "_proxy_buffered", broken)
    monkeypatch.setattr(server, "_current_backend_model", lambda: None)
    async with _client() as c:
        r = await c.post("/memory/v1/chat/completions", json={"messages": []})
    assert r.status_code == 502 and calls == 1


async def test_switch_still_waits_for_a_chat_reply(fake_switch, monkeypatch):
    order: list[str] = []

    async def proxy(request, path, body=None, extra_headers=None, *, base=None):
        order.append("chat-start")
        await asyncio.sleep(0.6)
        order.append("chat-end")
        return Response(content=b"{}", media_type="application/json")

    real_select = server.select_model

    def tracking_select(model):
        order.append("switch")
        return real_select(model)

    monkeypatch.setattr(server, "_proxy_buffered", proxy)
    monkeypatch.setattr(server, "_adapt_chat_body", lambda raw: (raw, {}))
    monkeypatch.setattr(server.model_router, "resolve", lambda name: None)
    monkeypatch.setattr(server, "select_model", tracking_select)
    async with _client() as c:
        chat = asyncio.create_task(c.post("/v1/chat/completions",
                                          json={"model": "ui-model", "messages": []}))
        await asyncio.sleep(0.1)
        sel = await c.post("/api/select", json={"id": "m1"})
        await chat
    assert sel.status_code == 200
    assert order.index("chat-end") < order.index("switch")


async def test_download_status_is_not_hardcoded(monkeypatch, tmp_path):
    monkeypatch.setenv("HIVI_MODELS_ROOT", str(tmp_path))
    part = tmp_path / "Coding" / "x" / ".cache" / "huggingface" / "download" / "a.gguf.incomplete"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"\0" * 10)
    async with _client() as c:
        data = (await c.get("/api/download/status")).json()
    assert data["incomplete_bytes"] == 10 and data["active"] is True
    assert "repo" not in data


def test_router_mode_sends_the_router_model_name(monkeypatch):
    # In router mode /v1/models lists every preset; requests must name the active one.
    monkeypatch.setattr(server, "backend_model_name", lambda: "gpt-oss-20b-q8-0")
    assert server._current_backend_model() == "gpt-oss-20b-q8-0"


async def test_engine_pin_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("HIVI_MODELS_ROOT", str(tmp_path / "Models"))
    monkeypatch.setenv("HIVI_DATA_ROOT", str(tmp_path / "data"))
    f = tmp_path / "Models" / "Coding" / "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"\0" * 16)
    mid = "qwen3-6-35b-a3b-uncensored-hauhaucs-aggressive-q5-k-p"
    async with _client() as c:
        r = await c.post("/api/engine", json={"id": mid, "engine": "llama"})
        assert r.status_code == 200 and r.json()["engine"] == "llama"
        r = await c.post("/api/engine", json={"id": mid, "engine": "auto"})
        assert r.json()["engine"] == "colibri"
        assert (await c.post("/api/engine", json={"id": mid, "engine": "x"})).status_code == 400
        assert (await c.post("/api/engine", json={"id": "nope", "engine": "llama"})).status_code == 404


async def test_router_v1_models_shows_only_the_active_model(monkeypatch):
    monkeypatch.setattr(server, "backend_model_name", lambda: "b")

    async def fake_proxy(request, path, body=None, extra_headers=None, *, base=None):
        rows = [{"id": m, "object": "model"} for m in ("a", "b", "c")]
        return Response(content=json.dumps({"object": "list", "data": rows}),
                        media_type="application/json")

    monkeypatch.setattr(server, "_proxy_buffered", fake_proxy)
    async with _client() as c:
        r = await c.get("/v1/models")
    assert [row["id"] for row in r.json()["data"]] == ["b"]


async def test_ui_files_are_always_revalidated():
    """The desktop app keeps a cache between launches; it must never show an old home page."""
    async with _client() as c:
        home = await c.get("/")
        brand = await c.get("/static/brand/brand.css")
    assert home.status_code == 200 and home.headers["cache-control"] == "no-cache"
    assert 'class="hero-lockup"' in home.text and 'class="hero-bunny"' in home.text
    assert 'class="brand-bunny"' in home.text and 'data-text="HIVI"' in home.text
    assert brand.status_code == 200 and brand.headers["cache-control"] == "no-cache"
