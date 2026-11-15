"""HEALTH-1 status endpoint, probe caches, and streaming speed metrics."""
from __future__ import annotations

import asyncio
import json
import time
from typing import ClassVar

import httpx
import pytest
import server


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://hivi")


def _fast(*, used=100, total=1000, temp=60):
    return {
        "gpu": {
            "name": "Test GPU",
            "vram_used_mib": used,
            "vram_total_mib": total,
            "temp_c": temp,
            "power_w": 60,
            "power_limit_w": 100,
            "throttle_mask": "0x0",
        },
        "processes": [],
        "system": {"ram_used_mib": 1000, "ram_total_mib": 8000, "cpu_pct": 12, "disk_free_gb": 42},
    }


def _slow(*, devices=None, battery=None):
    return {
        "llama_server": "llama-server.exe",
        "llama_found": True,
        "devices_ran": True,
        "devices": devices or [{"backend": "CUDA", "name": "Test GPU", "total_mib": 1000}],
        "battery": battery,
        "power_plan": "Best performance",
    }


@pytest.fixture(autouse=True)
def reset_health_cache():
    server._FAST_CACHE.update(at=0.0, value=None)
    server._SLOW_CACHE.update(at=0.0, key=None, value=None)
    yield
    server._FAST_CACHE.update(at=0.0, value=None)
    server._SLOW_CACHE.update(at=0.0, key=None, value=None)


async def test_health_shape_and_levels(monkeypatch):
    monkeypatch.setattr(server.gpu_doctor, "collect_fast", lambda: _fast(used=960, total=1000, temp=88))
    monkeypatch.setattr(server.gpu_doctor, "collect_slow", lambda: _slow(
        battery={"percent": 40, "plugged": False},
    ))
    async with _client() as client:
        body = (await client.get("/api/health")).json()
    assert body["level"] == "red"
    assert {"gpu", "system", "model", "speed", "warnings", "level", "sampled_at"} <= body.keys()
    assert body["gpu"]["vram_free_mib"] == 40
    assert body["warnings"][0]["level"] == "red"


@pytest.mark.parametrize(
    ("fast", "slow", "expected"),
    [
        (_fast(used=950), _slow(), "red"),
        (_fast(temp=87), _slow(), "red"),
        (_fast(used=850), _slow(), "amber"),
        (_fast(), _slow(battery={"percent": 20, "plugged": False}), "amber"),
        (_fast(), _slow(devices=[{"backend": "Vulkan", "name": "Intel", "total_mib": 1000}]), "red"),
    ],
)
async def test_health_level_rules(monkeypatch, fast, slow, expected):
    monkeypatch.setattr(server.gpu_doctor, "collect_fast", lambda: fast)
    monkeypatch.setattr(server.gpu_doctor, "collect_slow", lambda: slow)
    async with _client() as client:
        body = (await client.get("/api/health")).json()
    assert body["level"] == expected


async def test_probe_cache_and_inflight_sharing(monkeypatch):
    calls = {"fast": 0, "slow": 0}

    async def fast_probe():
        calls["fast"] += 1
        await asyncio.sleep(0.02)
        return _fast()

    async def slow_probe():
        calls["slow"] += 1
        await asyncio.sleep(0.02)
        return _slow()

    def fast_sync():
        return asyncio.run(fast_probe())

    def slow_sync():
        return asyncio.run(slow_probe())

    monkeypatch.setattr(server.gpu_doctor, "collect_fast", fast_sync)
    monkeypatch.setattr(server.gpu_doctor, "collect_slow", slow_sync)
    async with _client() as client:
        results = await asyncio.gather(client.get("/api/health"), client.get("/api/health"))
        assert all(r.status_code == 200 for r in results)
        await client.get("/api/health")
    assert calls == {"fast": 1, "slow": 1}


async def test_probe_failure_returns_null_sections(monkeypatch):
    def broken():
        raise TimeoutError("probe timeout")

    monkeypatch.setattr(server.gpu_doctor, "collect_fast", broken)
    monkeypatch.setattr(server.gpu_doctor, "collect_slow", broken)
    async with _client() as client:
        response = await client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["gpu"] is None and body["system"] == {}


async def test_stream_speed_timings_and_bytes_unchanged(monkeypatch):
    raw = (
        b"data: " + json.dumps({"choices": [{"delta": {"content": "Hi"}}]}).encode() + b"\n\n"
        b"data: " + json.dumps({
            "timings": {"predicted_per_second": 41.0, "predicted_n": 12},
        }).encode() + b"\n\n"
        b"data: [DONE]\n\n"
    )

    class Upstream:
        headers: ClassVar[dict[str, str]] = {"content-type": "text/event-stream"}
        status_code = 200

        async def aiter_raw(self):
            yield raw[:20]
            yield raw[20:]

        async def aclose(self):
            pass

    class Client:
        def build_request(self, *args, **kwargs):
            return object()

        async def send(self, *args, **kwargs):
            return Upstream()

        async def aclose(self):
            pass

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda *args, **kwargs: Client())
    started = time.perf_counter()
    response = await server._proxy_stream(
        httpx.Request("POST", "http://hivi/v1/chat/completions"),
        "v1/chat/completions",
        b"{}",
        speed_started=started,
        speed_model="test-model",
        speed_engine="llama",
    )
    chunks = [chunk async for chunk in response.body_iterator]
    assert b"".join(chunks) == raw
    assert server.STATE.last_speed["tok_s"] == 41.0
    assert server.STATE.last_speed["approx"] is False
    assert server.STATE.last_speed["tokens"] == 12


async def test_stream_without_timings_is_approximate(monkeypatch):
    events = [
        b'data: {"choices":[{"delta":{"content":"one"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"two"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    class Upstream:
        headers: ClassVar[dict[str, str]] = {"content-type": "text/event-stream"}
        status_code = 200

        async def aiter_raw(self):
            for event in events:
                yield event

        async def aclose(self):
            pass

    class Client:
        def build_request(self, *args, **kwargs):
            return object()

        async def send(self, *args, **kwargs):
            return Upstream()

        async def aclose(self):
            pass

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda *args, **kwargs: Client())
    response = await server._proxy_stream(
        httpx.Request("POST", "http://hivi/v1/chat/completions"),
        "v1/chat/completions",
        b"{}",
        speed_started=time.perf_counter(),
        speed_model="test-model",
        speed_engine="colibri",
    )
    [chunk async for chunk in response.body_iterator]
    assert server.STATE.last_speed["approx"] is True
    assert server.STATE.last_speed["tokens"] == 2
