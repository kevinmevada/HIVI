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


def _reset_caches():
    server._FAST_CACHE.update(at=0.0, value=None, gpu_at=0.0, gpu_wall=None)
    server._SLOW_CACHE.update(at=0.0, key=None, value=None)
    server._PROBE_TASKS.clear()


def _cancel_probe_tasks():
    for task in list(server._PROBE_TASKS.values()):
        if not task.done():
            task.cancel()


def _idle_status() -> dict:
    """What status_dict() says with no model loaded, without touching the network."""
    return {"model_id": None, "name": None, "engine": None, "status": "idle", "detail": "",
            "pid": None, "healthy": False, "router": False, "resident": [], "timings": {},
            "context_size": None, "speed": None, "vision": False, "switching": False}


@pytest.fixture(autouse=True)
def reset_health_cache(monkeypatch):
    # These tests are about the GPU readings: the model side is pinned to "idle"
    # so whatever other tests left in STATE can't add warnings or real backend
    # probes (status_dict() calls the backend when a model is "ready").
    monkeypatch.setattr(server, "status_dict", _idle_status)
    _cancel_probe_tasks()
    _reset_caches()
    yield
    _cancel_probe_tasks()
    _reset_caches()


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


async def test_readings_stay_live_while_slow_probe_runs(monkeypatch):
    """A slow llama-server --list-devices must never hold back the GPU readings."""
    temps = iter(range(50, 100))
    monkeypatch.setattr(server.gpu_doctor, "collect_fast", lambda: _fast(temp=next(temps)))
    monkeypatch.setattr(server, "_SLOW_FIRST_WAIT", 0.05)
    monkeypatch.setattr(server, "_FAST_PROBE_TTL", 0.0)

    def slow_sync():
        time.sleep(0.6)
        return _slow()

    monkeypatch.setattr(server.gpu_doctor, "collect_slow", slow_sync)
    async with _client() as client:
        seen = []
        started = time.monotonic()
        for _ in range(3):
            body = (await client.get("/api/health")).json()
            seen.append(body["gpu"]["temp_c"])
        elapsed = time.monotonic() - started
    assert elapsed < 0.5
    assert len(set(seen)) == 3


async def test_slow_nvidia_smi_answers_with_last_reading(monkeypatch):
    calls = {"n": 0}

    def fast_sync():
        calls["n"] += 1
        if calls["n"] > 1:
            time.sleep(0.5)
        return _fast(temp=60 + calls["n"])

    monkeypatch.setattr(server.gpu_doctor, "collect_fast", fast_sync)
    monkeypatch.setattr(server.gpu_doctor, "collect_slow", _slow)
    monkeypatch.setattr(server, "_FAST_PROBE_TTL", 0.0)
    monkeypatch.setattr(server, "_FAST_WAIT", 0.05)
    async with _client() as client:
        first = (await client.get("/api/health")).json()
        started = time.monotonic()
        second = (await client.get("/api/health")).json()
        assert time.monotonic() - started < 0.3
        assert second["gpu"]["temp_c"] == first["gpu"]["temp_c"] == 61
        await asyncio.sleep(0.6)
        third = (await client.get("/api/health")).json()
    assert third["gpu"]["temp_c"] == 62


async def test_failed_probe_keeps_last_reading_then_drops_it(monkeypatch):
    state = {"ok": True}

    def fast_sync():
        if not state["ok"]:
            raise TimeoutError("nvidia-smi hung")
        return _fast(temp=70)

    monkeypatch.setattr(server.gpu_doctor, "collect_fast", fast_sync)
    monkeypatch.setattr(server.gpu_doctor, "collect_slow", _slow)
    monkeypatch.setattr(server, "_FAST_PROBE_TTL", 0.0)
    async with _client() as client:
        assert (await client.get("/api/health")).json()["gpu"]["stale"] is False
        state["ok"] = False
        server._FAST_CACHE["gpu_wall"] -= 12  # last good reading is now 12 s old
        kept = (await client.get("/api/health")).json()["gpu"]
        assert kept["temp_c"] == 70 and kept["stale"] is True and kept["age_s"] >= 12
        server._FAST_CACHE["gpu_at"] -= server._GPU_KEEP_S
        await client.get("/api/health")
        assert (await client.get("/api/health")).json()["gpu"] is None


def test_process_table_never_blocks_a_reading(monkeypatch):
    import gpu_doctor

    def slow_run(args, timeout=20.0):
        if len(args) == 1:
            time.sleep(0.5)
            return "| Processes: |"
        return "GPU, 8151, 4000, 10, 40, 80, 80, P0, 2400, 3090, 60, 0x0"

    monkeypatch.setattr(gpu_doctor, "_run", slow_run)
    gpu_doctor._PROCESSES_CACHE.update(at=0.0, value=[], running=False)
    started = time.monotonic()
    out = gpu_doctor.collect_fast()
    assert time.monotonic() - started < 0.3
    assert out["gpu"]["temp_c"] == 60
