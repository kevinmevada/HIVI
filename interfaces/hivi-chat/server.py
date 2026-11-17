"""HIVI control + chat UI. Routes models to Colibrì or llama.cpp automatically.

Model router: a chat request whose ``model`` names a role or alias from
configs/model_roles.yaml (or a registry id) makes HIVI load that model first,
after in-flight requests finish. Embedding and rerank requests go to their own
side services (aux_services.py) and never disturb the chat model.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aux_services
import backups
import chat_history
import gpu_doctor
import httpx
import image_gen
import memory_engine
import model_router
import personal_memory as pm
import soul
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from haie import plan_request
from hivi_paths import paths
from registry import (
    EMBEDDING,
    RERANKER,
    ModelEntry,
    get_model,
    is_uncensored,
    models_as_dicts,
    set_engine_pin,
)
from runtime import (
    BACKEND_BASE,
    STATE,
    SWITCH_LOCK,
    backend_model_name,
    note_speed,
    note_switch_wait,
    select_model,
    status_dict,
    stop_runtime,
)

from routing import select_model as route_model

HERE = Path(__file__).resolve().parent


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    backups.start_scheduler()
    yield
    memory_engine.EXTRACTOR.cancel()
    # Side services are HIVI-owned processes; don't leave them running.
    await asyncio.to_thread(aux_services.stop_all)


app = FastAPI(title="HIVI", lifespan=_lifespan)
_TIMEOUT = httpx.Timeout(600.0, connect=30.0)
_HAIE_EXPOSE = (
    "X-HIVI-HAIE-Path, X-HIVI-HAIE-Intents, X-HIVI-HAIE-Complexity, "
    "X-HIVI-HAIE-Deliverable, X-HIVI-HAIE-Reasoning, X-HIVI-HAIE-Length, "
    "X-HIVI-HAIE-Temp, X-HIVI-HAIE-Max-Tokens, X-HIVI-HAIE-Think, "
    "X-HIVI-Adaptive-Bucket, X-HIVI-Adaptive-Temp, "
    "X-HIVI-Adaptive-Max-Tokens, X-HIVI-Adaptive-Think, "
    "X-HIVI-Memory, X-HIVI-Mood, X-HIVI-Recalled"
)
# How long a memory (Hindsight) request may wait for a model switch to finish,
# and how long a switch waits for in-flight memory requests before proceeding.
_MEMORY_WAIT_S = float(os.environ.get("HIVI_MEMORY_WAIT_S", "600"))
_MEMORY_DRAIN_S = float(os.environ.get("HIVI_MEMORY_DRAIN_S", "120"))
# Router-triggered switches: how long a request may wait for its model, and how
# long a switch waits for other agents' in-flight generations to finish.
_SWITCH_WAIT_S = float(os.environ.get("HIVI_SWITCH_WAIT_S", "900"))
_SWITCH_DRAIN_S = float(os.environ.get("HIVI_SWITCH_DRAIN_S", "600"))
# A memory call cut off by a switch is resent this many times.
_MEMORY_RETRIES = int(os.environ.get("HIVI_MEMORY_RETRIES", "2"))

# Live GPU readings: a fresh nvidia-smi sample at most every 1.5 s. A request never
# waits behind a probe once there is a reading: it gets the last one (with its age)
# while the refresh runs in the background.
_FAST_PROBE_TTL = 1.5
_SLOW_PROBE_TTL = 60.0
_FAST_PROBE_TIMEOUT = 6.0
_SLOW_PROBE_TIMEOUT = 75.0
# The first slow probe (llama-server --list-devices) gets this long before the
# response goes out without its warnings; they appear on a later poll.
_SLOW_FIRST_WAIT = 2.0
# A GPU reading this old is still shown if nvidia-smi hiccups, marked stale.
_GPU_KEEP_S = 30.0
_FAST_WAIT = 1.2
_FAST_CACHE = {"at": 0.0, "value": None, "gpu_at": 0.0, "gpu_wall": None}
_SLOW_CACHE = {"at": 0.0, "key": None, "value": None}
_PROBE_TASKS: dict[str, asyncio.Task] = {}
_IMAGE_ID = re.compile(r"^[A-Za-z0-9_-]{16,64}\.png$")


class BackendGate:
    """Keeps memory traffic and model switches from stepping on each other.

    Hindsight shares the one local backend with chat. Without coordination a
    model switch kills the backend mid-extraction and the retain fails. With the
    gate, requests that arrive during a switch wait for the new model instead of
    hitting a dead port.

    Foreground requests (chat, agents) are waited for. Background requests
    (Hindsight memory) are not: a switch cuts them off and the memory proxy
    retries them on the new model, so a click on "Load" never sits behind a
    minute of memory extraction. ``generation`` changes on every switch, which is
    how a failed background request knows it was cut off rather than broken.
    """

    def __init__(self) -> None:
        self._open = asyncio.Event()
        self._open.set()
        self._idle = asyncio.Event()  # no foreground requests in flight
        self._idle.set()
        self._fg = 0
        self._bg = 0
        self.generation = 0

    @property
    def inflight(self) -> int:
        return self._fg + self._bg

    @property
    def foreground(self) -> int:
        return self._fg

    @property
    def background(self) -> int:
        return self._bg

    @property
    def is_open(self) -> bool:
        return self._open.is_set()

    def close(self) -> None:
        self.generation += 1
        self._open.clear()

    def open(self) -> None:
        self._open.set()

    async def wait_idle(self, timeout: float) -> bool:
        """Wait for foreground requests only; background ones are retried later."""
        try:
            await asyncio.wait_for(self._idle.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def enter(self, timeout: float, *, background: bool = False) -> bool:
        try:
            await asyncio.wait_for(self._open.wait(), timeout)
        except asyncio.TimeoutError:
            return False
        if background:
            self._bg += 1
        else:
            self._fg += 1
            self._idle.clear()
        return True

    def leave(self, *, background: bool = False) -> None:
        if background:
            self._bg = max(0, self._bg - 1)
            return
        self._fg = max(0, self._fg - 1)
        if self._fg == 0:
            self._idle.set()


GATE = BackendGate()


def _current_is_uncensored() -> bool:
    """Loaded model is an uncensored build (abliterated / heretic / HauhauCS)."""
    if not STATE.model_id:
        return False
    try:
        entry = get_model(STATE.model_id)
    except Exception:
        entry = None
    if entry is not None:
        return bool(entry.uncensored)
    return is_uncensored(STATE.model_id)


def _adapt_chat_body(raw: bytes) -> tuple[bytes, dict[str, str]]:
    # Blocking (may call the backend for semantic classify) — run via to_thread.
    """HAIE: fast-path or semantic classify, then inject dynamic sampling params."""
    extra: dict[str, str] = {}
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        return raw, extra
    if not isinstance(body, dict):
        return raw, extra
    adapted, meta = plan_request(
        body,
        engine=STATE.engine,
        backend_base=BACKEND_BASE,
        model_id=STATE.model_id,
        uncensored=_current_is_uncensored(),
    )
    extra = meta.to_headers()
    return json.dumps(adapted).encode("utf-8"), extra


# The UI files change with every HIVI update. Without this header, the desktop app's window
# (which keeps a cache between launches) can keep showing an old home page. "no-cache" still
# lets it reuse a file when the server confirms it hasn't changed, so this costs almost nothing.
_NO_CACHE = {"Cache-Control": "no-cache"}



@app.get("/")
def index():
    return FileResponse(HERE / "index.html", headers=_NO_CACHE)


@app.get("/api/models")
def api_models(kind: str = "chat"):
    """Chat models by default; ?kind=embedding|reranker|all for the rest."""
    return {"models": models_as_dicts(None if kind == "all" else kind)}


@app.get("/api/roles")
async def api_roles():
    """What each agent role / alias in configs/model_roles.yaml resolves to."""
    return await asyncio.to_thread(model_router.describe)


@app.get("/api/gpu")
async def api_gpu():
    """GPU health: CUDA build, power, throttling, who holds VRAM (see gpu_doctor.py)."""
    return (await asyncio.to_thread(gpu_doctor.collect)).to_dict()


@app.get("/api/aux")
async def api_aux():
    return aux_services.status_all()


@app.post("/api/aux/stop")
async def api_aux_stop():
    await asyncio.to_thread(aux_services.stop_all)
    return aux_services.status_all()


@app.post("/api/route")
async def api_route(request: Request):
    body = await request.json()
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(400, "prompt must be a non-empty string")
    has_image = bool(body.get("has_image"))
    decision = await asyncio.to_thread(route_model, prompt, has_image=has_image)
    return {
        "model": None if decision.model is None else {
            "id": decision.model.id,
            "name": decision.model.name,
            "engine": decision.model.engine,
            "params": decision.model.params,
            "quant": decision.model.quant,
        },
        "task": decision.classification.task,
        "complexity": decision.classification.complexity,
        "score": decision.score,
        "reason": decision.reason,
    }


def _unload_chat_for_image() -> str | None:
    """Free the GPU for a render. Returns the chat model id to reload afterwards."""
    model_id = STATE.model_id
    if not model_id:
        return None
    if not SWITCH_LOCK.acquire(timeout=120):
        raise RuntimeError("a model switch is in progress; try again in a moment")
    try:
        stop_runtime()
    finally:
        SWITCH_LOCK.release()
    return model_id


def _reload_chat_after_image(model_id: str) -> None:
    model = get_model(model_id)
    if model is None:
        return
    if not SWITCH_LOCK.acquire(timeout=120):
        return
    try:
        select_model(model)
    finally:
        SWITCH_LOCK.release()


async def _run_image_job(job: image_gen.Job) -> None:
    # Chat requests wait while the GPU belongs to the image model.
    GATE.close()
    try:
        await GATE.wait_idle(_MEMORY_DRAIN_S)
        await asyncio.to_thread(
            image_gen.run_job, job,
            unload_chat=_unload_chat_for_image, reload_chat=_reload_chat_after_image,
        )
    finally:
        GATE.open()


@app.get("/api/images/status")
async def api_images_status():
    return await asyncio.to_thread(image_gen.status)


@app.get("/api/images/gallery")
async def api_images_gallery(limit: int = 60):
    return {"images": await asyncio.to_thread(image_gen.list_images, max(1, min(500, limit)))}


@app.post("/api/images/generate")
async def api_images_generate(request: Request):
    body = await request.json()
    prompt = body.get("prompt") if isinstance(body, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(400, "prompt must be a non-empty string")
    try:
        width = int(body.get("width", 768))
        height = int(body.get("height", 768))
        steps = int(body.get("steps", image_gen.DEFAULT_STEPS))
        seed = body.get("seed")
        seed = int(seed) if seed not in (None, "") else None
    except (TypeError, ValueError):
        raise HTTPException(400, "width, height, steps and seed must be numeric") from None
    if width not in image_gen.SIZES or height not in image_gen.SIZES:
        raise HTTPException(400, f"width and height must be one of {list(image_gen.SIZES)}")
    if steps < 1 or steps > image_gen.MAX_STEPS:
        raise HTTPException(400, f"steps must be between 1 and {image_gen.MAX_STEPS}")
    if len(prompt.strip()) > 2000:
        raise HTTPException(400, "prompt must be 2000 characters or fewer")
    setup = await asyncio.to_thread(image_gen.status)
    if not setup["ready"]:
        raise HTTPException(503, "; ".join(setup["missing"]))
    job = image_gen.new_job(prompt.strip(), width=width, height=height, steps=steps, seed=seed)
    asyncio.get_running_loop().create_task(_run_image_job(job))
    return job.to_dict()


@app.get("/api/images/jobs/{job_id}")
async def api_images_job(job_id: str):
    job = image_gen.get_job(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job.to_dict()


@app.post("/api/images/jobs/{job_id}/cancel")
async def api_images_job_cancel(job_id: str):
    job = image_gen.get_job(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    job.cancel()
    return job.to_dict()


@app.get("/api/images/{filename}")
async def api_image(filename: str):
    if not _IMAGE_ID.fullmatch(filename):
        raise HTTPException(404, "image not found")
    image = paths.data_root() / "images" / filename
    if not image.is_file():
        raise HTTPException(404, "image not found")
    return FileResponse(image, media_type="image/png", headers=_NO_CACHE)


@app.get("/api/status")
async def api_status():
    out = await asyncio.to_thread(status_dict)
    out["memory_inflight"] = GATE.background
    out["chat_inflight"] = GATE.foreground
    return out


def _probe_task(name: str, make) -> asyncio.Task:
    """One running probe per kind, shared by every request that needs it."""
    task = _PROBE_TASKS.get(name)
    if task is None or task.done() or task.get_loop() is not asyncio.get_running_loop():
        task = asyncio.get_running_loop().create_task(make())
        _PROBE_TASKS[name] = task
    return task


async def _run_fast_probe() -> None:
    try:
        value = await asyncio.wait_for(
            asyncio.to_thread(gpu_doctor.collect_fast),
            timeout=_FAST_PROBE_TIMEOUT,
        )
    except Exception:  # noqa: BLE001 - a failed probe must not fail /api/health
        value = None
    now = time.monotonic()
    previous = _FAST_CACHE["value"] or {}
    if value is None or not value.get("gpu"):
        # nvidia-smi can miss a beat under heavy load: keep the last GPU reading for
        # a while (marked stale) instead of blanking the chip.
        kept = previous.get("gpu")
        gpu_at = _FAST_CACHE.get("gpu_at", 0.0)
        if kept and now - gpu_at < _GPU_KEEP_S:
            value = dict(value or {})
            value["gpu"] = kept
            value.setdefault("processes", previous.get("processes") or [])
            value.setdefault("system", previous.get("system") or {})
            _FAST_CACHE.update(at=now, value=value, gpu_wall=_FAST_CACHE.get("gpu_wall"))
            return
        _FAST_CACHE.update(at=now, value=value, gpu_at=0.0, gpu_wall=None)
        return
    _FAST_CACHE.update(at=now, value=value, gpu_at=now, gpu_wall=time.time())


async def _cached_fast_probe() -> dict | None:
    fresh = time.monotonic() - _FAST_CACHE["at"] < _FAST_PROBE_TTL
    if fresh:
        return _FAST_CACHE["value"]
    task = _probe_task("fast", _run_fast_probe)
    if _FAST_CACHE["at"] == 0.0:
        # Nothing to show yet: wait for the first reading.
        await asyncio.shield(task)
    else:
        # A normal reading takes well under a second; if nvidia-smi is slow right
        # now, answer with the last reading and let the probe finish on its own.
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=_FAST_WAIT)
        except TimeoutError:
            pass
    return _FAST_CACHE["value"]


async def _run_slow_probe(key) -> None:
    try:
        value = await asyncio.wait_for(
            asyncio.to_thread(gpu_doctor.collect_slow),
            timeout=_SLOW_PROBE_TIMEOUT,
        )
    except Exception:  # noqa: BLE001 - a failed probe must not fail /api/health
        value = None
    _SLOW_CACHE.update(at=time.monotonic(), key=key, value=value)


async def _cached_slow_probe() -> dict | None:
    key = (STATE.model_id, STATE.engine)
    if (_SLOW_CACHE["key"] == key
            and time.monotonic() - _SLOW_CACHE["at"] < _SLOW_PROBE_TTL):
        return _SLOW_CACHE["value"]
    # Slow probes (llama-server --list-devices, power APIs) refresh in the
    # background; the live GPU readings never wait for them.
    task = _probe_task("slow", lambda: _run_slow_probe(key))
    if _SLOW_CACHE["at"] == 0.0:
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=_SLOW_FIRST_WAIT)
        except TimeoutError:
            pass
    return _SLOW_CACHE["value"]


def _warning_text(finding: dict) -> tuple[str, str] | None:
    text = str(finding.get("text", ""))
    lower = text.lower()
    if "nvidia-smi did not answer" in lower:
        return None
    if "not found at" in lower:
        return None
    if "not a cuda build" in lower or "no gpu at all" in lower:
        return "red", "This llama.cpp build is CPU-only"
    if "on battery" in lower:
        return "amber", "On battery"
    if "power limit is" in lower:
        return "amber", "GPU is power-limited"
    if "held back right now" in lower:
        return "amber", "GPU is being throttled"
    if "gpu at" in lower:
        return "red", "GPU is too hot"
    if "windows power mode" in lower:
        return "amber", "Windows power mode may limit performance"
    if "other programs are on" in lower:
        return "amber", "Other programs are using GPU memory"
    if "vram is unused" in lower:
        return "amber", text.split(". ", 1)[0]
    if finding.get("level") in ("bad", "warn"):
        return ("red" if finding["level"] == "bad" else "amber"), text
    return None


def _health_warnings(fast: dict, slow: dict, model: dict, speed: dict | None) -> list[dict]:
    gpu = fast.get("gpu")
    devices = slow.get("devices") or []
    report = gpu_doctor.Report(
        llama_server=slow.get("llama_server", ""),
        devices=devices,
        gpu=gpu,
        processes=fast.get("processes") or [],
        battery=slow.get("battery"),
        power_plan=slow.get("power_plan"),
    )
    findings = gpu_doctor.diagnose(
        report,
        llama_found=bool(slow.get("llama_found")),
        devices_ran=bool(slow.get("devices_ran")),
    )
    warnings: list[dict] = []
    for finding in findings:
        item = _warning_text(finding)
        if item:
            level, text = item
            if not any(w["text"] == text for w in warnings):
                warnings.append({"level": level, "text": text})

    if gpu:
        total = gpu.get("vram_total_mib") or 0
        used = gpu.get("vram_used_mib") or 0
        if total and used / total >= 0.95:
            warnings.append({"level": "red", "text": "VRAM is at least 95% full"})
        elif total and used / total >= 0.85:
            warnings.append({"level": "amber", "text": "VRAM is at least 85% full"})
        temp = gpu.get("temp_c") or 0
        if temp >= 87 and not any(w["text"] == "GPU is too hot" for w in warnings):
            warnings.append({"level": "red", "text": "GPU is too hot"})
        elif temp >= 80 and not any(w["text"] == "GPU is too hot" for w in warnings):
            warnings.append({"level": "amber", "text": "GPU temperature is high"})

    if model.get("status") == "unhealthy":
        warnings.append({"level": "red", "text": "Backend is unhealthy"})
    priority = {"red": 0, "amber": 1}
    return sorted(warnings, key=lambda w: priority[w["level"]])


def _health_model(status: dict) -> dict:
    raw_status = status.get("status")
    if raw_status == "ready" and status.get("healthy"):
        model_status = "ready"
    elif raw_status in ("starting", "loading"):
        model_status = "loading"
    elif status.get("model_id") or raw_status in ("error", "unhealthy"):
        model_status = "unhealthy"
    else:
        model_status = "idle"
    return {
        "name": status.get("name"),
        "engine": status.get("engine"),
        "router": bool(status.get("router")),
        "context_size": status.get("context_size"),
        "status": model_status,
        "timings": status.get("timings") or {},
    }


@app.get("/api/health")
async def api_health():
    fast, slow, status = await asyncio.gather(
        _cached_fast_probe(),
        _cached_slow_probe(),
        asyncio.to_thread(status_dict),
    )
    fast = fast or {}
    slow = slow or {}
    model = _health_model(status)
    speed = status.get("speed")
    gpu = fast.get("gpu")
    gpu_wall = _FAST_CACHE.get("gpu_wall")
    gpu_age = None if gpu is None or gpu_wall is None else max(0.0, time.time() - gpu_wall)
    if gpu is not None:
        gpu = dict(gpu)
        gpu["age_s"] = None if gpu_age is None else round(gpu_age, 1)
        gpu["stale"] = gpu_age is not None and gpu_age > 10
        total = gpu.get("vram_total_mib")
        used = gpu.get("vram_used_mib")
        gpu["vram_free_mib"] = (
            max(0.0, total - used) if total is not None and used is not None else None
        )
        gpu["throttle_reasons"] = gpu_doctor.throttle_reasons(gpu.get("throttle_mask"))
        gpu["processes"] = fast.get("processes") or []
    warnings = _health_warnings(fast, slow, model, speed)
    level = "red" if any(w["level"] == "red" for w in warnings) else (
        "amber" if warnings else "ok"
    )
    return {
        "gpu": gpu,
        "system": fast.get("system") or {},
        "model": model,
        "speed": speed,
        "warnings": warnings,
        "level": level,
        "sampled_at": time.time(),
    }


@app.get("/api/download/status")
def api_download_status():
    """In-progress Hugging Face downloads anywhere under the models root."""
    root = paths.models_root()
    now = time.time()
    files = []
    incomplete_bytes = 0
    completed_bytes = 0
    if root.exists():
        for path in root.rglob("*"):
            if not path.is_file() or ".cache" not in path.parts:
                continue
            st = path.stat()
            is_incomplete = path.name.endswith(".incomplete")
            if is_incomplete:
                incomplete_bytes += st.st_size
            else:
                completed_bytes += st.st_size
            files.append(
                {
                    "name": str(path.relative_to(root)),
                    "bytes": st.st_size,
                    "incomplete": is_incomplete,
                    "modified": st.st_mtime,
                }
            )
    active = any(now - item["modified"] < 60 for item in files if item["incomplete"])
    return {
        "models_root": str(root),
        "active": active,
        "incomplete_bytes": incomplete_bytes,
        "completed_bytes": completed_bytes,
        "files": sorted(files, key=lambda item: item["name"]),
    }


@app.post("/api/select")
async def api_select(request: Request):
    body = await request.json()
    mid = body.get("id")
    if not mid:
        raise HTTPException(400, "id required")
    model = await asyncio.to_thread(get_model, mid)
    if not model:
        raise HTTPException(404, f"unknown model id: {mid}")
    if not SWITCH_LOCK.acquire(blocking=False):
        raise HTTPException(409, "a model switch is already in progress")
    GATE.close()
    try:
        # Only chat replies are waited for; memory calls are retried after the switch.
        t0 = time.perf_counter()
        drained = await GATE.wait_idle(_MEMORY_DRAIN_S)
        waited = time.perf_counter() - t0
        if not drained:
            print(f"HIVI: switching with {GATE.foreground} chat request(s) still running")
        # Model loads take minutes; keep the event loop free for /api/status etc.
        note_switch_wait(waited)
        state = await asyncio.to_thread(select_model, model)
    finally:
        GATE.open()
        SWITCH_LOCK.release()
    code = 200 if state.status == "ready" else 500
    return JSONResponse(status_dict(), status_code=code)


@app.post("/api/engine")
async def api_engine(request: Request):
    """Pin a model to llama.cpp or Colibrì ({"id": ..., "engine": "llama"|"colibri"|"auto"}).

    Takes effect on the next load of that model.
    """
    body = await request.json()
    mid, engine = body.get("id"), body.get("engine")
    if not mid:
        raise HTTPException(400, "id required")
    if not await asyncio.to_thread(get_model, mid):
        raise HTTPException(404, f"unknown model id: {mid}")
    try:
        pins = await asyncio.to_thread(set_engine_pin, mid, engine)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    model = await asyncio.to_thread(get_model, mid)
    return {"id": mid, "engine": model.engine, "reason": model.engine_reason, "pins": pins}


@app.post("/api/stop")
async def api_stop():
    if not SWITCH_LOCK.acquire(blocking=False):
        raise HTTPException(409, "a model switch is in progress")
    GATE.close()
    try:
        await GATE.wait_idle(_MEMORY_DRAIN_S)
        await asyncio.to_thread(stop_runtime)
    finally:
        GATE.open()
        SWITCH_LOCK.release()
    return status_dict()


def _forward_headers(request: Request) -> dict[str, str]:
    return {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "accept-encoding")
    }


async def _proxy_buffered(
    request: Request,
    path: str,
    body: bytes | None = None,
    extra_headers: dict[str, str] | None = None,
    *,
    base: str = BACKEND_BASE,
) -> Response:
    url = f"{base}/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    if body is None:
        body = await request.body()
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            upstream = await client.request(
                request.method,
                url,
                content=body,
                headers=_forward_headers(request),
            )
        except httpx.RequestError as e:
            raise HTTPException(502, f"backend unreachable: {e}") from e
    excluded = {"content-encoding", "transfer-encoding", "content-length", "connection"}
    out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in excluded}
    if extra_headers:
        out_headers.update(extra_headers)
    return Response(content=upstream.content, status_code=upstream.status_code, headers=out_headers)


class _SpeedTracker:
    def __init__(self, started: float, model: str | None, engine: str | None) -> None:
        self.started = started
        self.model = model
        self.engine = engine
        self.buffer = ""
        self.first_content: float | None = None
        self.last_content: float | None = None
        self.content_chunks = 0
        self.server_tps: float | None = None
        self.server_tokens: int | None = None
        self.text: list[str] = []   # reply content, for the memory pass

    def _event(self, data: str) -> None:
        if not data or data == "[DONE]":
            return
        try:
            obj = json.loads(data)
        except (TypeError, ValueError):
            return
        usage = obj.get("usage") or {}
        if usage.get("completion_tokens") is not None:
            try:
                self.server_tokens = int(usage["completion_tokens"])
            except (TypeError, ValueError):
                pass
        timings = obj.get("timings") or {}
        if timings.get("predicted_per_second") is not None:
            try:
                self.server_tps = float(timings["predicted_per_second"])
            except (TypeError, ValueError):
                pass
        if timings.get("predicted_n") is not None:
            try:
                self.server_tokens = int(timings["predicted_n"])
            except (TypeError, ValueError):
                pass
        delta = (obj.get("choices") or [{}])[0].get("delta") or {}
        piece = delta.get("content")
        if isinstance(piece, str) and piece:
            now = time.perf_counter()
            if self.first_content is None:
                self.first_content = now
            self.last_content = now
            self.content_chunks += 1
            if sum(map(len, self.text)) < 20000:
                self.text.append(piece)

    @property
    def reply_text(self) -> str:
        return "".join(self.text)

    def feed(self, chunk: bytes) -> None:
        self.buffer += chunk.decode("utf-8", errors="replace")
        lines = self.buffer.splitlines()
        self.buffer = "" if self.buffer.endswith(("\n", "\r")) else lines.pop() if lines else self.buffer
        for line in lines:
            if line.startswith("data:"):
                self._event(line[5:].strip())

    def finish(self) -> None:
        if self.buffer.startswith("data:"):
            self._event(self.buffer[5:].strip())
        finished = time.time()
        end = self.last_content or time.perf_counter()
        first = self.first_content or end
        tokens = self.server_tokens if self.server_tokens is not None else self.content_chunks
        tps = self.server_tps
        approx = tps is None
        if tps is None:
            tps = tokens / max(end - first, 0.001) if tokens else None
        note_speed({
            "model": self.model,
            "engine": self.engine,
            "ttft_s": None if self.first_content is None else round(
                self.first_content - self.started, 4
            ),
            "tok_s": None if tps is None else round(tps, 3),
            "tokens": tokens,
            "approx": approx,
            "finished_at": finished,
        })


def _record_buffered_speed(response: Response, started: float,
                           model: str | None, engine: str | None) -> None:
    try:
        body = json.loads(response.body)
    except (TypeError, ValueError, UnicodeDecodeError):
        return
    usage = body.get("usage") or {}
    timings = body.get("timings") or {}
    try:
        tokens = int(timings.get("predicted_n", usage.get("completion_tokens", 0)))
    except (TypeError, ValueError):
        tokens = 0
    try:
        tps = float(timings["predicted_per_second"])
    except (KeyError, TypeError, ValueError):
        tps = None
    note_speed({
        "model": model,
        "engine": engine,
        "ttft_s": round(time.perf_counter() - started, 4),
        "tok_s": None if tps is None else round(tps, 3),
        "tokens": tokens,
        "approx": tps is None,
        "finished_at": time.time(),
    })


async def _proxy_stream(
    request: Request,
    path: str,
    body: bytes,
    extra_headers: dict[str, str] | None = None,
    on_close: Callable[[], None] | None = None,
    speed_started: float | None = None,
    speed_model: str | None = None,
    speed_engine: str | None = None,
    on_reply: Callable[[str], None] | None = None,
) -> StreamingResponse:
    """Pass through SSE / chunked bodies without buffering the full reply.

    ``on_close`` runs once when the stream ends (or fails to start), so the
    caller can hold the backend gate for the whole generation.
    """
    url = f"{BACKEND_BASE}/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    client = httpx.AsyncClient(timeout=_TIMEOUT)

    try:
        req = client.build_request(
            request.method,
            url,
            content=body,
            headers=_forward_headers(request),
        )
        upstream = await client.send(req, stream=True)
    except httpx.RequestError as e:
        await client.aclose()
        if on_close:
            on_close()
        raise HTTPException(502, f"backend unreachable: {e}") from e

    media = upstream.headers.get("content-type", "text/event-stream")

    tracker = (
        _SpeedTracker(speed_started, speed_model, speed_engine)
        if speed_started is not None else None
    )

    async def byte_iter():
        try:
            async for chunk in upstream.aiter_raw():
                if chunk:
                    # Yield first: parsing metrics must never delay bytes reaching
                    # the client.
                    yield chunk
                    if tracker:
                        tracker.feed(chunk)
        finally:
            await upstream.aclose()
            await client.aclose()
            if tracker:
                tracker.finish()
            if on_close:
                on_close()
            if on_reply and tracker:
                on_reply(tracker.reply_text)

    out_headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Access-Control-Expose-Headers": _HAIE_EXPOSE,
    }
    if extra_headers:
        out_headers.update(extra_headers)
    return StreamingResponse(
        byte_iter(),
        status_code=upstream.status_code,
        media_type=media,
        headers=out_headers,
    )


# --------------------------------------------------------------------------- #
# Retrieval side services (must be registered before the /v1 catch-all)
# --------------------------------------------------------------------------- #

async def _aux_proxy(kind: str, request: Request, path: str) -> Response:
    svc = aux_services.SERVICES[kind]
    raw = await request.body()
    try:
        await asyncio.to_thread(svc.ensure)
    except RuntimeError as e:
        raise HTTPException(503, f"{kind} service unavailable: {e}") from e
    svc.enter()
    try:
        return await _proxy_buffered(request, path, raw, base=svc.base)
    finally:
        svc.leave()


@app.post("/v1/embeddings")
async def embeddings_proxy(request: Request):
    return await _aux_proxy(EMBEDDING, request, "v1/embeddings")


@app.post("/v1/rerank")
@app.post("/v1/reranking")
@app.post("/rerank")
async def rerank_proxy(request: Request):
    return await _aux_proxy(RERANKER, request, "v1/rerank")




# --------------------------------------------------------------------------- #
# Personal memory, mood and character around each HIVI chat (MEM / SOUL)
# --------------------------------------------------------------------------- #

def _last_user_text(body: dict) -> str:
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return ""
    for m in reversed(msgs):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, list):
            return " ".join(str(p.get("text") or "") for p in c if isinstance(p, dict)).strip()
        return str(c or "").strip()
    return ""


def _chat_context(request: Request) -> dict[str, Any]:
    h = request.headers
    return {
        "hivi_chat": h.get("x-hivi-chat") is not None,
        "chat_id": (h.get("x-hivi-chat") or "").strip()[:64] or None,
        "title": (h.get("x-hivi-chat-title") or "").strip()[:120] or None,
        "temporary": h.get("x-hivi-temporary", "").strip() in ("1", "true", "yes"),
        "internal": bool(h.get("x-hivi-internal")),
    }


def _command_reply(kind: str, result: dict[str, Any], user_name: str) -> str:
    if kind == "remember":
        status = result.get("status")
        if status == "added":
            return "Noted, I'll remember that."
        if status == "duplicate":
            return "I already have that one."
        if status == "sensitive":
            return ("I don't keep passwords, card or ID numbers or keys, even when asked. "
                    "That one stays out of memory.")
        if status == "disabled":
            return "Memory is off right now. Turn it on in Settings and I'll keep it."
        return "I couldn't make a memory out of that. Try a plain sentence about you."
    removed = result.get("removed") or []
    if removed:
        lines = "\n".join(f"- {m['text']}" for m in removed)
        return f"Forgotten:\n{lines}"
    return "I couldn't find a memory that matches. Settings → Memory lists everything I have."


def _sse_text(text: str, model: str | None, extra: dict[str, str] | None = None) -> StreamingResponse:
    """A streamed reply HIVI writes itself (memory commands), in OpenAI's SSE shape."""
    cid = f"hivi-{int(time.time() * 1000)}"
    created = int(time.time())

    def chunk(delta: dict, finish: str | None = None) -> bytes:
        obj = {"id": cid, "object": "chat.completion.chunk", "created": created,
               "model": model or "hivi",
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return f"data: {json.dumps(obj)}\n\n".encode()

    async def gen():
        yield chunk({"role": "assistant", "content": ""})
        yield chunk({"content": text})
        yield chunk({}, "stop")
        yield b"data: [DONE]\n\n"

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
               "Access-Control-Expose-Headers": _HAIE_EXPOSE, "X-HIVI-HAIE-Path": "memory"}
    headers.update(extra or {})
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


def _json_reply(text: str, model: str | None) -> JSONResponse:
    return JSONResponse({
        "id": f"hivi-{int(time.time() * 1000)}", "object": "chat.completion",
        "created": int(time.time()), "model": model or "hivi",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }, headers={"X-HIVI-HAIE-Path": "memory"})


async def _memory_command(kind: str, fact: str, ctx: dict[str, Any]) -> dict[str, Any]:
    store = pm.store()
    if not store.setting("memory_enabled"):
        return {"status": "disabled"}
    batch = store.new_batch()
    if kind == "remember":
        status, mem = await asyncio.to_thread(
            store.add, fact, origin="manual", chat_id=ctx["chat_id"], chat_title=ctx["title"],
            batch=batch)
        event = memory_engine.MemoryEvent(batch=batch, added=[mem] if status == "added" else [])
        if event.changed and ctx["chat_id"]:
            memory_engine.EXTRACTOR.events[ctx["chat_id"]] = event
        return {"status": status, "memory": mem, "event": event.to_dict() if event.changed else None}
    removed = await asyncio.to_thread(store.forget, fact, batch=batch)
    event = memory_engine.MemoryEvent(batch=batch, removed=removed)
    if removed and ctx["chat_id"]:
        memory_engine.EXTRACTOR.events[ctx["chat_id"]] = event
    return {"status": "forgot" if removed else "none", "removed": removed,
            "event": event.to_dict() if removed else None}


async def _prepare_chat(body: dict[str, Any], ctx: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Build HIVI's real system prompt: character, loyalty, profile, recall, mood."""
    store = pm.store()
    settings = await asyncio.to_thread(store.settings)
    user_text = _last_user_text(body)
    headers: dict[str, str] = {}
    use_memory = bool(settings["memory_enabled"]) and not ctx["temporary"]
    profile = ""
    recalled = ""
    if use_memory:
        items = await asyncio.to_thread(store.all)
        chosen = pm.profile_memories(items)
        profile = pm.profile_block(chosen, settings.get("user_name") or "")
        chosen_ids = {m["id"] for m in chosen}
        hits = [m for m in await memory_engine.RECALL.for_message(user_text)
                if m["id"] not in chosen_ids]
        if hits:
            recalled = ("Also relevant to this message, from earlier chats:\n"
                        + "\n".join(f"- {m['text']}" for m in hits))
            headers["X-HIVI-Recalled"] = str(len(hits))
    elif settings.get("user_name") and not ctx["temporary"]:
        profile = pm.profile_block([], settings["user_name"])
    mood = soul.read_mood(user_text) if settings["mood_enabled"] else None
    if mood is not None and mood.label != "neutral":
        headers["X-HIVI-Mood"] = f"{mood.label} {mood.intensity}"
    client_system = next((m.get("content") for m in body.get("messages", [])
                          if isinstance(m, dict) and m.get("role") == "system"), None)
    base = client_system if isinstance(client_system, str) and client_system.strip() else soul.BASE
    system = soul.build_system_prompt(profile=profile, recalled=recalled, mood=mood,
                                      personality=bool(settings["personality"]), base=base)
    msgs = [m for m in body.get("messages", []) if not (isinstance(m, dict) and m.get("role") == "system")]
    out = dict(body)
    out["messages"] = [{"role": "system", "content": system}, *msgs]
    if use_memory:
        headers["X-HIVI-Memory"] = "on"
    elif ctx["temporary"]:
        headers["X-HIVI-Memory"] = "temporary"
    else:
        headers["X-HIVI-Memory"] = "off"
    return out, headers


def _schedule_extraction(user_text: str, reply_text: str, ctx: dict[str, Any]) -> None:
    if ctx["temporary"] or not user_text or not reply_text.strip():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    settings = pm.store().settings()
    if not (settings["memory_enabled"] and settings["auto_memory"]):
        return

    async def wait_idle() -> None:
        # Let the GPU go quiet: no chat in flight, and a short pause so a follow-up
        # message (which cancels this pass) gets priority.
        await asyncio.sleep(1.5)
        for _ in range(600):
            if GATE.foreground == 0 and GATE.is_open:
                return
            await asyncio.sleep(0.5)

    async def run() -> None:
        if not await GATE.enter(30.0, background=True):
            return
        try:
            await memory_engine.EXTRACTOR.run(
                backend_base=f"http://127.0.0.1:{_own_port()}", model=STATE.model_id or "local",
                user_text=user_text, reply_text=reply_text, chat_id=ctx["chat_id"],
                chat_title=ctx["title"], user_name=settings.get("user_name") or "",
                wait_idle=wait_idle,
            )
        finally:
            GATE.leave(background=True)

    loop.call_soon(lambda: memory_engine.EXTRACTOR.schedule(run()))


_OWN_PORT = {"value": 7860}


def _own_port() -> int:
    return _OWN_PORT["value"]


@app.get("/api/memory")
async def api_memory(q: str = ""):
    store = pm.store()
    items = await asyncio.to_thread(store.all, q)
    settings = await asyncio.to_thread(store.settings)
    return {"memories": items, "count": len(items), "settings": settings,
            "categories": list(pm.CATEGORIES), "feedback": store.feedback_stats(),
            "learning": memory_engine.EXTRACTOR.busy()}


@app.post("/api/memory")
async def api_memory_add(request: Request):
    body = await request.json()
    text = str(body.get("text") or "")
    status, mem = await asyncio.to_thread(pm.store().add, text, category=body.get("category"),
                                          origin="manual")
    if status == "sensitive":
        raise HTTPException(400, "HIVI never stores passwords, card or ID numbers or keys")
    if status == "empty":
        raise HTTPException(400, "write one short sentence")
    return {"status": status, "memory": mem}


@app.patch("/api/memory/{memory_id}")
async def api_memory_update(memory_id: int, request: Request):
    body = await request.json()
    status, mem = await asyncio.to_thread(pm.store().update, memory_id, body.get("text"),
                                          body.get("category"))
    if status == "missing":
        raise HTTPException(404, "no such memory")
    if status == "sensitive":
        raise HTTPException(400, "HIVI never stores passwords, card or ID numbers or keys")
    if status == "empty":
        raise HTTPException(400, "write one short sentence")
    return {"status": status, "memory": mem}


@app.delete("/api/memory/{memory_id}")
async def api_memory_delete(memory_id: int):
    old = await asyncio.to_thread(pm.store().delete, memory_id)
    if old is None:
        raise HTTPException(404, "no such memory")
    return {"status": "deleted", "memory": old}


@app.post("/api/memory/delete-all")
async def api_memory_delete_all(request: Request):
    body = await request.json()
    if body.get("confirm") != "DELETE":
        raise HTTPException(400, "send {\"confirm\": \"DELETE\"}")
    n = await asyncio.to_thread(pm.store().delete_all)
    memory_engine.EXTRACTOR.events.clear()
    return {"status": "deleted", "count": n}


@app.post("/api/memory/undo")
async def api_memory_undo(request: Request):
    body = await request.json()
    batch = str(body.get("batch") or "")
    if not batch:
        raise HTTPException(400, "batch required")
    n = await asyncio.to_thread(pm.store().undo, batch)
    for cid, ev in list(memory_engine.EXTRACTOR.events.items()):
        if ev.batch == batch:
            del memory_engine.EXTRACTOR.events[cid]
    return {"status": "undone", "count": n}


@app.get("/api/memory/events")
async def api_memory_events(chat: str = ""):
    ev = memory_engine.EXTRACTOR.events.get(chat) if chat else memory_engine.EXTRACTOR.last_event
    return {"event": ev.to_dict() if ev else None, "learning": memory_engine.EXTRACTOR.busy()}


@app.post("/api/memory/settings")
async def api_memory_settings(request: Request):
    body = await request.json()
    return {"settings": await asyncio.to_thread(pm.store().update_settings, body)}


@app.get("/api/memory/export")
async def api_memory_export(format: str = "json"):
    store = pm.store()
    stamp = time.strftime("%Y%m%d")
    if format == "md":
        return PlainTextResponse(
            await asyncio.to_thread(store.export_markdown), media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="hivi-memory-{stamp}.md"'})
    return JSONResponse(
        await asyncio.to_thread(store.export_json),
        headers={"Content-Disposition": f'attachment; filename="hivi-memory-{stamp}.json"'})


@app.post("/api/memory/import")
async def api_memory_import(request: Request):
    raw = (await request.body()).decode("utf-8", errors="replace")
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("application/json"):
        try:
            body = json.loads(raw)
        except ValueError:
            body = {}
        if isinstance(body, dict) and isinstance(body.get("content"), str):
            raw = body["content"]
    items = pm.parse_import(raw)
    if not items:
        raise HTTPException(400, "no memories found in that file")
    counts = await asyncio.to_thread(pm.store().import_items, items)
    return {"status": "imported", **counts}


@app.post("/api/feedback")
async def api_feedback(request: Request):
    body = await request.json()
    try:
        rating = int(body.get("rating"))
    except (TypeError, ValueError) as e:
        raise HTTPException(400, "rating must be 1 or -1") from e
    fid = await asyncio.to_thread(
        pm.store().add_feedback, rating=rating, prompt=str(body.get("prompt") or ""),
        reply=str(body.get("reply") or ""), better=str(body.get("better") or ""),
        chat_id=body.get("chat_id"), model=STATE.model_id)
    return {"status": "saved", "id": fid, "stats": pm.store().feedback_stats()}


@app.get("/api/feedback/export")
async def api_feedback_export():
    rows = await asyncio.to_thread(pm.store().feedback_rows)
    lines = [json.dumps({"prompt": r["prompt"], "reply": r["reply"], "rating": r["rating"],
                         "better": r["better"], "model": r["model"], "at": r["at"]})
             for r in rows]
    return PlainTextResponse("\n".join(lines), media_type="application/jsonl",
                             headers={"Content-Disposition": 'attachment; filename="hivi-feedback.jsonl"'})


@app.delete("/api/feedback")
async def api_feedback_delete():
    return {"status": "deleted", "count": await asyncio.to_thread(pm.store().delete_feedback)}


@app.get("/api/backups")
async def api_backups():
    return {"snapshots": await asyncio.to_thread(backups.list_snapshots),
            **await asyncio.to_thread(backups.last_backup)}


@app.post("/api/backups")
async def api_backups_create(request: Request):
    body = await request.json() if int(request.headers.get("content-length") or 0) else {}
    reason = str((body or {}).get("reason") or "manual")[:24]
    try:
        snap = await asyncio.to_thread(backups.create_snapshot, reason)
    except OSError as e:
        raise HTTPException(500, f"could not write the backup: {e}") from e
    return {"status": "created", "snapshot": snap, **backups.last_backup()}


@app.post("/api/backups/restore")
async def api_backups_restore(request: Request):
    body = await request.json()
    name = str(body.get("name") or "")
    if body.get("confirm") != "RESTORE":
        raise HTTPException(400, "send {\"confirm\": \"RESTORE\"}")
    try:
        result = await asyncio.to_thread(backups.restore_snapshot, name)
    except FileNotFoundError as e:
        raise HTTPException(404, "no such snapshot") from e
    memory_engine.EXTRACTOR.events.clear()
    return {"status": "restored", **result}


@app.get("/api/chats")
async def api_chats(q: str = ""):
    st = chat_history.store()
    return {"chats": await asyncio.to_thread(st.search if q.strip() else st.list, *((q,) if q.strip() else ())),
            "count": await asyncio.to_thread(st.count)}


@app.get("/api/chats/{chat_id}")
async def api_chat_get(chat_id: str):
    chat = await asyncio.to_thread(chat_history.store().get, chat_id)
    if chat is None:
        raise HTTPException(404, "no such chat")
    return chat


@app.put("/api/chats/{chat_id}")
async def api_chat_save(chat_id: str, request: Request):
    body = await request.json()
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, "messages required")
    try:
        return await asyncio.to_thread(chat_history.store().save, chat_id, messages,
                                       model=body.get("model") or STATE.model_id, title=body.get("title"))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.patch("/api/chats/{chat_id}")
async def api_chat_patch(chat_id: str, request: Request):
    body = await request.json()
    st = chat_history.store()
    ok = True
    if "title" in body:
        ok = await asyncio.to_thread(st.rename, chat_id, str(body["title"]))
    if "pinned" in body:
        ok = await asyncio.to_thread(st.pin, chat_id, bool(body["pinned"])) and ok
    if not ok:
        raise HTTPException(404, "no such chat")
    return {"status": "ok"}


@app.delete("/api/chats/{chat_id}")
async def api_chat_delete(chat_id: str):
    if not await asyncio.to_thread(chat_history.store().delete, chat_id):
        raise HTTPException(404, "no such chat")
    return {"status": "deleted"}


@app.post("/api/chats/delete-all")
async def api_chats_delete_all(request: Request):
    body = await request.json()
    if body.get("confirm") != "DELETE":
        raise HTTPException(400, "send {\"confirm\": \"DELETE\"}")
    return {"status": "deleted", "count": await asyncio.to_thread(chat_history.store().delete_all)}


@app.get("/api/soul/mood")
async def api_soul_mood(text: str = ""):
    return soul.read_mood(text).to_dict()


# --------------------------------------------------------------------------- #
# Chat model router
# --------------------------------------------------------------------------- #

def _is_loaded(model: ModelEntry) -> bool:
    return STATE.model_id == model.id and STATE.status == "ready"


async def _enter_with_model(target: ModelEntry | None) -> None:
    """Hold the backend gate with ``target`` loaded (any model when None).

    Swaps wait for in-flight requests to drain and are serialised by
    SWITCH_LOCK, so one agent's switch never cuts off another agent's reply.
    """
    deadline = time.monotonic() + _SWITCH_WAIT_S
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise HTTPException(503, "timed out waiting for a model switch")
        if target is None or _is_loaded(target):
            if not await GATE.enter(remaining):
                raise HTTPException(503, "backend switching; request timed out waiting")
            if target is None or _is_loaded(target):
                return
            GATE.leave()  # another switch won the race; try again
            continue
        acquired = await asyncio.to_thread(SWITCH_LOCK.acquire, True, remaining)
        if not acquired:
            raise HTTPException(503, f"timed out waiting to load {target.name}")
        try:
            if _is_loaded(target):
                continue  # someone else loaded it while we waited
            GATE.close()
            t0 = time.perf_counter()
            drained = await GATE.wait_idle(_SWITCH_DRAIN_S)
            waited = time.perf_counter() - t0
            if not drained:
                print(f"HIVI: switching with {GATE.foreground} chat request(s) still running")
            note_switch_wait(waited)
            state = await asyncio.to_thread(select_model, target)
            ok, detail = state.status == "ready", state.detail
        finally:
            GATE.open()
            SWITCH_LOCK.release()
        if not ok:
            raise HTTPException(502, f"could not load {target.name}: {detail}")


def _requested_model(raw: bytes) -> str | None:
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    model = body.get("model") if isinstance(body, dict) else None
    return model if isinstance(model, str) else None


def _with_backend_model(raw: bytes, current: str | None) -> bytes:
    """Colibrì 404s on unknown model names, so send the loaded model's own id."""
    if not current:
        return raw
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        return raw
    if not isinstance(body, dict):
        return raw
    body["model"] = current
    return json.dumps(body).encode("utf-8")


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def openai_proxy(path: str, request: Request):
    # Covers both /v1/chat/completions and /v1/completions.
    is_generation = request.method == "POST" and path.endswith("completions")
    if not is_generation:
        if request.method == "GET" and path.rstrip("/") == "models" and backend_model_name():
            return await _active_models_only(request)
        return await _proxy_buffered(request, f"v1/{path}")

    speed_started = time.perf_counter()
    raw = await request.body()
    ctx = _chat_context(request)
    user_text = ""
    memory_headers: dict[str, str] = {}
    if ctx["hivi_chat"] and path.endswith("chat/completions"):
        try:
            body_obj = json.loads(raw.decode("utf-8"))
        except ValueError:
            body_obj = None
        if isinstance(body_obj, dict):
            user_text = _last_user_text(body_obj)
            command = pm.parse_command(user_text) if not ctx["temporary"] else None
            if command:
                memory_engine.EXTRACTOR.cancel()
                result = await _memory_command(command[0], command[1], ctx)
                reply = _command_reply(command[0], result, "")
                if body_obj.get("stream"):
                    return _sse_text(reply, STATE.model_id, {"X-HIVI-Memory": "command"})
                return _json_reply(reply, STATE.model_id)
            memory_engine.EXTRACTOR.cancel()   # a new message outranks learning from the last one
            body_obj, memory_headers = await _prepare_chat(body_obj, ctx)
            raw = json.dumps(body_obj).encode("utf-8")
    resolved = await asyncio.to_thread(model_router.resolve, _requested_model(raw))
    await _enter_with_model(resolved.model if resolved else None)
    released = False
    handed_off = False  # True once a streaming response owns the release

    def release() -> None:
        nonlocal released
        if not released:
            released = True
            GATE.leave()

    try:
        if resolved is not None or backend_model_name():
            # Router mode needs a real preset name even for "local" / unknown names.
            current = await asyncio.to_thread(_current_backend_model)
            raw = _with_backend_model(raw, current)
        extra: dict[str, str] = {}
        adapted = raw
        if path.endswith("chat/completions"):
            # Internal / tooling can skip HAIE with this header (semantic classify already
            # talks to BACKEND_BASE directly; this is for any HIVI-level callers).
            skip = request.headers.get("x-hivi-haie-skip", "").strip() in ("1", "true", "yes")
            if not skip:
                adapted, extra = await asyncio.to_thread(_adapt_chat_body, raw)
        if resolved is not None:
            extra["X-HIVI-Model"] = resolved.model.id
            extra["X-HIVI-Model-Via"] = resolved.via
        extra.update(memory_headers)
        on_reply = None
        if ctx["hivi_chat"] and user_text:
            def on_reply(reply_text: str, _u=user_text, _c=ctx) -> None:
                _schedule_extraction(_u, reply_text, _c)

        async def receive():
            return {"type": "http.request", "body": adapted, "more_body": False}

        request = Request(request.scope, receive)
        try:
            wants_stream = bool(json.loads(adapted.decode("utf-8")).get("stream"))
        except Exception:
            wants_stream = False
        if wants_stream:
            # The stream releases the gate when the last chunk is sent.
            response = await _proxy_stream(
                request, f"v1/{path}", adapted, extra, on_close=release,
                speed_started=speed_started,
                speed_model=STATE.name or STATE.model_id or _requested_model(raw),
                speed_engine=STATE.engine,
                on_reply=on_reply,
            )
            handed_off = True
            return response
        response = await _proxy_buffered(request, f"v1/{path}", adapted, extra)
        _record_buffered_speed(
            response,
            speed_started,
            STATE.name or STATE.model_id or _requested_model(raw),
            STATE.engine,
        )
        if on_reply:
            try:
                data = json.loads(response.body)
                on_reply(str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""))
            except (ValueError, AttributeError):
                pass
        return response
    finally:
        if not handed_off:
            release()  # idempotent: safe if the stream already released on failure


async def _active_models_only(request: Request) -> Response:
    """Router mode lists every preset in /v1/models; clients that take data[0]
    (the chat UI, many OpenAI tools) must see only the model that is active."""
    resp = await _proxy_buffered(request, "v1/models")
    active = backend_model_name()
    try:
        body = json.loads(resp.body)
        rows = [r for r in body.get("data") or [] if isinstance(r, dict) and r.get("id") == active]
        body["data"] = rows or [{"id": active, "object": "model", "owned_by": "hivi"}]
        if isinstance(body.get("models"), list):
            body["models"] = [m for m in body["models"]
                              if isinstance(m, dict) and active in (m.get("name"), m.get("model"))]
    except Exception:
        return resp
    return JSONResponse(body, status_code=resp.status_code)


def _current_backend_model() -> str | None:
    """The id the running backend advertises (model names change on every switch)."""
    routed = backend_model_name()  # router mode: /v1/models lists every preset
    if routed:
        return routed
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{BACKEND_BASE}/v1/models")
            rows = r.json().get("data") or []
            if rows and isinstance(rows[0], dict) and rows[0].get("id"):
                return str(rows[0]["id"])
    except Exception:
        return None
    return None


@app.api_route("/memory/v1/{path:path}", methods=["GET", "POST"])
async def memory_proxy(path: str, request: Request):
    """OpenAI-compatible endpoint for Hindsight (point its LLM base URL here).

    No HAIE param injection, the model name is rewritten to whatever is loaded
    now, and requests wait out model switches instead of failing. Memory calls
    are background work: a switch does not wait for them. If one is cut off by a
    switch it is sent again, unchanged, once the new model is up.
    """
    original = await request.body()
    for attempt in range(1 + _MEMORY_RETRIES):
        if not await GATE.enter(_MEMORY_WAIT_S, background=True):
            raise HTTPException(503, "backend switching; memory request timed out waiting")
        generation = GATE.generation
        try:
            raw = original
            if request.method == "POST" and raw:
                try:
                    body = json.loads(raw.decode("utf-8"))
                    if isinstance(body, dict) and "model" in body:
                        current = await asyncio.to_thread(_current_backend_model)
                        if current:
                            body["model"] = current
                        raw = json.dumps(body).encode("utf-8")
                except Exception:
                    pass
            try:
                resp = await _proxy_buffered(request, f"v1/{path}", raw)
            except HTTPException:
                if GATE.generation != generation and attempt < _MEMORY_RETRIES:
                    continue  # cut off by a switch: retry on the new model
                raise
            if resp.status_code >= 500 and GATE.generation != generation \
                    and attempt < _MEMORY_RETRIES:
                continue
            return resp
        finally:
            GATE.leave(background=True)
    raise HTTPException(503, "memory request kept being cut off by model switches")


@app.get("/health")
async def health_proxy(request: Request):
    return await _proxy_buffered(request, "health")


class _NoCacheStaticFiles(StaticFiles):
    """Static UI files (brand images, styles) with the same no-cache rule as the home page."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers.update(_NO_CACHE)
        return response


if (HERE / "static").exists():
    app.mount("/static", _NoCacheStaticFiles(directory=str(HERE / "static")), name="static")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    _OWN_PORT["value"] = args.port
    print(f"HIVI UI  http://{args.host}:{args.port}")
    print("API      /api/models  /api/select  /api/status  /api/engine")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
