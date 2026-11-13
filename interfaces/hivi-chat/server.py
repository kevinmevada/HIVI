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
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import aux_services
import httpx
import model_router
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
    select_model,
    status_dict,
    stop_runtime,
)

from routing import select_model as route_model

HERE = Path(__file__).resolve().parent


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    yield
    # Side services are HIVI-owned processes; don't leave them running.
    await asyncio.to_thread(aux_services.stop_all)


app = FastAPI(title="HIVI", lifespan=_lifespan)
_TIMEOUT = httpx.Timeout(600.0, connect=30.0)
_HAIE_EXPOSE = (
    "X-HIVI-HAIE-Path, X-HIVI-HAIE-Intents, X-HIVI-HAIE-Complexity, "
    "X-HIVI-HAIE-Deliverable, X-HIVI-HAIE-Reasoning, X-HIVI-HAIE-Length, "
    "X-HIVI-HAIE-Temp, X-HIVI-HAIE-Max-Tokens, X-HIVI-HAIE-Think, "
    "X-HIVI-Adaptive-Bucket, X-HIVI-Adaptive-Temp, "
    "X-HIVI-Adaptive-Max-Tokens, X-HIVI-Adaptive-Think"
)
# How long a memory (Hindsight) request may wait for a model switch to finish,
# and how long a switch waits for in-flight memory requests before proceeding.
_MEMORY_WAIT_S = float(os.environ.get("HIVI_MEMORY_WAIT_S", "600"))
_MEMORY_DRAIN_S = float(os.environ.get("HIVI_MEMORY_DRAIN_S", "120"))
# Router-triggered switches: how long a request may wait for its model, and how
# long a switch waits for other agents' in-flight generations to finish.
_SWITCH_WAIT_S = float(os.environ.get("HIVI_SWITCH_WAIT_S", "900"))
_SWITCH_DRAIN_S = float(os.environ.get("HIVI_SWITCH_DRAIN_S", "600"))


class BackendGate:
    """Keeps memory traffic and model switches from stepping on each other.

    Hindsight shares the one local backend with chat. Without coordination a
    model switch kills the backend mid-extraction and the retain fails. With the
    gate, a switch first waits for in-flight memory calls to finish, and memory
    calls that arrive during the switch wait for the new model instead of
    hitting a dead port.
    """

    def __init__(self) -> None:
        self._open = asyncio.Event()
        self._open.set()
        self._idle = asyncio.Event()
        self._idle.set()
        self._inflight = 0

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def is_open(self) -> bool:
        return self._open.is_set()

    def close(self) -> None:
        self._open.clear()

    def open(self) -> None:
        self._open.set()

    async def wait_idle(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._idle.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def enter(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._open.wait(), timeout)
        except asyncio.TimeoutError:
            return False
        self._inflight += 1
        self._idle.clear()
        return True

    def leave(self) -> None:
        self._inflight = max(0, self._inflight - 1)
        if self._inflight == 0:
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


@app.get("/api/status")
async def api_status():
    out = await asyncio.to_thread(status_dict)
    out["memory_inflight"] = GATE.inflight
    return out


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
        drained = await GATE.wait_idle(_MEMORY_DRAIN_S)
        if not drained:
            print(f"HIVI: switching with {GATE.inflight} memory request(s) still running")
        # Model loads take minutes; keep the event loop free for /api/status etc.
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


async def _proxy_stream(
    request: Request,
    path: str,
    body: bytes,
    extra_headers: dict[str, str] | None = None,
    on_close: Callable[[], None] | None = None,
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

    async def byte_iter():
        try:
            async for chunk in upstream.aiter_raw():
                if chunk:
                    yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()
            if on_close:
                on_close()

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
            drained = await GATE.wait_idle(_SWITCH_DRAIN_S)
            if not drained:
                print(f"HIVI: switching with {GATE.inflight} request(s) still running")
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

    raw = await request.body()
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
                request, f"v1/{path}", adapted, extra, on_close=release
            )
            handed_off = True
            return response
        return await _proxy_buffered(request, f"v1/{path}", adapted, extra)
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
    now, and requests wait out model switches instead of failing.
    """
    if not await GATE.enter(_MEMORY_WAIT_S):
        raise HTTPException(503, "backend switching; memory request timed out waiting")
    try:
        raw = await request.body()
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
        return await _proxy_buffered(request, f"v1/{path}", raw)
    finally:
        GATE.leave()


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
    print(f"HIVI UI  http://{args.host}:{args.port}")
    print("API      /api/models  /api/select  /api/status  /api/engine")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
