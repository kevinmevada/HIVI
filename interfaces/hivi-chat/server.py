"""HIVI control + chat UI. Routes models to Colibrì or llama.cpp automatically."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from haie import plan_request
from registry import get_model, models_as_dicts
from routing import select_model as route_model
from runtime import BACKEND_BASE, STATE, select_model, status_dict, stop_runtime

HERE = Path(__file__).resolve().parent
app = FastAPI(title="HIVI")
_TIMEOUT = httpx.Timeout(600.0, connect=30.0)
_HAIE_EXPOSE = (
    "X-HIVI-HAIE-Path, X-HIVI-HAIE-Intents, X-HIVI-HAIE-Complexity, "
    "X-HIVI-HAIE-Deliverable, X-HIVI-HAIE-Reasoning, X-HIVI-HAIE-Length, "
    "X-HIVI-HAIE-Temp, X-HIVI-HAIE-Max-Tokens, X-HIVI-HAIE-Think, "
    "X-HIVI-Adaptive-Bucket, X-HIVI-Adaptive-Temp, "
    "X-HIVI-Adaptive-Max-Tokens, X-HIVI-Adaptive-Think"
)


def _adapt_chat_body(raw: bytes) -> tuple[bytes, dict[str, str]]:
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
    )
    extra = meta.to_headers()
    return json.dumps(adapted).encode("utf-8"), extra


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/models")
def api_models():
    return {"models": models_as_dicts()}


@app.post("/api/route")
async def api_route(request: Request):
    body = await request.json()
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(400, "prompt must be a non-empty string")
    decision = route_model(prompt)
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
def api_status():
    return status_dict()


@app.get("/api/download/status")
def api_download_status():
    root = (
        Path(__file__).resolve().parents[2]
        / "Models"
        / "General"
        / "NVIDIA"
        / "Nemotron-3.5-Bolt-74B-A9B"
    )
    now = time.time()
    files = []
    incomplete_bytes = 0
    completed_bytes = 0
    for path in root.rglob("*"):
        if not path.is_file() or ".cache" not in path.parts:
            continue
        size = path.stat().st_size
        is_incomplete = path.name.endswith(".incomplete")
        if is_incomplete:
            incomplete_bytes += size
        else:
            completed_bytes += size
        files.append(
            {
                "name": str(path.relative_to(root)),
                "bytes": size,
                "incomplete": is_incomplete,
                "modified": path.stat().st_mtime,
            }
        )
    active = any(now - item["modified"] < 60 for item in files if item["incomplete"])
    return {
        "repo": "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF",
        "out_dir": str(root),
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
    model = get_model(mid)
    if not model:
        raise HTTPException(404, f"unknown model id: {mid}")
    state = select_model(model)
    code = 200 if state.status == "ready" else 500
    return JSONResponse(status_dict(), status_code=code)


@app.post("/api/stop")
def api_stop():
    stop_runtime()
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
) -> Response:
    url = f"{BACKEND_BASE}/{path}"
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
) -> StreamingResponse:
    """Pass through SSE / chunked bodies without buffering the full reply."""
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


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def openai_proxy(path: str, request: Request):
    if path.endswith("chat/completions") and request.method == "POST":
        raw = await request.body()
        # Internal / tooling can skip HAIE with this header (semantic classify already
        # talks to BACKEND_BASE directly; this is for any HIVI-level callers).
        skip = request.headers.get("x-hivi-haie-skip", "").strip() in ("1", "true", "yes")
        if skip:
            adapted, extra = raw, {}
        else:
            adapted, extra = _adapt_chat_body(raw)

        async def receive():
            return {"type": "http.request", "body": adapted, "more_body": False}

        request = Request(request.scope, receive)
        wants_stream = False
        try:
            wants_stream = bool(json.loads(adapted.decode("utf-8")).get("stream"))
        except Exception:
            wants_stream = False
        if wants_stream:
            return await _proxy_stream(request, f"v1/{path}", adapted, extra)
        return await _proxy_buffered(request, f"v1/{path}", adapted, extra)
    return await _proxy_buffered(request, f"v1/{path}")


@app.get("/health")
async def health_proxy(request: Request):
    return await _proxy_buffered(request, "health")


if (HERE / "static").exists():
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    print(f"HIVI UI  http://{args.host}:{args.port}")
    print("API      /api/models  /api/select  /api/status")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
