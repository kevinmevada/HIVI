"""HIVI control + chat UI. Routes models to Colibrì or llama.cpp automatically."""
from __future__ import annotations

import argparse
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from registry import get_model, models_as_dicts
from runtime import BACKEND_BASE, select_model, status_dict, stop_runtime

HERE = Path(__file__).resolve().parent
app = FastAPI(title="HIVI")


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/models")
def api_models():
    return {"models": models_as_dicts()}


@app.get("/api/status")
def api_status():
    return status_dict()


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


async def _proxy(request: Request, path: str) -> Response:
    url = f"{BACKEND_BASE}/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    body = await request.body()
    timeout = httpx.Timeout(600.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            upstream = await client.request(
                request.method,
                url,
                content=body,
                headers=headers,
            )
        except httpx.RequestError as e:
            raise HTTPException(502, f"backend unreachable: {e}") from e
    excluded = {"content-encoding", "transfer-encoding", "content-length", "connection"}
    out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in excluded}
    return Response(content=upstream.content, status_code=upstream.status_code, headers=out_headers)


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def openai_proxy(path: str, request: Request):
    return await _proxy(request, f"v1/{path}")


@app.get("/health")
async def health_proxy(request: Request):
    return await _proxy(request, "health")


# static assets if any later
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
