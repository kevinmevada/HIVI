"""Model switching through llama-server router mode, against a mock router.

The mock follows tools/server/README.md ("Using multiple models"): GET /models
lists presets with status.value, POST /models/load returns once the model is
"loading", and the instance becomes "loaded" a moment later.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import registry
import runtime


class MockRouter:
    def __init__(self, preset: Path) -> None:
        self.preset = preset
        self.status: dict[str, str] = {}
        self.loads: list[str] = []
        self.unloads: list[str] = []
        self.fail: set[str] = set()
        self._read()

    def _read(self) -> None:
        for line in self.preset.read_text(encoding="utf-8").splitlines():
            if line.startswith("[") and line.endswith("]"):
                self.status.setdefault(line[1:-1], "unloaded")

    def handler(self):
        router = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send(self, code: int, body) -> None:
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path.startswith("/health"):
                    return self._send(200, {"status": "ok"})
                if self.path.startswith("/models"):
                    if "reload=1" in self.path:
                        router._read()
                    rows = []
                    for mid, st in router.status.items():
                        status = {"value": st}
                        if mid in router.fail and st == "unloaded":
                            status["failed"] = True
                        rows.append({"id": mid, "status": status})
                    return self._send(200, {"data": rows})
                return self._send(404, {"error": "nope"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                mid = body.get("model")
                if self.path == "/models/load":
                    if mid not in router.status:
                        return self._send(404, {"error": {"message": "model is not found"}})
                    if router.status[mid] != "unloaded":
                        return self._send(400, {"error": {"message": "model is already running"}})
                    router.loads.append(mid)
                    router.status[mid] = "loading"

                    def finish():
                        time.sleep(0.3)
                        router.status[mid] = "unloaded" if mid in router.fail else "loaded"

                    threading.Thread(target=finish, daemon=True).start()
                    return self._send(200, {"success": True})
                if self.path == "/models/unload":
                    router.unloads.append(mid)
                    router.status[mid] = "unloaded"
                    return self._send(200, {"success": True})
                return self._send(404, {"error": "nope"})

        return H


class _Alive:
    pid = 12345

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    models = tmp_path / "Models"
    monkeypatch.setenv("HIVI_MODELS_ROOT", str(models))
    monkeypatch.setenv("HIVI_DATA_ROOT", str(tmp_path / "data"))
    for rel, size in (
        ("Reasoning/oss/gpt-oss-20b-abliterated-q8_0.gguf", 64),
        ("General/gemma/gemma-4-26b-a4b-it-Q5_K_M.gguf", 64),
        ("Coding/hauhau/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf", 64),
    ):
        p = models / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\0" * size)

    servers: list[ThreadingHTTPServer] = []
    state = {"router": None}

    def fake_launch(args, *, cwd, log, engine, tag):
        preset = Path(args[args.index("--models-preset") + 1])
        router = MockRouter(preset)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), router.handler())
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        monkeypatch.setattr(runtime, "BACKEND_BASE", f"http://127.0.0.1:{srv.server_address[1]}")
        runtime.STATE.proc = _Alive()
        runtime.STATE.pid = 12345
        runtime.STATE.log_path = str(log)
        runtime.STATE.engine = engine
        state["router"] = router

    def fake_release():
        for srv in servers:
            srv.shutdown()
        servers.clear()
        runtime.STATE.proc = None
        runtime.STATE.router = False
        runtime.STATE.backend_model = None
        runtime.STATE.resident = []

    monkeypatch.setattr(runtime, "_launch_raw", fake_launch)
    monkeypatch.setattr(runtime, "release_backend", fake_release)
    monkeypatch.setattr(runtime, "_assert_port_free", lambda: None)
    monkeypatch.setattr(runtime, "_llama_help", lambda llama: "--models-preset --models-memory-margin")
    monkeypatch.setattr(runtime, "ROUTER_ENABLED", True)
    monkeypatch.setattr(runtime, "_COLIBRI_FAILED", {})
    monkeypatch.setattr(runtime, "_total_ram_gb", lambda: 32.0)
    fake_release()
    yield state
    fake_release()


def _model(name_part: str) -> registry.ModelEntry:
    (m,) = [m for m in registry.scan_models() if name_part in m.path]
    return m


def test_switches_stay_in_one_router_process(env):
    oss, gemma = _model("gpt-oss"), _model("gemma")
    assert oss.engine == gemma.engine == "llama"

    st = runtime.select_model(oss)
    assert st.status == "ready", st.detail
    router = env["router"]
    assert runtime.backend_model_name() == oss.id
    assert "llama router" in st.detail

    st = runtime.select_model(gemma)
    assert st.status == "ready", st.detail
    assert env["router"] is router  # no restart
    assert runtime.backend_model_name() == gemma.id
    assert runtime.STATE.resident == [oss.id, gemma.id]

    # Back to a model that is still resident: no new load request at all.
    t0 = time.perf_counter()
    st = runtime.select_model(oss)
    assert st.status == "ready" and time.perf_counter() - t0 < 0.5
    assert router.loads == [oss.id, gemma.id]
    assert runtime.STATE.resident == [gemma.id, oss.id]
    assert runtime.status_dict()["router"] is True


def test_ram_guard_unloads_least_recent(env, monkeypatch):
    oss, gemma = _model("gpt-oss"), _model("gemma")
    runtime.select_model(oss)
    runtime.select_model(gemma)
    # Each fake model counts ~1 GB (weights + KV); leave room for one only. oss is
    # still resident and is the target, so gemma is the one that has to go.
    monkeypatch.setattr(runtime, "_total_ram_gb", lambda: runtime.RAM_RESERVE_GB + 1.5)
    st = runtime.select_model(oss)
    assert st.status == "ready"
    assert env["router"].unloads == [gemma.id]


def test_failed_load_reports_error(env):
    oss = _model("gpt-oss")
    runtime.select_model(_model("gemma"))
    env["router"].fail.add(oss.id)
    st = runtime.select_model(oss)
    assert st.status == "error" and "exited during load" in st.detail


def test_colibri_failure_falls_back_to_llama_router(env, monkeypatch):
    hauhau = _model("HauhauCS")
    assert hauhau.engine == "colibri"

    def colibri_fails(model):
        if model.engine == "colibri":
            raise RuntimeError("coli import failed: unsupported tensor type")
        return True

    real_start_single = runtime._start_single
    monkeypatch.setattr(runtime, "_start_single", colibri_fails)
    st = runtime.select_model(hauhau)
    assert st.status == "ready", st.detail
    assert st.engine == "llama" and "Colibrì failed" in st.detail
    assert runtime.backend_model_name() == hauhau.id
    # Remembered: the next switch goes straight to llama.cpp.
    monkeypatch.setattr(runtime, "_start_single", real_start_single)
    assert hauhau.id in runtime._COLIBRI_FAILED
    preset = (runtime._preset_path()).read_text(encoding="utf-8")
    assert f"[{hauhau.id}]" in preset
