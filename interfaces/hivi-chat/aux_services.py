"""Embedding and reranker side services.

Retrieval models never take the main chat slot (port 8080). Each one gets its
own llama-server on its own port, started on first use and stopped after it
has been idle for a while, so a chat-model swap never unloads RAG and RAG never
pushes the chat model out of VRAM.

Defaults keep both on the CPU (``-ngl 0``): the 8 GB GPU belongs to the chat
model. Override with HIVI_AUX_GPU_LAYERS if you have VRAM to spare.

Ownership rules match runtime.py: HIVI only stops processes it started, records
their PIDs in data/run/hivi_aux_<kind>.json, and refuses a port held by anyone
else instead of killing it.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import runtime
from hivi_paths import paths
from registry import EMBEDDING, RERANKER, ModelEntry, scan_models

AUX_HOST = runtime.BACKEND_HOST
IDLE_S = float(os.environ.get("HIVI_AUX_IDLE_S", "600"))
READY_TIMEOUT_S = float(os.environ.get("HIVI_AUX_READY_TIMEOUT_S", "180"))
_REAPER_EVERY_S = 30.0

_DEFAULT_PORTS = {EMBEDDING: 8081, RERANKER: 8082}
_PORT_ENV = {EMBEDDING: "HIVI_EMBED_PORT", RERANKER: "HIVI_RERANK_PORT"}
_MODEL_ENV = {EMBEDDING: "HIVI_EMBEDDING_MODEL", RERANKER: "HIVI_RERANKER_MODEL"}


def _gpu_layers() -> str:
    return os.environ.get("HIVI_AUX_GPU_LAYERS", "0")


def _aux_threads() -> int:
    # The chat backend evaluates MoE experts on the CPU too; take half.
    return max(2, runtime._cpu_threads() // 2)


def build_aux_args(
    kind: str, model: ModelEntry, *, llama: Path, port: int, threads: int
) -> list[str]:
    """Pure command-line builder (unit tested)."""
    args = [
        str(llama),
        "-m", model.path,
        "--host", AUX_HOST,
        "--port", str(port),
        "-c", "8192",
        # Embedding / rerank inputs are processed in one ubatch: keep it ≥ input length.
        "-b", "8192",
        "-ub", "8192",
        "-np", "1",
        "-t", str(threads),
        "-ngl", _gpu_layers(),
    ]
    if kind == EMBEDDING:
        # Qwen3-Embedding uses last-token pooling.
        args += ["--embedding", "--pooling", "last"]
    elif kind == RERANKER:
        args += ["--reranking"]
    else:
        raise ValueError(f"not a side-service kind: {kind}")
    return args


class AuxService:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.port = int(os.environ.get(_PORT_ENV[kind], str(_DEFAULT_PORTS[kind])))
        self.base = f"http://{AUX_HOST}:{self.port}"
        self.proc: subprocess.Popen | None = None
        self.model: ModelEntry | None = None
        self.status = "idle"  # idle|starting|ready|error
        self.detail = ""
        self.log_path: str | None = None
        self.last_used = 0.0
        self._inflight = 0
        self._lock = threading.Lock()

    # ---- bookkeeping -------------------------------------------------------

    def _pidfile(self) -> Path:
        d = paths.data_root() / "run"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"hivi_aux_{self.kind}.json"

    def enter(self) -> None:
        with self._lock:
            self._inflight += 1
            self.last_used = time.time()

    def leave(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            self.last_used = time.time()

    def pick_model(self) -> ModelEntry | None:
        wanted = os.environ.get(_MODEL_ENV[self.kind])
        candidates = [m for m in scan_models() if m.kind == self.kind]
        if wanted:
            return next((m for m in candidates if m.id == wanted), None)
        candidates.sort(key=lambda m: (m.size_gb, m.name.lower()))
        return candidates[0] if candidates else None

    def healthy(self, timeout_s: float = 3.0) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base}/health", timeout=timeout_s) as r:
                return r.status == 200
        except Exception:
            return False

    # ---- lifecycle ---------------------------------------------------------

    def ensure(self) -> None:
        """Blocking: start the service if needed and wait until it answers."""
        with self._lock:
            if self.status == "ready" and self.proc is not None and self.proc.poll() is None:
                self.last_used = time.time()
                return
            self._stop_locked()
            model = self.pick_model()
            if model is None:
                self.status = "error"
                self.detail = f"no {self.kind} model found under {paths.models_root()}"
                raise RuntimeError(self.detail)
            owner = runtime._port_owner_pid(self.port)
            if owner is not None:
                self.status = "error"
                self.detail = (
                    f"port {self.port} is held by PID {owner}, which HIVI did not start. "
                    f"Stop it or set {_PORT_ENV[self.kind]}."
                )
                raise RuntimeError(self.detail)
            llama = paths.llama_server()
            runtime._require_file(llama, "llama-server", "HIVI_LLAMA_SERVER")
            args = build_aux_args(
                self.kind, model, llama=llama, port=self.port, threads=_aux_threads()
            )
            log = runtime._log_dir() / f"hivi_aux_{self.kind}.log"
            logf = log.open("w", encoding="utf-8")
            self.proc = subprocess.Popen(
                args, stdout=logf, stderr=subprocess.STDOUT, cwd=str(llama.parent),
                **runtime._popen_kwargs(),
            )
            self.model = model
            self.log_path = str(log)
            self.status = "starting"
            self.detail = f"llama-server {self.kind} starting"
            self._pidfile().write_text(
                json.dumps({"pid": self.proc.pid, "kind": self.kind, "model_id": model.id,
                            "port": self.port}),
                encoding="utf-8",
            )
            deadline = time.time() + READY_TIMEOUT_S
            while time.time() < deadline:
                if self.healthy():
                    self.status = "ready"
                    self.detail = f"{model.name} ready on {self.base}"
                    self.last_used = time.time()
                    _start_reaper()
                    return
                if self.proc.poll() is not None:
                    break
                time.sleep(1.0)
            self.status = "error"
            self.detail = f"{self.kind} service failed to become ready; see {self.log_path}"
            self._stop_locked()
            self.status = "error"
            raise RuntimeError(self.detail)

    def _owned_pids(self) -> set[int]:
        owned: set[int] = set()
        if self.proc is not None:
            owned.add(self.proc.pid)
        try:
            rec = json.loads(self._pidfile().read_text(encoding="utf-8"))
        except Exception:
            rec = None
        if rec and rec.get("port") == self.port and rec.get("pid"):
            pid = int(rec["pid"])
            if pid in owned or runtime._pid_looks_like_backend(pid):
                owned.add(pid)
        return owned

    def _stop_locked(self) -> None:
        for pid in self._owned_pids():
            try:
                runtime._kill_tree(pid)
            except Exception:
                pass
        if self.proc is not None:
            try:
                self.proc.wait(timeout=15)
            except Exception:
                pass
        self.proc = None
        try:
            self._pidfile().unlink()
        except FileNotFoundError:
            pass
        self.status = "idle"
        self.detail = "stopped"

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def stop_if_idle(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            if self.status != "ready" or self._inflight > 0:
                return False
            if now - self.last_used < IDLE_S:
                return False
            self._stop_locked()
            self.detail = f"stopped after {int(IDLE_S)}s idle"
            return True

    def status_dict(self) -> dict:
        return {
            "kind": self.kind,
            "status": self.status,
            "detail": self.detail,
            "model": None if self.model is None else self.model.name,
            "base": self.base,
            "pid": None if self.proc is None else self.proc.pid,
            "inflight": self._inflight,
            "log": self.log_path,
        }


SERVICES: dict[str, AuxService] = {EMBEDDING: AuxService(EMBEDDING), RERANKER: AuxService(RERANKER)}

_reaper_started = False
_reaper_lock = threading.Lock()


def _reaper() -> None:
    while True:
        time.sleep(_REAPER_EVERY_S)
        for svc in SERVICES.values():
            try:
                svc.stop_if_idle()
            except Exception:
                pass


def _start_reaper() -> None:
    global _reaper_started
    with _reaper_lock:
        if _reaper_started:
            return
        threading.Thread(target=_reaper, name="hivi-aux-reaper", daemon=True).start()
        _reaper_started = True


def stop_all() -> None:
    for svc in SERVICES.values():
        svc.stop()


def status_all() -> dict:
    return {kind: svc.status_dict() for kind, svc in SERVICES.items()}
