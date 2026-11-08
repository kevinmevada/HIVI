"""Start/stop Colibrì or llama-server for the selected HIVI model."""
from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from registry import ModelEntry


def _cpu_threads() -> int:
    n = os.cpu_count() or 8
    # Leave a couple cores for UI / OS; MoE expert eval is CPU-heavy.
    return max(4, min(16, n - 2))

ROOT = Path(r"C:\Users\mevad\Desktop\HIVI")
COLI = Path(r"C:\Users\mevad\Downloads\colibri-main\colibri-main\c\coli")
LLAMA = ROOT / "data/P1_1/runtime/llama.cpp/bin/llama-server.exe"
LOG_DIR = ROOT / "benchmarks/27b_comparison/logs"
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8080
BACKEND_BASE = f"http://{BACKEND_HOST}:{BACKEND_PORT}"


@dataclass
class RuntimeState:
    model_id: str | None = None
    engine: str | None = None
    name: str | None = None
    status: str = "idle"  # idle|starting|ready|error
    detail: str = ""
    pid: int | None = None
    proc: subprocess.Popen | None = field(default=None, repr=False)
    log_path: str | None = None


STATE = RuntimeState()


def _gpu_free_ish() -> None:
    # Best-effort stop of prior engines
    try:
        subprocess.run(
            ["python", str(COLI), "stop"],
            cwd=str(COLI.parent),
            capture_output=True,
            text=True,
            timeout=40,
        )
    except Exception:
        pass
    # kill llama-server / coli python serve on our port
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Get-CimInstance Win32_Process | "
                    "Where-Object { $_.Name -match 'llama-server' -or ($_.CommandLine -match 'coli (web|serve)') } | "
                    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; "
                    f"Get-NetTCPConnection -LocalPort {BACKEND_PORT} -ErrorAction SilentlyContinue | "
                    "ForEach-Object { if ($_.OwningProcess -gt 0) { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } }"
                ),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
        _ = out
    except Exception:
        pass
    time.sleep(2)


def _health(timeout_s: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"{BACKEND_BASE}/health", timeout=timeout_s) as r:
            return r.status == 200
    except Exception:
        try:
            with urllib.request.urlopen(f"{BACKEND_BASE}/v1/models", timeout=timeout_s) as r:
                return r.status == 200
        except Exception:
            return False


def _wait_ready(seconds: float = 300.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < seconds:
        if _health():
            return True
        if STATE.proc is not None and STATE.proc.poll() is not None:
            return False
        time.sleep(1.0)
    return False


def ensure_colibri(model: ModelEntry) -> Path:
    gguf = Path(model.path)
    colibri = Path(model.colibri_path) if model.colibri_path else gguf.with_suffix(".colibri")
    if colibri.exists():
        return colibri
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(COLI),
        "import",
        str(gguf),
        "--mode",
        "link",
        "--out",
        str(colibri),
    ]
    r = subprocess.run(cmd, cwd=str(COLI.parent), capture_output=True, text=True, timeout=300)
    if r.returncode != 0 and not colibri.exists():
        raise RuntimeError(f"coli import failed: {r.stderr or r.stdout}")
    return colibri


def start_llama(model: ModelEntry) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"hivi_llama_{model.id}.log"
    threads = str(_cpu_threads())
    args = [
        str(LLAMA),
        "-m",
        model.path,
        "--host",
        BACKEND_HOST,
        "--port",
        str(BACKEND_PORT),
        "-c",
        "8192",
        "-np",
        "1",
        "-t",
        threads,
        "-tb",
        threads,
        "-b",
        "512",
        "-ub",
        "256",
        "--fit",
        "on",
        "--fit-target",
        "512",
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
        "--jinja",
        "-fa",
        "on",
        "--repeat-penalty",
        "1.1",
        "--frequency-penalty",
        "0.2",
    ]
    path_l = model.path.lower()
    id_l = model.id.lower()
    is_gpt_oss = "gpt-oss" in path_l or "gpt-oss" in id_l
    # gpt-oss: peg-native chat parsing often drops the final answer when a
    # commentary/constrain channel is malformed. Keep raw text in content and
    # let the UI extract the final channel.
    if is_gpt_oss:
        args += [
            "--reasoning",
            "on",
            "--reasoning-format",
            "none",
            "--reasoning-effort",
            "low",
            "--reasoning-budget",
            "64",
        ]
    else:
        args += ["--reasoning", "off"]
    # MoE-friendly: keep experts in system RAM on 8GB VRAM; trim ctx for speed
    if "deepseek-coder-v2" in path_l:
        args += ["--cpu-moe", "--moe-cache-mib", "1536"]
        # override earlier -c
        for i, a in enumerate(args):
            if a == "-c" and i + 1 < len(args):
                args[i + 1] = "4096"
                break
    if "qwen3-coder-30b" in path_l or "30b-a3b" in path_l or "30b-a3b" in id_l:
        args += ["--cpu-moe", "--moe-cache-mib", "3072"]
        for i, a in enumerate(args):
            if a == "-c" and i + 1 < len(args):
                args[i + 1] = "4096"  # faster KV; raise in UI if needed later
                break
    logf = log.open("w", encoding="utf-8")
    errf = (LOG_DIR / f"hivi_llama_{model.id}.log.err").open("w", encoding="utf-8")
    proc = subprocess.Popen(args, stdout=logf, stderr=errf, cwd=str(LLAMA.parent))
    STATE.proc = proc
    STATE.pid = proc.pid
    STATE.log_path = str(log)
    STATE.engine = "llama"
    STATE.detail = "llama-server starting"


def start_colibri(model: ModelEntry) -> None:
    colibri = ensure_colibri(model)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"hivi_coli_{model.id}.log"
    args = [
        "python",
        str(COLI),
        "serve",
        "--model",
        str(colibri),
        "--host",
        BACKEND_HOST,
        "--port",
        str(BACKEND_PORT),
        "--gpu",
        "auto",
        "--ctx",
        "4096",
        "--policy",
        "experimental-fast",
        "--auto-tier",
        "--no-think",
        "--effort",
        "low",
    ]
    logf = log.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        args,
        stdout=logf,
        stderr=subprocess.STDOUT,
        cwd=str(COLI.parent),
    )
    STATE.proc = proc
    STATE.pid = proc.pid
    STATE.log_path = str(log)
    STATE.engine = "colibri"
    STATE.detail = "coli serve starting"


def stop_runtime() -> None:
    STATE.status = "idle"
    STATE.detail = "stopped"
    STATE.model_id = None
    STATE.name = None
    STATE.engine = None
    if STATE.proc and STATE.proc.poll() is None:
        try:
            STATE.proc.terminate()
            STATE.proc.wait(timeout=15)
        except Exception:
            try:
                STATE.proc.kill()
            except Exception:
                pass
    STATE.proc = None
    STATE.pid = None
    _gpu_free_ish()


def select_model(model: ModelEntry) -> RuntimeState:
    STATE.status = "starting"
    STATE.model_id = model.id
    STATE.name = model.name
    STATE.engine = model.engine
    STATE.detail = f"switching to {model.engine}"
    _gpu_free_ish()
    try:
        if model.engine == "colibri":
            start_colibri(model)
        else:
            start_llama(model)
        ok = _wait_ready(420.0)
        if not ok:
            STATE.status = "error"
            STATE.detail = f"engine failed to become ready; see {STATE.log_path}"
            return STATE
        STATE.status = "ready"
        STATE.detail = f"{model.engine} ready on {BACKEND_BASE}"
        return STATE
    except Exception as e:
        STATE.status = "error"
        STATE.detail = str(e)
        return STATE


def status_dict() -> dict:
    healthy = _health() if STATE.status == "ready" else False
    return {
        "model_id": STATE.model_id,
        "name": STATE.name,
        "engine": STATE.engine,
        "status": "ready" if healthy else STATE.status,
        "detail": STATE.detail,
        "pid": STATE.pid,
        "backend": BACKEND_BASE,
        "openai": f"{BACKEND_BASE}/v1",
        "healthy": healthy,
        "log": STATE.log_path,
    }
