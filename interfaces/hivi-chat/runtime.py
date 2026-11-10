"""Start/stop Colibrì or llama-server for the selected HIVI model.

Ownership rule: HIVI only stops processes it started itself. It records the PID
of every backend it launches (in memory and in a pidfile, so a restarted control
server still knows), and refuses to take a port that some other program holds
instead of killing whatever is listening there.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from hivi_paths import paths
from registry import ModelEntry

BACKEND_HOST = os.environ.get("HIVI_BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.environ.get("HIVI_BACKEND_PORT", "8080"))
BACKEND_BASE = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
READY_TIMEOUT_S = float(os.environ.get("HIVI_READY_TIMEOUT_S", "420"))
# Benchmarks and tuning were done on Colibrì 2.1.0; older builds lack the qwen36 path.
MIN_COLIBRI_VERSION = (2, 1, 0)

# Serialises model switches; the server rejects a second switch while one runs.
SWITCH_LOCK = threading.Lock()


def _cpu_threads() -> int:
    n = os.cpu_count() or 8
    # Leave a couple cores for UI / OS; MoE expert eval is CPU-heavy.
    return max(4, min(16, n - 2))


def _log_dir() -> Path:
    d = paths.logs_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pidfile() -> Path:
    d = paths.data_root() / "run"
    d.mkdir(parents=True, exist_ok=True)
    return d / "hivi_backend.json"


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
    vision: bool = False


STATE = RuntimeState()


# --------------------------------------------------------------------------- #
# Process ownership
# --------------------------------------------------------------------------- #

def _popen_kwargs() -> dict:
    """Start backends in their own process group so we can stop the whole tree."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            return str(pid) in out
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _pid_looks_like_backend(pid: int) -> bool:
    """Pidfile PIDs can be reused after a reboot; only claim llama/python processes."""
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10,
            ).stdout.lower()
        else:
            out = subprocess.run(
                ["ps", "-p", str(pid), "-o", "comm="],
                capture_output=True, text=True, timeout=10,
            ).stdout.lower()
    except Exception:
        return False
    return any(name in out for name in ("llama-server", "python", "coli"))


def _kill_tree(pid: int) -> None:
    """Stop a process we own and every child it spawned (Colibrì forks workers)."""
    if not _pid_alive(pid):
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True, text=True, timeout=30,
        )
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        return
    deadline = time.time() + 10
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(0.25)
    if _pid_alive(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError:
            pass


def _write_pidfile(pid: int, engine: str, model_id: str) -> None:
    _pidfile().write_text(
        json.dumps({"pid": pid, "engine": engine, "model_id": model_id, "port": BACKEND_PORT}),
        encoding="utf-8",
    )


def _read_pidfile() -> dict | None:
    try:
        return json.loads(_pidfile().read_text(encoding="utf-8"))
    except Exception:
        return None


def _clear_pidfile() -> None:
    try:
        _pidfile().unlink()
    except FileNotFoundError:
        pass


def _port_owner_pid(port: int) -> int | None:
    """PID listening on the port, or None. Best-effort; never kills anything."""
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    f"(Get-NetTCPConnection -LocalPort {port} -State Listen "
                    "-ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess",
                ],
                capture_output=True, text=True, timeout=20,
            ).stdout.strip()
        else:
            out = subprocess.run(
                ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip().splitlines()
            out = out[0] if out else ""
        return int(out) if out and out.isdigit() and int(out) > 0 else None
    except Exception:
        return None


def release_backend() -> None:
    """Stop the backend HIVI owns. Leaves every other process alone."""
    owned: set[int] = set()
    if STATE.proc is not None:
        owned.add(STATE.proc.pid)
    record = _read_pidfile()
    if record and record.get("port") == BACKEND_PORT and record.get("pid"):
        rec_pid = int(record["pid"])
        if rec_pid in owned or _pid_looks_like_backend(rec_pid):
            owned.add(rec_pid)
    was_colibri = STATE.engine == "colibri" or (record or {}).get("engine") == "colibri"
    if was_colibri and owned:
        # Colibrì's engine re-execs itself and can outlive the wrapper's process
        # tree. `coli stop --port N` is port-scoped, so it only touches our engine.
        coli = paths.colibri_cli()
        if coli.exists():
            try:
                subprocess.run(
                    [paths.python_exe(), str(coli), "stop", "--port", str(BACKEND_PORT)],
                    cwd=str(coli.parent), capture_output=True, text=True, timeout=40,
                )
            except Exception:
                pass
    for pid in owned:
        try:
            _kill_tree(pid)
        except Exception:
            pass
    if STATE.proc is not None:
        try:
            STATE.proc.wait(timeout=15)
        except Exception:
            pass
    STATE.proc = None
    STATE.pid = None
    _clear_pidfile()
    # Give the OS a moment to release the port and the GPU memory.
    deadline = time.time() + 15
    while time.time() < deadline and _port_owner_pid(BACKEND_PORT) in owned:
        time.sleep(0.5)


def _assert_port_free() -> None:
    owner = _port_owner_pid(BACKEND_PORT)
    if owner is not None:
        raise RuntimeError(
            f"port {BACKEND_PORT} is held by PID {owner}, which HIVI did not start. "
            f"Stop it yourself or set HIVI_BACKEND_PORT to another port."
        )


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

def _health(timeout_s: float = 3.0) -> bool:
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(f"{BACKEND_BASE}{path}", timeout=timeout_s) as r:
                if r.status == 200:
                    return True
        except Exception:
            continue
    return False


def _wait_ready(seconds: float = READY_TIMEOUT_S) -> bool:
    t0 = time.time()
    while time.time() - t0 < seconds:
        if _health():
            return True
        if STATE.proc is not None and STATE.proc.poll() is not None:
            return False
        time.sleep(1.0)
    return False


# --------------------------------------------------------------------------- #
# Command lines (pure functions — unit tested)
# --------------------------------------------------------------------------- #

def _set_ctx(args: list[str], ctx: int) -> None:
    i = args.index("-c")
    args[i + 1] = str(ctx)


def build_llama_args(model: ModelEntry, *, llama: Path, threads: int) -> list[str]:
    args = [
        str(llama),
        "-m", model.path,
        "--host", BACKEND_HOST,
        "--port", str(BACKEND_PORT),
        "-c", "8192",
        "-np", "1",
        "-t", str(threads),
        "-tb", str(threads),
        "-b", "512",
        "-ub", "256",
        "--fit", "on",
        "--fit-target", "512",
        "-ctk", "q8_0",
        "-ctv", "q8_0",
        "--jinja",
        "-fa", "on",
        "--repeat-penalty", "1.1",
        "--frequency-penalty", "0.2",
    ]
    if model.vision_projector_path:
        args += ["--mmproj", model.vision_projector_path]
    path_l = model.path.lower()
    id_l = model.id.lower()
    # gpt-oss: peg-native chat parsing often drops the final answer when a
    # commentary/constrain channel is malformed. Keep raw text in content and
    # let the UI extract the final channel.
    if "gpt-oss" in path_l or "gpt-oss" in id_l:
        args += [
            "--reasoning", "on",
            "--reasoning-format", "none",
            "--reasoning-effort", "low",
            "--reasoning-budget", "64",
        ]
    else:
        args += ["--reasoning", "off"]
    # MoE-friendly: keep experts in system RAM on 8GB VRAM; trim ctx for speed
    if "deepseek-coder-v2" in path_l:
        args += ["--cpu-moe", "--moe-cache-mib", "1536"]
        _set_ctx(args, 4096)
    if "qwen3-coder-30b" in path_l or "30b-a3b" in path_l or "30b-a3b" in id_l:
        args += ["--cpu-moe", "--moe-cache-mib", "3072"]
        _set_ctx(args, 4096)
    return args


def build_colibri_args(colibri_model: Path, *, coli: Path, python: str) -> list[str]:
    return [
        python, str(coli),
        "serve",
        "--model", str(colibri_model),
        "--host", BACKEND_HOST,
        "--port", str(BACKEND_PORT),
        "--gpu", "auto",
        "--ctx", "4096",
        "--policy", "experimental-fast",
        "--auto-tier",
        "--no-think",
        "--effort", "low",
    ]


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #

def _require_file(path: Path, what: str, env: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{what} not found at {path}. Set {env} or run scripts/fetch_backends.ps1.")


def colibri_version(coli: Path) -> tuple[int, ...] | None:
    """Read __version__ from c/version.py next to the coli CLI."""
    import re as _re

    try:
        text = (coli.parent / "version.py").read_text(encoding="utf-8")
    except OSError:
        return None
    m = _re.search(r"__version__\s*=\s*[\"']([0-9]+(?:\.[0-9]+)*)", text)
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def _require_colibri(coli: Path) -> None:
    _require_file(coli, "Colibrì CLI", "HIVI_COLI")
    ver = colibri_version(coli)
    if ver is not None and ver < MIN_COLIBRI_VERSION:
        want = ".".join(map(str, MIN_COLIBRI_VERSION))
        raise RuntimeError(
            f"Colibrì at {coli} is {'.'.join(map(str, ver))}; HIVI needs >= {want}. "
            "Run scripts/fetch_backends.ps1 -Only colibri -Force, or set HIVI_COLI to a newer coli."
        )


def ensure_colibri(model: ModelEntry) -> Path:
    gguf = Path(model.path)
    colibri = Path(model.colibri_path) if model.colibri_path else gguf.with_suffix(".colibri")
    if colibri.exists():
        return colibri
    coli = paths.colibri_cli()
    _require_colibri(coli)
    cmd = [paths.python_exe(), str(coli), "import", str(gguf), "--mode", "link", "--out", str(colibri)]
    r = subprocess.run(cmd, cwd=str(coli.parent), capture_output=True, text=True, timeout=300)
    if r.returncode != 0 and not colibri.exists():
        raise RuntimeError(f"coli import failed: {r.stderr or r.stdout}")
    return colibri


def _launch(args: list[str], *, cwd: Path, log: Path, engine: str, model: ModelEntry) -> None:
    logf = log.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        args, stdout=logf, stderr=subprocess.STDOUT, cwd=str(cwd), **_popen_kwargs()
    )
    STATE.proc = proc
    STATE.pid = proc.pid
    STATE.log_path = str(log)
    STATE.engine = engine
    _write_pidfile(proc.pid, engine, model.id)


def start_llama(model: ModelEntry) -> None:
    llama = paths.llama_server()
    _require_file(llama, "llama-server", "HIVI_LLAMA_SERVER")
    args = build_llama_args(model, llama=llama, threads=_cpu_threads())
    _launch(args, cwd=llama.parent, log=_log_dir() / f"hivi_llama_{model.id}.log",
            engine="llama", model=model)
    STATE.detail = "llama-server starting"


def start_colibri(model: ModelEntry) -> None:
    colibri = ensure_colibri(model)
    coli = paths.colibri_cli()
    _require_colibri(coli)
    args = build_colibri_args(colibri, coli=coli, python=paths.python_exe())
    _launch(args, cwd=coli.parent, log=_log_dir() / f"hivi_coli_{model.id}.log",
            engine="colibri", model=model)
    STATE.detail = "coli serve starting"


def stop_runtime() -> None:
    STATE.status = "idle"
    STATE.detail = "stopped"
    STATE.model_id = None
    STATE.name = None
    STATE.engine = None
    STATE.vision = False
    release_backend()


def select_model(model: ModelEntry) -> RuntimeState:
    """Blocking switch. Call from a worker thread while holding SWITCH_LOCK."""
    STATE.status = "starting"
    STATE.model_id = model.id
    STATE.name = model.name
    STATE.engine = model.engine
    STATE.vision = bool(model.vision_projector_path)
    STATE.detail = f"switching to {model.engine}"
    try:
        release_backend()
        _assert_port_free()
        if model.engine == "colibri":
            start_colibri(model)
        else:
            start_llama(model)
        if not _wait_ready():
            STATE.status = "error"
            STATE.detail = f"engine failed to become ready; see {STATE.log_path}"
            release_backend()
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
    status = STATE.status
    if status == "ready" and not healthy:
        status = "unhealthy"  # was reported as "ready" even after the engine died
    return {
        "model_id": STATE.model_id,
        "name": STATE.name,
        "engine": STATE.engine,
        "status": status,
        "detail": STATE.detail,
        "pid": STATE.pid,
        "backend": BACKEND_BASE,
        "openai": f"{BACKEND_BASE}/v1",
        "healthy": healthy,
        "log": STATE.log_path,
        "vision": STATE.vision,
        "switching": SWITCH_LOCK.locked(),
    }
