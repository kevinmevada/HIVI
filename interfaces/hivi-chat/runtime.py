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
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
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
    # HIVI_THREADS wins. On hybrid Intel CPUs (P + E cores) llama.cpp is usually
    # fastest with threads = number of P-cores; E-cores stall the sync points of
    # MoE expert eval on CPU. Measure with scripts/bench_threads.ps1.
    env = os.environ.get("HIVI_THREADS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
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
    router: bool = False  # llama-server router mode: models load inside one process
    backend_model: str | None = None  # name to send as "model" (router section id)
    resident: list[str] = field(default_factory=list)  # router models, least recent first


STATE = RuntimeState()
# Models Colibrì could not import/start this session: they go straight to llama.cpp.
_COLIBRI_FAILED: dict[str, str] = {}


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
    STATE.router = False
    STATE.backend_model = None
    STATE.resident = []
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
    ]
    # Uncensored builds run with the model's own defaults: no extra penalties,
    # no forced reasoning-off. Their refusal removal is measured under the stock
    # template, and these overrides pushed them back into refusing (2026-11-11).
    if model.uncensored:
        _set_ctx(args, 16384)  # thinking on: room for reasoning + the answer
    else:
        args += ["--repeat-penalty", "1.1", "--frequency-penalty", "0.2"]
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
        ]
        # A 64-token analysis budget cuts gpt-oss off mid-deliberation; uncensored
        # builds get their full (low-effort) analysis so they reach the answer.
        if not model.uncensored:
            args += ["--reasoning-budget", "64"]
    elif not model.uncensored:
        args += ["--reasoning", "off"]
    # (uncensored non-gpt-oss: leave reasoning at the model default, i.e. thinking on)
    # MoE-friendly: keep experts in system RAM on 8GB VRAM; trim ctx for speed
    if "deepseek-coder-v2" in path_l:
        args += ["--cpu-moe", "--moe-cache-mib", "1536"]
        if not model.uncensored:
            _set_ctx(args, 4096)
    if "qwen3-coder-30b" in path_l or "30b-a3b" in path_l or "30b-a3b" in id_l:
        args += ["--cpu-moe", "--moe-cache-mib", "3072"]
        if not model.uncensored:  # thinking needs the 16K set above
            _set_ctx(args, 4096)
    return args


def build_colibri_args(
    colibri_model: Path, *, coli: Path, python: str, uncensored: bool = False
) -> list[str]:
    # Thinking is decided per request (HAIE sends enable_thinking / reasoning_effort;
    # `coli serve` ignores --no-think/--effort and reads them from each request).
    # --policy quality: keeps the GGUF's quantization and the router's exact top-K.
    return [
        python, str(coli),
        "serve",
        "--model", str(colibri_model),
        "--host", BACKEND_HOST,
        "--port", str(BACKEND_PORT),
        "--gpu", "auto",
        "--ctx", "16384" if uncensored else "8192",  # thinking + answer (HAIE caps max_tokens)
        "--policy", "quality",
        "--auto-tier",
        "--no-think",
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


def _launch_raw(args: list[str], *, cwd: Path, log: Path, engine: str, tag: str) -> None:
    logf = log.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        args, stdout=logf, stderr=subprocess.STDOUT, cwd=str(cwd), **_popen_kwargs()
    )
    STATE.proc = proc
    STATE.pid = proc.pid
    STATE.log_path = str(log)
    STATE.engine = engine
    _write_pidfile(proc.pid, engine, tag)


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
    args = build_colibri_args(
        colibri, coli=coli, python=paths.python_exe(), uncensored=model.uncensored
    )
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
    release_backend()  # also ends the llama router and every model loaded in it


def _start_single(model: ModelEntry) -> bool:
    """Old path: one backend process for one model (Colibrì, or llama without router)."""
    release_backend()
    _assert_port_free()
    if model.engine == "colibri":
        start_colibri(model)
    else:
        start_llama(model)
    return _wait_ready()


def _as_llama(model: ModelEntry, why: str) -> ModelEntry:
    return replace(model, engine="llama", engine_reason=f"llama.cpp fallback: {why}")


def select_model(model: ModelEntry) -> RuntimeState:
    """Blocking switch. Call from a worker thread while holding SWITCH_LOCK."""
    if model.engine == "colibri" and model.id in _COLIBRI_FAILED:
        model = _as_llama(model, _COLIBRI_FAILED[model.id])
    STATE.status = "starting"
    STATE.model_id = model.id
    STATE.name = model.name
    STATE.engine = model.engine
    STATE.vision = bool(model.vision_projector_path)
    STATE.detail = f"switching to {model.engine}"
    note = ""
    try:
        if model.engine == "colibri":
            try:
                ok = _start_single(model)
                why = f"Colibrì did not become ready; see {STATE.log_path}"
            except Exception as e:  # import failed, CLI missing, too old, ...
                ok, why = False, str(e)
            if ok:
                STATE.status = "ready"
                STATE.detail = f"colibri ready on {BACKEND_BASE}"
                return STATE
            # Colibrì can't run this file (unsupported quant/arch): llama.cpp can.
            print(f"HIVI: {model.name} failed on Colibrì ({why}); falling back to llama.cpp")
            _COLIBRI_FAILED[model.id] = why[:300]
            release_backend()
            model = _as_llama(model, why[:300])
            STATE.engine = "llama"
            note = " (Colibrì failed, using llama.cpp)"
        llama = paths.llama_server()
        if ROUTER_ENABLED and router_supported(llama) and _preset_safe(model):
            _select_in_router(model, llama)
        elif not _start_single(model):
            STATE.status = "error"
            STATE.detail = f"engine failed to become ready; see {STATE.log_path}"
            release_backend()
            return STATE
        STATE.status = "ready"
        where = "llama router" if STATE.router else "llama"
        STATE.detail = f"{where} ready on {BACKEND_BASE}{note}"
        return STATE
    except Exception as e:
        STATE.status = "error"
        STATE.detail = str(e)
        return STATE


# --------------------------------------------------------------------------- #
# llama.cpp router mode
#
# One llama-server stays up and loads models inside itself (POST /models/load).
# Switching between llama models then skips the process kill, the port wait and
# the restart; a model that is still resident switches instantly. The router
# unloads idle models (least recent first) when VRAM runs short
# (--models-memory-margin), but it assumes host RAM is unlimited, so HIVI
# unloads by RAM itself before each load (HIVI_RAM_RESERVE_GB kept free).
# --------------------------------------------------------------------------- #

def _env_on(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v not in ("0", "off", "false", "no")


ROUTER_ENABLED = _env_on("HIVI_LLAMA_ROUTER", True)
MODELS_MAX = max(1, int(os.environ.get("HIVI_MODELS_MAX", "3") or 3))
# MiB of VRAM the router keeps free; "off" lets models pile up on the GPU.
MEMORY_MARGIN = os.environ.get("HIVI_MODELS_MEMORY_MARGIN", "512").strip()
RAM_RESERVE_GB = float(os.environ.get("HIVI_RAM_RESERVE_GB", "8") or 8)
_LIVE = ("loaded", "loading", "sleeping")

_HELP: dict[str, str] = {}


def _llama_help(llama: Path) -> str:
    try:
        key = f"{llama}:{llama.stat().st_mtime_ns}"
    except OSError:
        return ""
    if key not in _HELP:
        try:
            r = subprocess.run([str(llama), "--help"], capture_output=True, text=True, timeout=60)
            _HELP[key] = (r.stdout or "") + (r.stderr or "")
        except Exception:
            _HELP[key] = ""
    return _HELP[key]


def router_supported(llama: Path) -> bool:
    """Older llama-server builds have no router; HIVI then restarts per model."""
    return "--models-preset" in _llama_help(llama)


def _preset_safe(model: ModelEntry) -> bool:
    # Preset values end at ';' or '#' (INI comments); such paths use the old path.
    paths_ = [model.path, model.vision_projector_path or ""]
    return not any(ch in p for p in paths_ for ch in ";#")


def _is_value(tok: str) -> bool:
    if not tok.startswith("-"):
        return True
    try:
        float(tok)
        return True
    except ValueError:
        return False


def preset_options(args: list[str]) -> list[tuple[str, str]]:
    """llama-server CLI (binary first) -> preset key/values. The router owns host/port."""
    out: list[tuple[str, str]] = []
    i = 1
    while i < len(args):
        key = args[i].lstrip("-")
        if i + 1 < len(args) and _is_value(args[i + 1]):
            value, i = args[i + 1], i + 2
        else:
            value, i = "true", i + 1
        if key in ("host", "port"):
            continue
        out.append(("model" if key == "m" else key, value))
    return out


def build_llama_preset(models: list[ModelEntry], *, llama: Path, threads: int) -> str:
    lines = [
        "; Written by HIVI (runtime.py) for llama-server router mode. Regenerated on",
        "; every switch, so edit build_llama_args instead of this file.",
        "version = 1",
        "",
    ]
    for m in sorted(models, key=lambda m: m.id):
        lines.append(f"[{m.id}]")
        for k, v in preset_options(build_llama_args(m, llama=llama, threads=threads)):
            lines.append(f"{k} = {v}")
        lines.append("stop-timeout = 5")
        lines.append("")
    return "\n".join(lines)


def build_router_args(llama: Path, preset: Path, *, margin_ok: bool) -> list[str]:
    args = [
        str(llama),
        "--host", BACKEND_HOST,
        "--port", str(BACKEND_PORT),
        "--models-preset", str(preset),
        "--models-max", str(MODELS_MAX),
        "--no-models-autoload",  # HIVI loads explicitly; a stale name never loads by surprise
    ]
    if margin_ok and MEMORY_MARGIN.lower() not in ("", "off", "0", "none"):
        args += ["--models-memory-margin", MEMORY_MARGIN]
    return args


def _preset_path() -> Path:
    d = paths.data_root() / "run"
    d.mkdir(parents=True, exist_ok=True)
    return d / "hivi_llama_models.ini"


def _router_models_wanted(model: ModelEntry) -> list[ModelEntry]:
    from registry import CHAT, scan_models

    wanted = {m.id: m for m in scan_models() if m.kind == CHAT and m.engine == "llama"
              and _preset_safe(m)}
    wanted[model.id] = model  # e.g. a Colibrì model that fell back to llama
    for mid, why in _COLIBRI_FAILED.items():
        if mid not in wanted:
            for m in scan_models():
                if m.id == mid and _preset_safe(m):
                    wanted[mid] = _as_llama(m, why)
    return list(wanted.values())


def _sync_preset(model: ModelEntry, llama: Path) -> bool:
    """Write the preset; True when it changed."""
    text = build_llama_preset(_router_models_wanted(model), llama=llama, threads=_cpu_threads())
    path = _preset_path()
    try:
        if path.read_text(encoding="utf-8") == text:
            return False
    except OSError:
        pass
    path.write_text(text, encoding="utf-8")
    return True


def _http_json(method: str, path: str, payload: dict | None = None, timeout: float = 10.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BACKEND_BASE}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8") or "null")
        except Exception:
            body = None
        return e.code, body


def _router_rows(reload: bool = False) -> list[dict]:
    code, body = _http_json("GET", "/models?reload=1" if reload else "/models")
    rows = (body or {}).get("data") if isinstance(body, dict) else None
    if code != 200 or not isinstance(rows, list):
        raise RuntimeError(f"llama router /models answered {code}")
    return [r for r in rows if isinstance(r, dict)]


def _row_status(row: dict | None) -> tuple[str, bool]:
    st = (row or {}).get("status") or {}
    if isinstance(st, str):
        return st, False
    return str(st.get("value") or "unloaded"), bool(st.get("failed"))


def _router_row(mid: str, rows: list[dict] | None = None) -> dict | None:
    for r in rows if rows is not None else _router_rows():
        if r.get("id") == mid:
            return r
    return None


def _router_alive() -> bool:
    return (
        STATE.router
        and STATE.proc is not None
        and STATE.proc.poll() is None
        and _health()
    )


def _start_router(llama: Path) -> None:
    release_backend()  # Colibrì or a single-model llama-server
    _assert_port_free()
    args = build_router_args(llama, _preset_path(),
                             margin_ok="--models-memory-margin" in _llama_help(llama))
    _launch_raw(args, cwd=llama.parent, log=_log_dir() / "hivi_llama_router.log",
                engine="llama", tag="llama-router")
    STATE.router = True
    STATE.detail = "llama router starting"
    if not _wait_ready():
        raise RuntimeError(f"llama router did not start; see {STATE.log_path}")


def _total_ram_gb() -> float | None:
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().total / 1024**3
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):  # type: ignore[attr-defined]
                return ms.ullTotalPhys / 1024**3
        except Exception:
            return None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024**2
    except Exception:
        return None
    return None


def model_footprint_gb(m: ModelEntry) -> float:
    """Weights + projector + ~1 GB for KV cache and compute buffers."""
    extra = 0.0
    if m.vision_projector_path:
        try:
            extra = Path(m.vision_projector_path).stat().st_size / 1024**3
        except OSError:
            extra = 1.0
    return float(m.size_gb or 0) + extra + 1.0


def plan_evictions(
    target: ModelEntry,
    resident: list[str],
    sizes: dict[str, float],
    budget_gb: float,
) -> list[str]:
    """Which resident models (least recent first) to unload so ``target`` fits in RAM."""
    keep = [m for m in resident if m != target.id]
    used = sum(sizes.get(m, 0.0) for m in keep)
    need = model_footprint_gb(target)
    out: list[str] = []
    for mid in keep:  # oldest first
        if used + need <= budget_gb:
            break
        out.append(mid)
        used -= sizes.get(mid, 0.0)
    return out


def _make_ram_room(model: ModelEntry, rows: list[dict]) -> None:
    from registry import scan_models

    live = [r["id"] for r in rows if _row_status(r)[0] in _LIVE and r.get("id") != model.id]
    if not live:
        return
    # Least recent first: ones HIVI doesn't know about, then HIVI's LRU order.
    order = [m for m in live if m not in STATE.resident] + [m for m in STATE.resident if m in live]
    total = _total_ram_gb() or 32.0
    sizes = {m.id: model_footprint_gb(m) for m in scan_models() if m.id in live}
    for mid in plan_evictions(model, order, sizes, total - RAM_RESERVE_GB):
        STATE.detail = f"unloading {mid} to free RAM"
        _http_json("POST", "/models/unload", {"model": mid})
        deadline = time.time() + 30
        while time.time() < deadline:
            if _row_status(_router_row(mid))[0] not in _LIVE:
                break
            time.sleep(0.25)
        if mid in STATE.resident:
            STATE.resident.remove(mid)


def _router_load(model: ModelEntry) -> None:
    status, _ = _row_status(_router_row(model.id))
    if status != "loaded":
        if status != "loading":
            # Synchronous until the instance is spawned: the router first waits for
            # the model's memory measurement and unloads idle models to make room.
            STATE.detail = f"making room for {model.name}"
            code, body = _http_json("POST", "/models/load", {"model": model.id}, timeout=300)
            msg = (body or {}).get("error") if isinstance(body, dict) else body
            if code >= 400 and "already running" not in json.dumps(msg):
                raise RuntimeError(f"llama router refused to load {model.id}: {msg}")
        STATE.detail = f"loading {model.name} in the llama router"
        t0 = time.time()
        while True:
            if STATE.proc is not None and STATE.proc.poll() is not None:
                raise RuntimeError(f"llama router exited; see {STATE.log_path}")
            status, failed = _row_status(_router_row(model.id))
            if status == "loaded":
                break
            if status == "unloaded":  # the load request returns once it is "loading"
                why = "exited during load" if failed else "was unloaded during load"
                raise RuntimeError(f"{model.name} {why}; see {STATE.log_path}")
            if time.time() - t0 > READY_TIMEOUT_S:
                raise RuntimeError(f"{model.name} did not load within {READY_TIMEOUT_S:.0f}s")
            time.sleep(0.25)
    if model.id in STATE.resident:
        STATE.resident.remove(model.id)
    STATE.resident.append(model.id)
    STATE.backend_model = model.id


def _select_in_router(model: ModelEntry, llama: Path) -> None:
    changed = _sync_preset(model, llama)
    if not _router_alive():
        _start_router(llama)
    elif changed:
        _router_rows(reload=True)  # pick up new / changed presets; others stay loaded
    STATE.engine = "llama"
    STATE.backend_model = None
    rows = _router_rows()
    _make_ram_room(model, rows)
    _router_load(model)
    STATE.resident = [m for m in STATE.resident
                      if _row_status(_router_row(m))[0] in _LIVE]  # router may have evicted


def backend_model_name() -> str | None:
    """Model name the backend expects in requests (router mode only)."""
    return STATE.backend_model if STATE.router else None


def status_dict() -> dict:
    healthy = False
    if STATE.status == "ready":
        healthy = _health()
        if healthy and STATE.router and STATE.backend_model:
            try:
                healthy = _row_status(_router_row(STATE.backend_model))[0] in ("loaded", "sleeping")
            except Exception:
                healthy = False
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
        "router": STATE.router,
        "resident": list(STATE.resident),
    }
