"""Is HIVI getting the whole GPU? Checks the things code alone can't see.

    python scripts\\gpu_check.py        (or GET /api/gpu while HIVI runs)

Checks:
  * the llama-server build really has a CUDA device (not CPU-only, not the Intel iGPU)
  * the laptop is plugged in and the GPU isn't held back (power cap, thermal, P-state)
  * how much VRAM the loaded model uses, how much is left on the table,
    and which other programs are holding NVIDIA VRAM
  * the Windows power plan

Parsing is split from probing so it can be unit tested without a GPU.
"""
from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field

from hivi_paths import paths

# --------------------------------------------------------------------------- #
# Parsing (pure)
# --------------------------------------------------------------------------- #

_DEVICE = re.compile(
    r"^\s*(?P<id>[A-Za-z]+\d+):\s*(?P<name>.+?)\s*\((?P<total>\d+)\s*MiB,\s*(?P<free>\d+)\s*MiB free\)",
)


def parse_list_devices(text: str) -> list[dict]:
    """``llama-server --list-devices`` -> [{id, backend, name, total_mib, free_mib}]."""
    out = []
    for line in text.splitlines():
        m = _DEVICE.match(line)
        if m:
            out.append({
                "id": m["id"],
                "backend": re.sub(r"\d+$", "", m["id"]).upper(),
                "name": m["name"],
                "total_mib": int(m["total"]),
                "free_mib": int(m["free"]),
            })
    return out


GPU_FIELDS = (
    "name", "memory.total", "memory.used", "utilization.gpu", "power.draw",
    "power.limit", "power.max_limit", "pstate", "clocks.sm", "clocks.max.sm",
    "temperature.gpu", "clocks_event_reasons.active",
)


def _num(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_gpu_query(text: str) -> dict | None:
    line = next((ln for ln in text.splitlines() if ln.strip()), "")
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < len(GPU_FIELDS):
        return None
    row = dict(zip(GPU_FIELDS, parts))
    return {
        "name": row["name"],
        "vram_total_mib": _num(row["memory.total"]),
        "vram_used_mib": _num(row["memory.used"]),
        "util_pct": _num(row["utilization.gpu"]),
        "power_w": _num(row["power.draw"]),
        "power_limit_w": _num(row["power.limit"]),
        "power_max_w": _num(row["power.max_limit"]),
        "pstate": row["pstate"],
        "sm_mhz": _num(row["clocks.sm"]),
        "sm_max_mhz": _num(row["clocks.max.sm"]),
        "temp_c": _num(row["temperature.gpu"]),
        "throttle_mask": row["clocks_event_reasons.active"],
    }


# NVML clock-event bits worth telling a laptop user about.
_THROTTLE_BITS = {
    0x4: "power cap",
    0x8: "hardware slowdown",
    0x20: "thermal (software)",
    0x40: "thermal (hardware)",
    0x80: "power brake (charger / battery)",
}


def throttle_reasons(mask: str | None) -> list[str]:
    try:
        value = int(str(mask), 16)
    except (TypeError, ValueError):
        return []
    return [why for bit, why in _THROTTLE_BITS.items() if value & bit]


def parse_gpu_processes(text: str) -> list[dict]:
    """Rows from the plain ``nvidia-smi`` process table (works on Windows WDDM)."""
    rows = []
    in_table = False
    for line in text.splitlines():
        if "Processes:" in line:
            in_table = True
            continue
        if not in_table:
            continue
        m = re.match(r"^\|\s+\d+\s+\S+\s+\S+\s+(\d+|N/A)\s+([CG+]+)\s+(.+?)\s+(\d+MiB|N/A)\s*\|", line)
        if m:
            rows.append({
                "pid": None if m[1] == "N/A" else int(m[1]),
                "type": m[2],
                "name": m[3].strip().replace("\\", "/").split("/")[-1],
                "mib": None if m[4] == "N/A" else int(m[4][:-3]),
            })
    return rows


# --------------------------------------------------------------------------- #
# Probing (talks to the machine)
# --------------------------------------------------------------------------- #

# Windows: no console window flashing up for each probe when HIVI runs windowless.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _run(args: list[str], timeout: float = 20.0) -> str | None:
    try:
        r = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False,
            creationflags=_NO_WINDOW,
        )
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return None


def _battery() -> dict | None:
    try:
        import psutil

        b = psutil.sensors_battery()
        return None if b is None else {"percent": b.percent, "plugged": bool(b.power_plugged)}
    except Exception:
        return None


# Windows 11 "Power mode" (Settings > System > Power & battery) is an overlay on the
# plan, so the plan name says "Balanced" even in Best performance. Read the overlay.
_POWER_MODES = {
    "ded574b5-45a0-4f42-8737-46345c09c238": "Best performance",
    "00000000-0000-0000-0000-000000000000": "Balanced",
    "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency",
}


def power_mode_name(guid: str | None) -> str | None:
    if not guid:
        return None
    return _POWER_MODES.get(guid.strip().lower(), None)


def _power_plan() -> str | None:
    if sys.platform != "win32":
        return None
    try:
        import winreg  # type: ignore[import-not-found]

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes")
        plugged = (_battery() or {}).get("plugged", True)
        name = "ActiveOverlayAcPowerScheme" if plugged else "ActiveOverlayDcPowerScheme"
        mode = power_mode_name(winreg.QueryValueEx(key, name)[0])
        if mode:
            return mode
    except Exception:
        pass
    out = _run(["powercfg", "/getactivescheme"], timeout=10)
    m = re.search(r"\((.+?)\)\s*$", (out or "").strip())
    return m[1] if m else None


def _system_snapshot() -> dict:
    try:
        import psutil

        memory = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=None)
        disk = psutil.disk_usage(str(paths.models_root()))
        return {
            "ram_used_mib": round(memory.used / 1024 / 1024, 1),
            "ram_total_mib": round(memory.total / 1024 / 1024, 1),
            "cpu_pct": round(cpu_pct, 1),
            "disk_free_gb": round(disk.free / 1024**3, 2),
        }
    except Exception:  # noqa: BLE001 - optional system probes degrade to unavailable
        return {}


@dataclass
class Report:
    llama_server: str = ""
    devices: list[dict] = field(default_factory=list)
    gpu: dict | None = None
    processes: list[dict] = field(default_factory=list)
    battery: dict | None = None
    power_plan: str | None = None
    findings: list[dict] = field(default_factory=list)  # {level: ok|warn|bad, text}

    def to_dict(self) -> dict:
        return asdict(self)


# Which programs hold VRAM changes rarely and the full nvidia-smi table is the
# slowest call, so it refreshes every few seconds, not on every live reading.
_PROCESSES_TTL = 8.0
_PROCESSES_CACHE: dict = {"at": 0.0, "value": [], "running": False}


def _gpu_query() -> dict | None:
    out = _run([
        "nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}",
        "--format=csv,noheader,nounits",
    ], timeout=5) or ""
    gpu = parse_gpu_query(out)
    if gpu is None and "clocks_event_reasons" in out:
        # Older drivers only know the pre-2023 name of the throttle field.
        legacy = [f.replace("clocks_event_reasons", "clocks_throttle_reasons") for f in GPU_FIELDS]
        gpu = parse_gpu_query(_run([
            "nvidia-smi", f"--query-gpu={','.join(legacy)}",
            "--format=csv,noheader,nounits",
        ], timeout=5) or "")
    return gpu


def _refresh_processes() -> None:
    try:
        out = _run(["nvidia-smi"], timeout=10)
        if out is not None:
            _PROCESSES_CACHE["value"] = parse_gpu_processes(out)
    finally:
        _PROCESSES_CACHE["at"] = time.monotonic()
        _PROCESSES_CACHE["running"] = False


def _gpu_processes() -> list[dict]:
    """Last known VRAM users; refreshed on a side thread so it never slows a reading."""
    if (not _PROCESSES_CACHE.get("running")
            and time.monotonic() - _PROCESSES_CACHE["at"] >= _PROCESSES_TTL):
        _PROCESSES_CACHE["running"] = True
        threading.Thread(target=_refresh_processes, name="hivi-gpu-procs", daemon=True).start()
    return _PROCESSES_CACHE["value"]


def collect_fast() -> dict:
    """Collect short-lived probes used by the status chip and panel."""
    return {
        "gpu": _gpu_query(),
        "processes": _gpu_processes(),
        "system": _system_snapshot(),
    }


def collect_slow() -> dict:
    """Collect probes that can spawn engines or touch Windows power APIs."""
    llama = paths.llama_server()
    devices_out = _run([str(llama), "--list-devices"], timeout=60) if llama.exists() else None
    return {
        "llama_server": str(llama),
        "llama_found": llama.exists(),
        "devices_ran": devices_out is not None,
        "devices": parse_list_devices(devices_out or ""),
        "battery": _battery(),
        "power_plan": _power_plan(),
    }


def collect() -> Report:
    fast = collect_fast()
    slow = collect_slow()
    rep = Report()
    rep.gpu = fast["gpu"]
    rep.processes = fast["processes"]
    rep.llama_server = slow["llama_server"]
    rep.devices = slow["devices"]
    rep.battery = slow["battery"]
    rep.power_plan = slow["power_plan"]
    rep.findings = diagnose(
        rep,
        llama_found=slow["llama_found"],
        devices_ran=slow["devices_ran"],
    )
    return rep


# --------------------------------------------------------------------------- #
# Diagnosis (pure)
# --------------------------------------------------------------------------- #

_OURS = ("llama-server", "python", "coli")


def diagnose(rep: Report, *, llama_found: bool = True, devices_ran: bool = True) -> list[dict]:
    f: list[dict] = []

    def add(level: str, text: str) -> None:
        f.append({"level": level, "text": text})

    # 1. Is llama.cpp on the NVIDIA GPU at all?
    if not llama_found:
        add("bad", f"llama-server not found at {rep.llama_server}.")
    elif not devices_ran:
        add("warn", "Could not run llama-server --list-devices (older build?).")
    else:
        cuda = [d for d in rep.devices if d["backend"] == "CUDA"]
        if cuda:
            d = cuda[0]
            add("ok", f"llama.cpp uses CUDA on {d['name']} ({d['total_mib']} MiB).")
        elif rep.devices:
            names = ", ".join(f"{d.get('id', '?')} {d.get('name', 'unknown')}" for d in rep.devices)
            add("bad", "llama-server is not a CUDA build; it sees only "
                       f"{names}. Rebuild with -DGGML_CUDA=ON or download the CUDA "
                       "Windows release. This alone can cost 2-5x speed.")
        else:
            add("bad", "llama-server sees no GPU at all (CPU-only build). Rebuild with "
                       "-DGGML_CUDA=ON or download the CUDA Windows release.")

    gpu = rep.gpu
    if gpu is None:
        add("bad", "nvidia-smi did not answer: is the NVIDIA driver installed?")
        return f

    # 2. Power: plugged in, not capped, not hot.
    if rep.battery is not None and not rep.battery["plugged"]:
        add("bad", f"On battery ({rep.battery['percent']:.0f}%). Laptop GPUs drop to a "
                   "fraction of their power on battery: plug in the charger.")
    limit, top = gpu.get("power_limit_w"), gpu.get("power_max_w")
    if limit and top and limit < 0.85 * top:
        add("warn", f"GPU power limit is {limit:.0f} W of a possible {top:.0f} W. Switch "
                    "your laptop's vendor app to Performance/Turbo mode.")
    reasons = throttle_reasons(gpu.get("throttle_mask"))
    if reasons:
        add("warn", "GPU is being held back right now: " + ", ".join(reasons) + ".")
    if (gpu.get("temp_c") or 0) >= 87:
        add("warn", f"GPU at {gpu['temp_c']:.0f} °C: raise the laptop for airflow / use a cooling pad.")
    if rep.power_plan and not re.search(r"high|ultimate|best perf", rep.power_plan, re.IGNORECASE):
        add("warn", f"Windows power mode is '{rep.power_plan}'. Set Settings -> System -> "
                    "Power & battery -> Power mode to 'Best performance' (while plugged in).")

    # 3. VRAM: who holds it, and is any left unused?
    total, used = gpu.get("vram_total_mib") or 0, gpu.get("vram_used_mib") or 0
    others = [p for p in rep.processes if not any(o in p["name"].lower() for o in _OURS)]
    hogs = sorted({p["name"] for p in others})
    if hogs:
        add("warn", "Other programs are on the NVIDIA GPU: " + ", ".join(hogs[:8]) +
                    ". Each takes VRAM from the model. In Windows Settings -> System -> "
                    "Display -> Graphics, set them (and HIVI's window, msedgewebview2.exe) "
                    "to 'Power saving' so they use the Intel GPU.")
    model_on_gpu = any(any(o in p["name"].lower() for o in _OURS) for p in rep.processes)
    if model_on_gpu and total and total - used > 1200:
        add("warn", f"{total - used:.0f} MiB of VRAM is unused with a model loaded. "
                    "Lower HIVI_FIT_TARGET_MIB (default 512) so --fit puts more of the "
                    "model on the GPU.")
    if not f or all(x["level"] == "ok" for x in f):
        add("ok", f"{gpu['name']}: {used:.0f}/{total:.0f} MiB used, {gpu.get('pstate')}, "
                  f"{gpu.get('power_w') or 0:.0f} W.")
    return f
