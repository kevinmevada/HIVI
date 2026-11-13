"""HIVI desktop app — double-click the shortcut and HIVI opens in her own window.

- If HIVI is already running, the window connects to it and leaves it running on close.
- Otherwise the app starts the HIVI server in the background (no console window), shows a
  start-up screen until it's ready, and when you close the window it stops everything it
  started — server, model backend, side services — so the GPU is free again.
  Pass --keep-running (or set HIVI_KEEP_RUNNING=1) to leave HIVI running after closing.

HIVI never opens in a browser. The window is a native app window drawn by pywebview with
Microsoft's WebView2 engine (built into Windows 10 / 11) — the same approach desktop chat apps
use. Needs ``uv sync --all-extras``; if it's missing, the app says exactly what to install.

Logs: data/run/hivi_desktop.log (this app) and data/run/hivi_desktop_server.log (server).
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import inspect
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHAT_DIR = ROOT / "interfaces" / "hivi-chat"
ICON = HERE / "hivi.ico"
TITLE = "HIVI"
APP_ID = "HIVI.Desktop"  # Windows AppUserModelID: groups the window, pins and shortcuts
HOST = "127.0.0.1"
PORT = int(os.environ.get("HIVI_PORT", "7860") or 7860)
URL = f"http://{HOST}:{PORT}"
IS_WIN = sys.platform == "win32"
START_TIMEOUT_S = float(os.environ.get("HIVI_DESKTOP_START_TIMEOUT_S", "120") or 120)

log = logging.getLogger("hivi.desktop")
_MUTEX: list = [None]


class StartupError(RuntimeError):
    """HIVI could not be started; the message is shown to the user as-is."""


# --------------------------------------------------------------------------- #
# Paths and small helpers
# --------------------------------------------------------------------------- #


def data_root() -> Path:
    env = os.environ.get("HIVI_DATA_ROOT")
    return Path(env) if env else ROOT / "data"


def run_dir() -> Path:
    d = data_root() / "run"
    d.mkdir(parents=True, exist_ok=True)
    return d


def server_log_path() -> Path:
    return run_dir() / "hivi_desktop_server.log"


def python_exe() -> str:
    """Python for the server: the project's venv, never pythonw (it has no stdout)."""
    venv = ROOT / ".venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")
    if venv.exists():
        return str(venv)
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


def hivi_running(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/api/status", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return isinstance(data, dict) and "status" in data
    except (OSError, ValueError):  # not listening, timed out, or not HIVI's JSON
        return False


def port_in_use() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, PORT)) == 0


def _kill_tree(pid: int) -> None:
    import psutil

    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return
    try:
        procs = parent.children(recursive=True) + [parent]
    except psutil.Error:
        return
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(procs, timeout=8)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass


# --------------------------------------------------------------------------- #
# Starting and stopping the server
# --------------------------------------------------------------------------- #


def _is_hivi_server(p) -> bool:
    import psutil

    try:
        cmd = " ".join(p.info.get("cmdline") or [])
        if "server.py" not in cmd:
            return False
        if str(CHAT_DIR).lower() in cmd.lower():
            return True
        return Path(p.cwd()).resolve() == CHAT_DIR.resolve()  # started as `python server.py`
    except (psutil.Error, OSError):
        return False


def _recorded_processes() -> list:
    """Model backend and side services HIVI recorded in data/run (only if they still look like ours)."""
    import psutil

    procs = []
    rd = run_dir()
    for f in [*rd.glob("hivi_backend.json"), *rd.glob("hivi_aux_*.json")]:
        try:
            pid = int(json.loads(f.read_text(encoding="utf-8")).get("pid") or 0)
            # Guard against PID reuse after a reboot: only stop it if it still looks like ours.
            if pid and psutil.pid_exists(pid):
                p = psutil.Process(pid)
                if any(k in p.name().lower() for k in ("llama-server", "python", "coli")):
                    procs.append(p)
        except (OSError, ValueError, psutil.Error):
            pass
        try:
            f.unlink()
        except OSError:
            pass
    return procs


def _stop_all(procs: list, grace: float = 3.0) -> None:
    """Stop processes and everything they started, all at once (not one after another)."""
    import psutil

    victims: dict[int, psutil.Process] = {}
    for p in procs:
        if p is None:
            continue
        try:
            for c in [p, *p.children(recursive=True)]:  # list children before anything dies
                victims[c.pid] = c
        except psutil.Error:
            pass
    for p in victims.values():
        try:
            p.terminate()  # on Windows this ends the process immediately
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(list(victims.values()), timeout=grace)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass


def stop_stale_hivi() -> None:
    """Same clean-up as scripts/hivi_ui.ps1 — only processes HIVI itself started."""
    import psutil

    me = os.getpid()
    stale = [
        p
        for p in psutil.process_iter(["pid", "cmdline"])
        if p.info["pid"] != me and _is_hivi_server(p)
    ]
    for p in stale:
        log.info("stopping a previous HIVI server (PID %s)", p.pid)
    _stop_all(stale + _recorded_processes())


def start_server() -> subprocess.Popen:
    logf = open(server_log_path(), "a", encoding="utf-8", errors="replace")  # noqa: SIM115
    logf.write(f"\n=== HIVI desktop start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    logf.flush()
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if IS_WIN else 0
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    cmd = [python_exe(), str(CHAT_DIR / "server.py"), "--host", HOST, "--port", str(PORT)]
    log.info("starting: %s", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        cwd=str(CHAT_DIR),
        stdout=logf,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=flags,
        env=env,
    )


def wait_ready(
    proc: subprocess.Popen, timeout: float, tick: Callable[[], None] | None = None
) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if proc.poll() is not None:
            return False
        if hivi_running(1.0):
            return True
        if tick:
            tick()
        time.sleep(0.4)
    return False


def ensure_server(
    owned: list,
    status: Callable[[str], None] = lambda _m: None,
    cancelled: Callable[[], bool] = lambda: False,
) -> None:
    """Make sure HIVI answers on URL. A server this app starts is put in owned[0]."""
    if hivi_running():
        status("Connecting to HIVI…")
        log.info("HIVI already running at %s; not owned by this window", URL)
        return
    status("Cleaning up…")
    stop_stale_hivi()
    if port_in_use():
        time.sleep(1.5)
        if port_in_use():
            raise StartupError(
                f"Port {PORT} is used by another program.\n"
                "Close it, or set HIVI_PORT to a free port."
            )
    if cancelled():
        raise StartupError("cancelled")
    status("Starting HIVI…")
    proc = start_server()
    owned[0] = proc
    t0 = time.monotonic()

    def tick() -> None:
        if cancelled():
            _raise_cancelled()
        status(f"Starting HIVI… {int(time.monotonic() - t0)}s")

    try:
        ok = wait_ready(proc, START_TIMEOUT_S, tick=tick)
        if ok and cancelled():  # came up just as the window closed
            _raise_cancelled()
    except StartupError:
        _stop_all([psutil_process(proc.pid)])
        owned[0] = None
        raise
    if not ok:
        code = proc.poll()
        _kill_tree(proc.pid)
        owned[0] = None
        why = f"exited with code {code}" if code is not None else "didn't answer in time"
        raise StartupError(f"HIVI {why}.\nDetails: {server_log_path()}")
    log.info("HIVI ready at %s (PID %s)", URL, proc.pid)


def psutil_process(pid: int):
    import psutil

    try:
        return psutil.Process(pid)
    except psutil.Error:
        return None


def _raise_cancelled() -> None:
    raise StartupError("cancelled")


def stop_owned(proc: subprocess.Popen) -> None:
    """Stop the HIVI this window started: server, model backend, side services. Takes ~1 s.

    No polite /api/stop: it waits for in-flight memory requests (up to minutes), and nothing is
    lost by ending a local server whose window is gone. The OS frees the GPU memory at once.
    """
    import psutil

    t0 = time.monotonic()
    procs = _recorded_processes()
    try:
        procs.insert(0, psutil.Process(proc.pid))
    except psutil.Error:
        pass
    _stop_all(procs)
    log.info("HIVI stopped in %.1fs", time.monotonic() - t0)


# --------------------------------------------------------------------------- #
# Windows niceties: one instance, window icon, message boxes
# --------------------------------------------------------------------------- #


def single_instance() -> bool:
    if not IS_WIN:
        return True
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = ctypes.c_void_p
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    _MUTEX[0] = k32.CreateMutexW(None, False, "Local\\HIVI.Desktop")
    if ctypes.get_last_error() != 183:  # ERROR_ALREADY_EXISTS
        return True
    k32.CloseHandle(_MUTEX[0])  # don't keep the other instance's mutex alive
    _MUTEX[0] = None
    return False


def claim_instance(wait_s: float = 15.0) -> str:
    """'ok' to start, 'focused' if a HIVI window was brought forward, 'busy' if one is stuck closing.

    A HIVI that is still closing (window gone, processes stopping) is waited for, so opening HIVI
    right after closing it just works instead of showing an error.
    """
    end = time.monotonic() + wait_s
    while True:
        if single_instance():
            return "ok"
        if focus_existing():
            return "focused"
        if time.monotonic() > end:
            return "busy"
        time.sleep(0.25)


def set_app_identity() -> None:
    """Give Windows taskbar and recent-app entries a stable HIVI identity."""
    if not IS_WIN:
        return
    try:
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        shell32.SetCurrentProcessExplicitAppUserModelID.argtypes = [ctypes.c_wchar_p]
        shell32.SetCurrentProcessExplicitAppUserModelID.restype = ctypes.c_ulong
        shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except (OSError, ctypes.ArgumentError) as e:
        log.info("app identity: %s", e)


def app_launcher() -> str:
    """pythonw.exe that runs this app: no console window, same environment."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        return str(exe)
    venv_w = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    if venv_w.exists():
        return str(venv_w)
    if exe.with_name("pythonw.exe").exists():
        return str(exe.with_name("pythonw.exe"))
    return str(exe)


def relaunch_command() -> str:
    return f'"{app_launcher()}" "{HERE / "hivi_app.py"}"'


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def parse(cls, text: str) -> _GUID:
        u = uuid.UUID(text)
        return cls(u.time_low, u.time_mid, u.time_hi_version, (ctypes.c_ubyte * 8)(*u.bytes[8:]))


class _PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", _GUID), ("pid", ctypes.c_uint32)]


class _PROPVARIANT(ctypes.Structure):  # VT_LPWSTR only; same size as the real union
    _fields_ = [
        ("vt", ctypes.c_ushort),
        ("r1", ctypes.c_ushort),
        ("r2", ctypes.c_ushort),
        ("r3", ctypes.c_ushort),
        ("pwszVal", ctypes.c_wchar_p),
        ("pad", ctypes.c_void_p),
    ]


_IID_IPROPERTYSTORE = "886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"
_FMTID_APPUSERMODEL = "9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"
_PID_RELAUNCH_COMMAND, _PID_RELAUNCH_ICON, _PID_RELAUNCH_NAME, _PID_APP_ID = 2, 3, 4, 5


def set_window_identity(hwnd: int) -> bool:
    """Tell the taskbar what a pin of this window is: HIVI's name, icon and launch command.

    Without this, pinning the open window pins pythonw.exe — Python's icon, and clicking
    it doesn't start HIVI. Uses the window's shell property store (Windows 7+).
    """
    if not IS_WIN or not hwnd:
        return False
    ole32 = ctypes.WinDLL("ole32")
    shell32 = ctypes.WinDLL("shell32")
    ole32.CoInitialize(None)  # this thread may not have COM yet; harmless if it does
    shell32.SHGetPropertyStoreForWindow.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_GUID),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    shell32.SHGetPropertyStoreForWindow.restype = ctypes.HRESULT
    store = ctypes.c_void_p()
    iid = _GUID.parse(_IID_IPROPERTYSTORE)
    shell32.SHGetPropertyStoreForWindow(
        ctypes.c_void_p(hwnd), ctypes.byref(iid), ctypes.byref(store)
    )
    vtbl = ctypes.cast(store, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    set_value = ctypes.WINFUNCTYPE(
        ctypes.HRESULT, ctypes.c_void_p, ctypes.POINTER(_PROPERTYKEY), ctypes.POINTER(_PROPVARIANT)
    )(vtbl[6])
    commit = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p)(vtbl[7])
    release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtbl[2])
    fmtid = _GUID.parse(_FMTID_APPUSERMODEL)
    try:
        # Relaunch details first, the ID last (the order Windows documents).
        for pid, value in (
            (_PID_RELAUNCH_COMMAND, relaunch_command()),
            (_PID_RELAUNCH_NAME, TITLE),
            (_PID_RELAUNCH_ICON, f"{ICON},0"),
            (_PID_APP_ID, APP_ID),
        ):
            key = _PROPERTYKEY(fmtid, pid)
            var = _PROPVARIANT(vt=31, pwszVal=value)  # VT_LPWSTR; the store copies it
            set_value(store, ctypes.byref(key), ctypes.byref(var))
        commit(store)
    finally:
        release(store)
    return True


def set_folder_icon() -> None:
    """Mark the launcher folder so Explorer uses the HIVI icon from desktop.ini (once, no consoles)."""
    if not IS_WIN or not (HERE / "desktop.ini").exists():
        return
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetFileAttributesW.restype = ctypes.c_uint32
        hidden, system = 0x2, 0x4
        for path, want in ((HERE, system), (HERE / "desktop.ini", hidden | system)):
            attrs = k32.GetFileAttributesW(str(path))
            if attrs != 0xFFFFFFFF and attrs & want != want:
                k32.SetFileAttributesW(str(path), attrs | want)
    except (OSError, AttributeError) as e:
        log.info("folder icon: %s", e)


def _find_window() -> int:
    u32 = ctypes.windll.user32
    u32.FindWindowW.restype = ctypes.c_void_p
    return u32.FindWindowW(None, TITLE) or 0


def focus_existing() -> bool:
    if not IS_WIN:
        return False
    hwnd = _find_window()
    if hwnd:
        u32 = ctypes.windll.user32
        u32.ShowWindow(ctypes.c_void_p(hwnd), 9)  # SW_RESTORE
        u32.SetForegroundWindow(ctypes.c_void_p(hwnd))
    return bool(hwnd)


def set_window_icon() -> None:
    """Backup for the window icon, in case the window was created before it was set."""
    if not IS_WIN or not ICON.exists():
        return
    try:
        u32 = ctypes.windll.user32
        u32.LoadImageW.restype = ctypes.c_void_p
        u32.SendMessageW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        hwnd = _find_window()
        if not hwnd:
            return
        for which, metric in ((1, 11), (0, 49)):  # ICON_BIG / SM_CXICON, ICON_SMALL / SM_CXSMICON
            size = u32.GetSystemMetrics(metric)
            h = u32.LoadImageW(None, str(ICON), 1, size, size, 0x10)  # IMAGE_ICON, LR_LOADFROMFILE
            if h:
                u32.SendMessageW(hwnd, 0x80, which, h)  # WM_SETICON
    except Exception as e:  # noqa: BLE001 - cosmetic, must never stop the app
        log.info("window icon: %s", e)


def message(text: str) -> None:
    log.error(text)
    if IS_WIN:
        ctypes.windll.user32.MessageBoxW(None, text, TITLE, 0x10)  # MB_ICONERROR
    else:
        print(text, file=sys.stderr)


# --------------------------------------------------------------------------- #
# Screens shown before the chat UI loads
# --------------------------------------------------------------------------- #

# Same look as HIVI's home page (interfaces/hivi-chat/index.html): Apple-style black / white
# background, the HIVI wordmark with its colour wave, and the bunny mark from the shared
# interfaces/hivi-chat/static/brand/brand.css (so both screens always match).
# Intro: the bunny pops in, then HIVI slides out to its right. Outro (hiviOutro, called when HIVI
# is ready): everything fades up and out, and the home page picks up with its own entrance.
# Placeholders use str.replace, so CSS braces stay as they are.
BRAND_DIR = CHAT_DIR / "static" / "brand"
BUNNY_PARTS = ("outline", "fur", "inner_ears", "face", "blush", "blush_highlight")

_PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,500;14..32,700&display=swap" rel="stylesheet">
<style>__BRAND_CSS__</style><style>
html,body{margin:0;height:100%}
.hv{--bg:#fff;--label:#1d1d1f;--label-2:rgba(60,60,67,.72);--label-3:rgba(60,60,67,.42);--red:#ff3b30;
  --c1:#f08a2a;--c2:#f0475a;--c3:#d4519e;--c4:#8a4fe0;--c5:#4f63ff;--sheen:rgba(255,255,255,.6);
  --halo-op:.45;--bloom-op:.3;--ease:cubic-bezier(.32,.72,0,1);
  --font:-apple-system,BlinkMacSystemFont,"SF Pro Text","Inter","Segoe UI Variable Text","Segoe UI",system-ui,sans-serif;
  --font-display:-apple-system,BlinkMacSystemFont,"SF Pro Display","Inter","Segoe UI Variable Display","Segoe UI",system-ui,sans-serif;
  height:100%;display:grid;place-items:center;background:var(--bg);color:var(--label);font-family:var(--font);
  -webkit-font-smoothing:antialiased;overflow:hidden;user-select:none}
@media (prefers-color-scheme:dark){.hv{--bg:#000;--label:#f5f5f7;--label-2:rgba(235,235,245,.66);
  --label-3:rgba(235,235,245,.36);--red:#ff453a;--c1:#ffba71;--c2:#ff6778;--c3:#f5b9ea;--c4:#bc82f3;
  --c5:#8d9fff;--sheen:rgba(255,255,255,.7);--halo-op:.6;--bloom-op:.5}}
.col{display:flex;flex-direction:column;align-items:center;text-align:center;padding:24px;max-width:560px}
.lockup{display:flex;align-items:center;font-size:78px}
.lockup .hb{font-size:1.18em;animation:hb-wave var(--hb-speed) linear infinite,hb-in .75s cubic-bezier(.34,1.56,.64,1) both}
.wrap{display:block;max-width:0;animation:grow .8s .62s var(--ease) forwards}
.wordmark{--spectrum-wave:repeating-linear-gradient(90deg,var(--c1) calc(var(--p) + 0%),var(--c2) calc(var(--p) + 25%),
  var(--c3) calc(var(--p) + 50%),var(--c4) calc(var(--p) + 75%),var(--c5) calc(var(--p) + 100%),
  var(--c4) calc(var(--p) + 125%),var(--c3) calc(var(--p) + 150%),var(--c2) calc(var(--p) + 175%),
  var(--c1) calc(var(--p) + 200%));
  display:block;width:max-content;position:relative;isolation:isolate;margin-left:.22em;font-family:var(--font-display);
  font-weight:700;font-size:1em;line-height:1;letter-spacing:-.045em;padding:0 .05em .06em;color:transparent;
  -webkit-text-fill-color:transparent;background-image:repeating-linear-gradient(105deg,transparent calc(var(--p)*1.5 + 0%),
  transparent calc(var(--p)*1.5 + 38%),var(--sheen) calc(var(--p)*1.5 + 50%),transparent calc(var(--p)*1.5 + 62%),
  transparent calc(var(--p)*1.5 + 300%)),var(--spectrum-wave);-webkit-background-clip:text;background-clip:text;
  animation:wave 2s linear infinite,word-in .8s .62s var(--ease) both}
.wordmark::before,.wordmark::after{content:attr(data-text);position:absolute;inset:0;z-index:-1;padding:inherit;
  background-image:var(--spectrum-wave);-webkit-background-clip:text;background-clip:text;color:transparent;
  -webkit-text-fill-color:transparent;pointer-events:none}
.wordmark::before{filter:blur(8px) saturate(1.2);opacity:var(--halo-op)}
.wordmark::after{filter:blur(26px) saturate(1.3);opacity:var(--bloom-op);transform:scale(1.06)}
.status{margin-top:24px;font-size:15px;font-weight:500;letter-spacing:-.01em;min-height:22px;color:transparent;
  background:linear-gradient(90deg,var(--label-3) 0%,var(--label-3) 35%,var(--label) 50%,var(--label-3) 65%,
  var(--label-3) 100%);background-size:300% 100%;-webkit-background-clip:text;background-clip:text;
  animation:shimmer 1.8s linear infinite,arrive .6s 1.3s var(--ease) both}
.sub{margin-top:6px;font-size:12px;color:var(--label-3);font-variant-numeric:tabular-nums;white-space:pre-wrap;
  animation:arrive .6s 1.4s var(--ease) both}
.hv.out .lockup{animation:lock-out .55s .08s var(--ease) forwards}
.hv.out .status,.hv.out .sub{animation:fade-out .25s ease forwards}
.hv.err .lockup .hb{animation:hb-wave var(--hb-speed) linear infinite;animation-play-state:paused}
.hv.err .wrap{max-width:none;animation:none}
.hv.err .wordmark{animation:wave 2s linear infinite;animation-play-state:paused}
.hv.err .status{color:var(--red);background:none;animation:arrive .4s var(--ease) both}
.hv.err .sub{color:var(--label-2);font-size:13px;line-height:1.5;user-select:text;animation:arrive .4s .05s var(--ease) both}
@keyframes wave{to{--p:200%}}
@keyframes shimmer{from{background-position:100% 0}to{background-position:0% 0}}
@keyframes arrive{from{opacity:0;transform:translateY(8px) scale(.98)}to{opacity:1;transform:none}}
@keyframes hb-in{from{opacity:0;transform:scale(.35) rotate(-14deg)}55%{opacity:1}to{opacity:1;transform:none}}
@keyframes grow{to{max-width:var(--w,4em)}}
@keyframes word-in{from{opacity:0;transform:translateX(-14px);filter:blur(6px);clip-path:inset(-80% 100% -80% -20%)}
  to{opacity:1;transform:none;filter:none;clip-path:inset(-80% -80% -80% -20%)}}
@keyframes lock-out{to{opacity:0;transform:scale(1.05);filter:blur(6px)}}
@keyframes fade-out{to{opacity:0;transform:translateY(-4px)}}
@media (prefers-reduced-motion:reduce){.hv *,.hv *::before,.hv *::after{animation-duration:1ms!important;
  animation-delay:0s!important;animation-iteration-count:1!important}.status{color:var(--label-2)}}
</style></head><body><div class="hv __STATE__"><div class="col">
<div class="lockup">__BUNNY__<div class="wrap" id="wrap"><div class="wordmark" id="word" data-text="HIVI">HIVI</div></div></div>
<div class="status" id="status">__TEXT__</div>
<div class="sub" id="sub">__SUB__</div>
</div></div>
<script>
(function(){var wrap=document.getElementById('wrap'),word=document.getElementById('word');
 function size(){wrap.style.setProperty('--w',(word.offsetWidth+parseFloat(getComputedStyle(word).marginLeft))+'px')}
 size();if(document.fonts)document.fonts.ready.then(size);})();
var reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
window.__introDone=false;window.__outroDone=false;
setTimeout(function(){window.__introDone=true},reduce?0:1600);
function setStatus(t){document.getElementById('status').textContent=t}
function hiviOutro(){
 if(!window.__introDone){setTimeout(hiviOutro,80);return true}
 document.querySelector('.hv').classList.add('out');
 setTimeout(function(){window.__outroDone=true},reduce?0:650);return true}
</script>
</body></html>"""


def _data_uri(path: Path, mime: str = "image/png") -> str:
    try:
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
    except OSError:
        return ""


def _bunny_html() -> str:
    """The shared bunny mark, with its part masks embedded (the page isn't served from HIVI)."""
    masks = {p: _data_uri(BRAND_DIR / "parts" / f"{p}.png") for p in BUNNY_PARTS}
    if not all(masks.values()):  # brand files missing: fall back to the plain icon
        return f'<img class="hb" alt="" style="width:1.18em;height:1.18em" src="{_data_uri(ICON, "image/x-icon")}">'
    style = ";".join(f"--hb-m-{p}:url({uri})" for p, uri in masks.items())
    layers = "".join(
        f'<i class="hb-l hb-{p}{" hb-hivi" if p == "blush_highlight" else ""}"></i>'
        for p in BUNNY_PARTS
    )
    return f'<span class="hb" aria-hidden="true" style="{style}">{layers}</span>'


def _brand_css() -> str:
    try:
        return (BRAND_DIR / "brand.css").read_text(encoding="utf-8")
    except OSError:
        return ""


def _page(state: str, text: str, sub: str) -> str:
    import html

    return (
        _PAGE.replace("__BRAND_CSS__", _brand_css())
        .replace("__STATE__", state)
        .replace("__BUNNY__", _bunny_html())
        .replace("__TEXT__", html.escape(text))
        .replace("__SUB__", html.escape(sub))
    )


def splash_html() -> str:
    return _page("", "Waking up…", "Running locally on this PC")


def error_html(text: str) -> str:
    return _page("err", "HIVI couldn't start", text)


# --------------------------------------------------------------------------- #
# Front ends
# --------------------------------------------------------------------------- #


FIX_PYWEBVIEW = (
    "HIVI's app window needs pywebview.\n\n"
    "In the HIVI folder, run:\n    uv sync --all-extras\n\nthen open HIVI again."
)
FIX_WEBVIEW2 = (
    "HIVI's app window couldn't start Microsoft Edge WebView2.\n\n"
    'Install the "WebView2 Runtime" (Evergreen) from Microsoft, then open HIVI again:\n'
    "    https://developer.microsoft.com/microsoft-edge/webview2/\n\n"
    "Details: {log}"
)


def system_dark() -> bool:
    """Windows app theme, so the window opens in the same colour as the page."""
    if not IS_WIN:
        return True
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        )
        value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        return value == 0
    except OSError:
        return True


def have_webview() -> bool:
    try:
        import webview  # noqa: F401
    except ImportError:
        return False
    return True


def play_outro(window, timeout: float = 5.0) -> None:
    """Let the start-up screen finish its intro and play its outro before the home page loads."""
    try:
        window.evaluate_js("hiviOutro()")
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if window.evaluate_js("window.__outroDone === true"):
                return
            time.sleep(0.08)
    except Exception as e:  # noqa: BLE001 - never block HIVI on an animation
        log.debug("outro skipped: %s", e)


class FullscreenApi:
    """Functions the page can call (F11). Keep state private: pywebview exposes every public
    attribute to the page and walks into it. A public `window` made it walk the whole native
    window (thousands of objects, on every page load), which froze HIVI for a long time."""

    def __init__(self) -> None:
        self._window = None
        self._fullscreen = True

    def toggle_fullscreen(self) -> bool:
        if self._window is None:
            return self._fullscreen
        self._window.toggle_fullscreen()
        self._fullscreen = not self._fullscreen
        return self._fullscreen


class ServerStarter(threading.Thread):
    """Starts HIVI in the background, so the server and the window get ready at the same time."""

    def __init__(self, owned: list) -> None:
        super().__init__(daemon=True, name="hivi-start")
        self.owned = owned
        self.status = "Waking up…"
        self.error: str | None = None
        self.cancel = threading.Event()
        self.done = threading.Event()

    def run(self) -> None:
        t0 = time.monotonic()
        try:
            ensure_server(self.owned, self._set_status, self.cancel.is_set)
            log.info("HIVI ready after %.1fs", time.monotonic() - t0)
        except StartupError as e:
            self.error = str(e)
        except Exception as e:
            log.exception("start failed")
            self.error = f"HIVI couldn't start: {e}"
        finally:
            self.done.set()

    def _set_status(self, text: str) -> None:
        self.status = text


def run_window(owned: list, starter: ServerStarter | None = None) -> None:
    import webview

    t0 = time.monotonic()
    if starter is None:
        starter = ServerStarter(owned)
        starter.start()

    api = FullscreenApi()
    window = webview.create_window(
        TITLE,
        html=splash_html(),
        width=1280,
        height=860,
        fullscreen=True,
        js_api=api,
        min_size=(900, 600),
        background_color="#000000" if system_dark() else "#FFFFFF",
    )
    api._window = window

    def status(msg: str) -> None:
        try:
            window.evaluate_js(f"setStatus({json.dumps(msg)})")
        except Exception as e:  # noqa: BLE001 - the window may be closing
            log.debug("status update skipped: %s", e)

    def boot() -> None:
        set_window_icon()
        log.info("window open after %.1fs", time.monotonic() - t0)
        shown = None
        while not starter.done.is_set():
            if starter.status != shown:
                shown = starter.status
                status(shown)
            starter.done.wait(0.2)
        if starter.error:
            log.error("%s", starter.error)
            window.load_html(error_html(starter.error))
            return
        play_outro(window)
        window.load_url(URL)
        log.info("home page after %.1fs", time.monotonic() - t0)

    def on_shown() -> None:
        set_window_icon()
        try:
            set_window_identity(_find_window())
        except Exception as e:  # noqa: BLE001 - pinning metadata is cosmetic, never fatal
            log.info("taskbar identity: %s", e)

    window.events.shown += on_shown
    # Force WebView2 on Windows: never fall back to the old Internet Explorer engine.
    gui = "edgechromium" if IS_WIN else None
    opts = {"gui": gui, "private_mode": False, "storage_path": str(run_dir() / "webview")}
    if "icon" in inspect.signature(webview.start).parameters:  # pywebview >= 6.2 on Windows
        opts["icon"] = str(ICON)  # window + taskbar icon
    webview.start(boot, **opts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="HIVI desktop app")
    ap.add_argument(
        "--keep-running",
        action="store_true",
        default=os.environ.get("HIVI_KEEP_RUNNING") == "1",
        help="leave HIVI running after the window closes",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        filename=str(run_dir() / "hivi_desktop.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        encoding="utf-8",
    )

    claim = claim_instance()
    if claim != "ok":
        if claim == "busy":
            message("HIVI is still closing. Try again in a few seconds.")
        return 0
    if not have_webview():
        message(FIX_PYWEBVIEW)
        return 1

    set_app_identity()
    set_folder_icon()
    owned: list = [None]
    starter = ServerStarter(owned)
    starter.start()  # before the window: WebView2 and the server start up in parallel
    try:
        run_window(owned, starter)
    except Exception:
        log.exception("app window failed to start")
        message(FIX_WEBVIEW2.format(log=run_dir() / "hivi_desktop.log"))
        return 1
    finally:
        starter.cancel.set()  # window closed while starting: don't start or keep a server
        starter.done.wait(2)
        if owned[0] is not None and not args.keep_running:
            stop_owned(owned[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
