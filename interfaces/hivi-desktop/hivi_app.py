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
import ctypes
import json
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHAT_DIR = ROOT / "interfaces" / "hivi-chat"
ICON = HERE / "hivi.ico"
TITLE = "HIVI"
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


def _post(path: str, timeout: float) -> None:
    req = urllib.request.Request(
        f"{URL}{path}", data=b"{}", method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req, timeout=timeout).read()
    except OSError as e:  # stopping is best-effort; the process tree is killed after
        log.info("POST %s: %s", path, e)


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


def stop_stale_hivi() -> None:
    """Same clean-up as scripts/hivi_ui.ps1 — only processes HIVI itself started."""
    import psutil

    me = os.getpid()
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.info["pid"] != me and _is_hivi_server(p):
            log.info("stopping a previous HIVI server (PID %s)", p.info["pid"])
            _kill_tree(p.info["pid"])
    rd = run_dir()
    for f in [*rd.glob("hivi_backend.json"), *rd.glob("hivi_aux_*.json")]:
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            pid = int(rec.get("pid") or 0)
            # Guard against PID reuse after a reboot: only stop it if it still looks like ours.
            if pid and psutil.pid_exists(pid):
                name = psutil.Process(pid).name().lower()
                if any(k in name for k in ("llama-server", "python", "coli")):
                    log.info("stopping a HIVI-owned process (PID %s, %s)", pid, name)
                    _kill_tree(pid)
        except (OSError, ValueError, psutil.Error):
            pass
        try:
            f.unlink()
        except OSError:
            pass


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


def ensure_server(owned: list, status: Callable[[str], None] = lambda _m: None) -> None:
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
    status("Starting HIVI…")
    proc = start_server()
    owned[0] = proc
    t0 = time.monotonic()
    ok = wait_ready(
        proc, START_TIMEOUT_S, tick=lambda: status(f"Starting HIVI… {int(time.monotonic() - t0)}s")
    )
    if not ok:
        code = proc.poll()
        _kill_tree(proc.pid)
        owned[0] = None
        why = f"exited with code {code}" if code is not None else "didn't answer in time"
        raise StartupError(f"HIVI {why}.\nDetails: {server_log_path()}")
    log.info("HIVI ready at %s (PID %s)", URL, proc.pid)


def stop_owned(proc: subprocess.Popen) -> None:
    log.info("stopping HIVI (PID %s)", proc.pid)
    if proc.poll() is None:
        _post("/api/stop", 45)  # model backend (waits for memory requests to finish)
        _post("/api/aux/stop", 15)  # embedding / reranker side services
    _kill_tree(proc.pid)


# --------------------------------------------------------------------------- #
# Windows niceties: one instance, window icon, message boxes
# --------------------------------------------------------------------------- #


def single_instance() -> bool:
    if not IS_WIN:
        return True
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = ctypes.c_void_p
    _MUTEX[0] = k32.CreateMutexW(None, False, "Local\\HIVI.Desktop")
    return ctypes.get_last_error() != 183  # ERROR_ALREADY_EXISTS


def set_app_identity() -> None:
    """Give Windows taskbar and recent-app entries a stable HIVI identity."""
    if not IS_WIN:
        return
    try:
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        shell32.SetCurrentProcessExplicitAppUserModelID.argtypes = [ctypes.c_wchar_p]
        shell32.SetCurrentProcessExplicitAppUserModelID.restype = ctypes.c_ulong
        shell32.SetCurrentProcessExplicitAppUserModelID("HIVI.Desktop")
    except (OSError, ctypes.ArgumentError) as e:
        log.info("app identity: %s", e)


def set_folder_icon() -> None:
    """Mark the launcher folder so Explorer uses the HIVI icon from desktop.ini."""
    if not IS_WIN or not (HERE / "desktop.ini").exists():
        return
    try:
        result = subprocess.run(
            ["attrib", "+s", str(HERE)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            log.info("folder icon: attrib returned %s: %s", result.returncode, result.stderr.strip())
        result = subprocess.run(
            ["attrib", "+h", "+s", str(HERE / "desktop.ini")],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            log.info("desktop.ini attributes: attrib returned %s: %s", result.returncode, result.stderr.strip())
    except OSError as e:
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
    """pywebview can't set the window icon on Windows, so set it directly."""
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

_PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;height:100%;background:#0B1220;color:#DCE6F2;
  font-family:"Segoe UI",system-ui,sans-serif;display:grid;place-items:center;overflow:hidden}}
.wrap{{text-align:center;max-width:520px;padding:24px}}
.orb{{width:96px;height:96px;margin:0 auto 28px;border-radius:50%;
  background:radial-gradient(circle at 38% 34%,#CCFBF1 0%,#2DD4BF 42%,#2563EB 100%);
  box-shadow:0 0 0 2px rgba(94,234,212,.35),0 0 48px rgba(45,212,191,.45);
  animation:breathe 2.4s ease-in-out infinite}}
.orb.err{{background:radial-gradient(circle at 38% 34%,#FED7AA 0%,#F97316 50%,#9A3412 100%);
  box-shadow:0 0 0 2px rgba(251,146,60,.35),0 0 40px rgba(249,115,22,.35);animation:none}}
h1{{font-weight:600;font-size:22px;letter-spacing:.06em;margin:0 0 10px}}
p{{margin:0;color:#8EA0B5;font-size:14px;line-height:1.55;white-space:pre-wrap}}
@keyframes breathe{{0%,100%{{transform:scale(1);opacity:.9}}50%{{transform:scale(1.08);opacity:1}}}}
@media (prefers-reduced-motion:reduce){{.orb{{animation:none}}}}
</style></head><body><div class="wrap"><div class="orb {cls}"></div><h1>{title}</h1>
<p id="status">{text}</p></div>
<script>function setStatus(t){{document.getElementById('status').textContent=t}}</script>
</body></html>"""


def splash_html() -> str:
    return _PAGE.format(cls="", title="HIVI", text="Waking up…")


def error_html(text: str) -> str:
    import html

    return _PAGE.format(cls="err", title="HIVI couldn't start", text=html.escape(text))


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


def have_webview() -> bool:
    try:
        import webview  # noqa: F401
    except ImportError:
        return False
    return True


def run_window(owned: list) -> None:
    import webview

    window = webview.create_window(
        TITLE,
        html=splash_html(),
        width=1280,
        height=860,
        min_size=(900, 600),
        background_color="#0B1220",
    )

    def status(msg: str) -> None:
        try:
            window.evaluate_js(f"setStatus({json.dumps(msg)})")
        except Exception as e:  # noqa: BLE001 - the window may be closing
            log.debug("status update skipped: %s", e)

    def boot() -> None:
        set_window_icon()
        try:
            ensure_server(owned, status)
        except StartupError as e:
            log.error("%s", e)
            window.load_html(error_html(str(e)))
            return
        window.load_url(URL)

    window.events.shown += set_window_icon
    # Force WebView2 on Windows: never fall back to the old Internet Explorer engine.
    gui = "edgechromium" if IS_WIN else None
    webview.start(boot, gui=gui, private_mode=False, storage_path=str(run_dir() / "webview"))


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

    if not single_instance():
        if not focus_existing():
            message("HIVI is still closing. Try again in a few seconds.")
        return 0
    if not have_webview():
        message(FIX_PYWEBVIEW)
        return 1

    set_app_identity()
    set_folder_icon()
    owned: list = [None]
    try:
        run_window(owned)
    except Exception:
        log.exception("app window failed to start")
        message(FIX_WEBVIEW2.format(log=run_dir() / "hivi_desktop.log"))
        return 1
    finally:
        if owned[0] is not None and not args.keep_running:
            stop_owned(owned[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())