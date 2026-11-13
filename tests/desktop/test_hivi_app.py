"""HIVI desktop app: start / reuse / stop logic (no GUI needed)."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "interfaces" / "hivi-desktop" / "hivi_app.py"


@pytest.fixture
def app(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("hivi_app_under_test", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setenv("HIVI_DATA_ROOT", str(tmp_path / "data"))
    port = _free_port()
    monkeypatch.setattr(mod, "PORT", port)
    monkeypatch.setattr(mod, "URL", f"http://127.0.0.1:{port}")
    return mod


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_reuses_a_running_hivi_and_does_not_own_it(app, monkeypatch):
    monkeypatch.setattr(app, "hivi_running", lambda timeout=1.5: True)
    monkeypatch.setattr(app, "start_server", lambda: pytest.fail("must not start a second server"))
    owned = [None]
    app.ensure_server(owned)
    assert owned[0] is None  # closing the window must leave it running


def test_port_held_by_another_program_is_a_clear_error(app, monkeypatch):
    monkeypatch.setattr(app, "stop_stale_hivi", lambda: None)
    monkeypatch.setattr(app.time, "sleep", lambda _s: None)
    with socket.socket() as s:
        s.bind(("127.0.0.1", app.PORT))
        s.listen(16)
        with pytest.raises(app.StartupError, match=str(app.PORT)):
            app.ensure_server([None])


def test_stale_pid_files_are_removed_without_killing_strangers(app):
    run = app.run_dir()
    (run / "hivi_backend.json").write_text(json.dumps({"pid": 2**22 + 12345}), encoding="utf-8")
    (run / "hivi_aux_embed.json").write_text("not json", encoding="utf-8")
    app.stop_stale_hivi()
    assert not list(run.glob("hivi_*.json"))


def test_server_python_prefers_the_project_venv(app, monkeypatch, tmp_path):
    venv_py = tmp_path / ".venv" / ("Scripts/python.exe" if app.IS_WIN else "bin/python")
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("")
    monkeypatch.setattr(app, "ROOT", tmp_path)
    assert app.python_exe() == str(venv_py)


def test_error_page_escapes_the_message(app):
    page = app.error_html("<script>x</script>")
    assert "<script>x</script>" not in page and "&lt;script&gt;" in page


def test_fullscreen_api_toggles_the_window(app):
    class Window:
        def __init__(self):
            self.toggles = 0

        def toggle_fullscreen(self):
            self.toggles += 1

    window = Window()
    api = app.FullscreenApi()
    api._window = window
    assert api.toggle_fullscreen() is False
    assert api.toggle_fullscreen() is True
    assert window.toggles == 2


def test_branding_points_to_the_desktop_icon(app):
    assert app.ICON.name == "hivi.ico"
    assert app.ICON.is_file()
    assert (
        (app.HERE / "desktop.ini")
        .read_text(encoding="utf-8")
        .startswith("[.ShellClassInfo]\nIconResource=hivi.ico,0")
    )


@pytest.mark.skipif(
    os.environ.get("HIVI_DESKTOP_LIVE") != "1",
    reason="starts the real server; set HIVI_DESKTOP_LIVE=1",
)
def test_starts_and_stops_the_real_server(app):
    owned = [None]
    app.ensure_server(owned)
    try:
        assert owned[0] is not None and app.hivi_running()
    finally:
        t0 = time.monotonic()
        app.stop_owned(owned[0])
        assert time.monotonic() - t0 < 5, "closing HIVI must not hang"
    assert owned[0].poll() is not None
    assert not app.hivi_running(0.5)


def test_missing_window_library_never_falls_back_to_a_browser(app, monkeypatch):
    shown = []
    monkeypatch.setattr(app, "single_instance", lambda: True)
    monkeypatch.setattr(app, "have_webview", lambda: False)
    monkeypatch.setattr(app, "message", shown.append)
    monkeypatch.setattr(app, "ensure_server", lambda *a, **k: pytest.fail("must not start HIVI"))
    assert app.main([]) == 1
    assert "uv sync --all-extras" in shown[0]
    assert not hasattr(app, "run_browser_window")


def test_pin_relaunches_hivi_not_bare_python(app):
    cmd = app.relaunch_command()
    assert cmd.endswith('hivi_app.py"') and cmd.startswith('"')
    assert str(app.HERE / "hivi_app.py") in cmd


def test_window_property_structs_match_the_windows_layout(app):
    import ctypes

    assert ctypes.sizeof(app._GUID) == 16
    assert ctypes.sizeof(app._PROPERTYKEY) == 20
    assert ctypes.sizeof(app._PROPVARIANT) == (24 if ctypes.sizeof(ctypes.c_void_p) == 8 else 16)
    g = app._GUID.parse(app._FMTID_APPUSERMODEL)
    assert (g.Data1, g.Data2, g.Data3) == (0x9F4C2855, 0x9F79, 0x4B39)
    assert bytes(g.Data4) == bytes.fromhex("A8D0E1D42DE1D5F3")


def test_identity_is_a_no_op_off_windows(app, monkeypatch):
    monkeypatch.setattr(app, "IS_WIN", False)
    assert app.set_window_identity(1234) is False


def test_startup_screens_use_the_shared_bunny_mark(app):
    for page in (app.splash_html(), app.error_html("boom")):
        assert "orb" not in page
        assert ".hb-l.hb-hivi" in page  # brand.css is inlined
        assert 'class="hb-l hb-blush_highlight hb-hivi"' in page
        assert "--hb-m-fur:url(data:image/png;base64," in page  # masks embedded, nothing to fetch
    assert "function hiviOutro" in app.splash_html()


def test_missing_brand_files_fall_back_to_the_icon(app, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "BRAND_DIR", tmp_path / "nope")
    page = app.splash_html()
    assert 'src="data:image/x-icon;base64,' in page


def test_outro_never_blocks_startup(app):
    class Window:
        def __init__(self):
            self.calls = []

        def evaluate_js(self, js):
            self.calls.append(js)
            return True

    w = Window()
    app.play_outro(w, timeout=1)
    assert w.calls == ["hiviOutro()", "window.__outroDone === true"]

    class Broken:
        def evaluate_js(self, js):
            raise RuntimeError("window closed")

    app.play_outro(Broken(), timeout=1)  # must not raise


def test_page_api_exposes_only_the_fullscreen_toggle(app):
    """pywebview walks every public attribute of js_api; a public window object froze HIVI."""
    public = [n for n in dir(app.FullscreenApi()) if not n.startswith("_")]
    assert public == ["toggle_fullscreen"]


def test_closed_window_never_starts_a_server(app, monkeypatch):
    monkeypatch.setattr(app, "hivi_running", lambda timeout=1.5: False)
    monkeypatch.setattr(app, "stop_stale_hivi", lambda: None)
    monkeypatch.setattr(app, "start_server", lambda: pytest.fail("must not start"))
    with pytest.raises(app.StartupError, match="cancelled"):
        app.ensure_server([None], cancelled=lambda: True)


def test_reopening_waits_for_a_closing_hivi_instead_of_refusing(app, monkeypatch):
    answers = iter([False, False, True])  # still closing, still closing, free
    monkeypatch.setattr(app, "single_instance", lambda: next(answers))
    monkeypatch.setattr(app, "focus_existing", lambda: False)
    monkeypatch.setattr(app.time, "sleep", lambda _s: None)
    assert app.claim_instance(wait_s=5) == "ok"
