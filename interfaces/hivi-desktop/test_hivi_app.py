"""HIVI desktop app: start / reuse / stop logic (no GUI needed)."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
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


def test_branding_points_to_the_desktop_icon(app):
    assert app.ICON.name == "hivi.ico"
    assert app.ICON.is_file()
    assert (app.HERE / "desktop.ini").read_text(encoding="utf-8").startswith(
        "[.ShellClassInfo]\nIconResource=hivi.ico,0"
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
        app.stop_owned(owned[0])
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