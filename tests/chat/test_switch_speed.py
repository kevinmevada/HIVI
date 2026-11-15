"""Model-switch speed: timing per phase, and no slow process spawns in hot loops."""
from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest
import runtime


def test_switch_records_phases_and_logs_them(monkeypatch, tmp_path):
    monkeypatch.setenv("HIVI_LOG_DIR", str(tmp_path))
    model = type("M", (), {"id": "m1", "name": "M1", "engine": "llama", "size_gb": 5.3})()

    def fake_select(m):
        with runtime._phase("stop"):
            time.sleep(0.1)
        with runtime._phase("load"):
            time.sleep(0.2)
        runtime.STATE.status, runtime.STATE.detail = "ready", "llama ready"
        return runtime.STATE

    monkeypatch.setattr(runtime, "_select_model", fake_select)
    runtime.note_switch_wait(0.5)
    runtime.select_model(model)
    t = runtime.STATE.timings
    assert t["wait"] == 0.5 and 0.1 <= t["stop"] < 0.3 and 0.2 <= t["load"] < 0.4
    assert 0.8 <= t["total"] < 1.2
    assert "loaded in" in runtime.STATE.detail and "load 0.2s" in runtime.STATE.detail
    rec = json.loads((tmp_path / "hivi_switch_times.jsonl").read_text().splitlines()[-1])
    assert rec["model"] == "m1" and rec["status"] == "ready" and rec["load"] >= 0.2
    # The wait is used once, not carried into the next switch.
    runtime.select_model(model)
    assert "wait" not in runtime.STATE.timings


def test_failed_switch_is_timed_too(monkeypatch, tmp_path):
    monkeypatch.setenv("HIVI_LOG_DIR", str(tmp_path))
    model = type("M", (), {"id": "m2", "name": "M2", "engine": "llama", "size_gb": 1})()

    def boom(m):
        runtime.STATE.status = "error"
        return runtime.STATE

    monkeypatch.setattr(runtime, "_select_model", boom)
    runtime.select_model(model)
    rec = json.loads((tmp_path / "hivi_switch_times.jsonl").read_text().splitlines()[-1])
    assert rec["status"] == "error" and "total" in rec


def test_free_port_check_spawns_nothing(monkeypatch):
    def no_spawn(*a, **k):
        raise AssertionError("spawned a process to check a free port")

    monkeypatch.setattr(subprocess, "run", no_spawn)
    t = time.perf_counter()
    assert runtime._port_owner_pid(1) is None  # nothing listens on port 1
    assert time.perf_counter() - t < 1.0


def test_process_checks_spawn_nothing(monkeypatch):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        monkeypatch.setattr(subprocess, "run",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("spawned")))
        assert runtime._pid_alive(proc.pid)
        assert runtime._pid_looks_like_backend(proc.pid)
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.skipif(sys.platform == "win32", reason="zombies are a POSIX thing")
def test_exited_child_counts_as_dead_before_it_is_reaped():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    time.sleep(0.5)  # exited, not yet waited on: a zombie
    try:
        assert not runtime._pid_alive(proc.pid)  # used to look alive for 10 s
    finally:
        proc.wait()


def test_port_in_use_sees_a_listener():
    import socket

    with socket.socket() as srv:
        srv.bind((runtime.BACKEND_HOST, 0))
        srv.listen()
        port = srv.getsockname()[1]
        assert runtime._port_in_use(port)
        assert runtime._port_owner_pid(port) is not None
    assert not runtime._port_in_use(port)
