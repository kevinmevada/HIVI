import copy
import dataclasses
import sys
from pathlib import Path

import pytest

CHAT_DIR = Path(__file__).parents[2] / "interfaces" / "hivi-chat"
sys.path.insert(0, str(CHAT_DIR))


@pytest.fixture(autouse=True)
def _fresh_backend_gate(monkeypatch):
    """The gate's asyncio events bind to one event loop; each test gets its own."""
    import server

    monkeypatch.setattr(server, "GATE", server.BackendGate())


@pytest.fixture(autouse=True)
def _fresh_model_scan():
    """Tests create GGUF files on the fly; never serve a scan from another test."""
    import registry

    registry.clear_scan_cache()
    yield
    registry.clear_scan_cache()


@pytest.fixture(autouse=True)
def _restore_runtime_state():
    """Each test starts from the model state it found, whatever it changed.

    Some tests set STATE directly (a fake switch marks a model "ready"). Left
    behind, that made later tests see a loaded model with no backend: /api/status
    and /api/health then probed the empty port for real (slow on Windows, where a
    refused local connection takes ~2 s) and reported it "unhealthy".
    """
    import runtime

    saved = {f.name: copy.copy(getattr(runtime.STATE, f.name))
             for f in dataclasses.fields(runtime.STATE) if f.name != "proc"}
    yield
    for name, value in saved.items():
        setattr(runtime.STATE, name, value)


@pytest.fixture(autouse=True)
def _no_real_vram_reads(monkeypatch):
    """Model switches wait for VRAM to be released (runtime._wait_vram_released). In
    tests there is no real model on the GPU: never call the real nvidia-smi (on a PC
    that has one, every switch would wait out the timeout)."""
    import runtime

    monkeypatch.setattr(runtime, "_vram_used_mib", lambda: None)


@pytest.fixture(autouse=True)
def _gguf_facts_cache_in_tmp(monkeypatch, tmp_path):
    """The model-facts cache (gguf_context) lives in tmp, never in the repo's data/."""
    import gguf_context

    monkeypatch.setattr(gguf_context, "_info_file", lambda: tmp_path / "gguf_info.json")
    monkeypatch.setattr(gguf_context, "_INFO", {})
    monkeypatch.setattr(gguf_context, "_INFO_LOADED", False)

    def warm_now(paths_, on_done=None):   # no side threads outliving the test
        for p in paths_:
            gguf_context.model_facts(p)
        return False

    monkeypatch.setattr(gguf_context, "warm", warm_now)
