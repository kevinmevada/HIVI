"""Paths resolver, registry engine choice, and backend command lines."""
from __future__ import annotations

from pathlib import Path

import pytest

from hivi import paths
import registry
import runtime


@pytest.fixture()
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "Models"
    monkeypatch.setenv("HIVI_MODELS_ROOT", str(root))
    return root


def _touch(p: Path, size: int = 16) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * size)
    return p


# ---- paths ----------------------------------------------------------------

def test_no_machine_specific_paths_in_resolvers(monkeypatch: pytest.MonkeyPatch):
    for var in ("HIVI_ROOT", "HIVI_MODELS_ROOT", "HIVI_COLI", "HIVI_DATA_ROOT"):
        monkeypatch.delenv(var, raising=False)
    repo = Path(__file__).resolve().parents[2]
    assert paths.repo_root() == repo
    assert paths.models_root() == repo / "Models"  # from configs/storage.yaml
    assert paths.colibri_cli().is_relative_to(repo)


def test_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HIVI_MODELS_ROOT", str(tmp_path / "m"))
    monkeypatch.setenv("HIVI_LLAMA_SERVER", str(tmp_path / "bin" / "llama-server"))
    monkeypatch.setenv("HIVI_COLI", str(tmp_path / "coli"))
    assert paths.models_root() == tmp_path / "m"
    assert paths.llama_server() == tmp_path / "bin" / "llama-server"
    assert paths.colibri_cli() == tmp_path / "coli"


def test_repo_has_no_hardcoded_user_paths():
    repo = Path(__file__).resolve().parents[2]
    offenders = []
    for f in list(repo.glob("interfaces/**/*.py")) + list(repo.glob("scripts/**/*.py")) + list(
        repo.glob("src/hivi/*.py")
    ) + list(repo.glob("src/hivi/[!i]*/**/*.py")) + list(repo.glob("scripts/**/*.ps1")):
        text = f.read_text(encoding="utf-8-sig", errors="ignore")
        if "C:\\Users\\" in text or "C:/Users/" in text:
            offenders.append(str(f.relative_to(repo)))
    assert offenders == []


# ---- registry ---------------------------------------------------------------

def test_projector_routes_model_to_llama_with_vision(models_dir: Path):
    folder = models_dir / "General" / "Qwen3.6-35B-A3B"
    _touch(folder / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
    _touch(folder / "mmproj-F16.gguf")
    entries = registry.scan_models()
    assert len(entries) == 1  # projector itself is not listed as a model
    m = entries[0]
    assert m.engine == "llama"
    assert m.vision_projector_path and m.vision_projector_path.endswith("mmproj-F16.gguf")


def test_projector_does_not_leak_across_folders(models_dir: Path):
    _touch(models_dir / "General" / "base" / "Qwen3.6-35B-A3B-Q4_K_M.gguf")
    _touch(models_dir / "General" / "base" / "mmproj-F16.gguf")
    _touch(models_dir / "General" / "heretic" / "Qwen3.6-35B-A3B-uncensored-heretic-Q4_K_M.gguf")
    by_name = {Path(m.path).parent.name: m for m in registry.scan_models()}
    assert by_name["base"].vision_projector_path
    assert by_name["heretic"].vision_projector_path is None
    assert by_name["heretic"].engine == "llama"  # uncensored builds run on llama.cpp


def test_plain_qwen3_text_model_keeps_colibri(models_dir: Path):
    _touch(models_dir / "General" / "plain" / "Qwen3.6-35B-A3B-Q5_K_M.gguf")
    (m,) = registry.scan_models()
    assert m.engine == "colibri"


def test_uncensored_builds_route_to_llama(models_dir: Path):
    names = [
        "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf",
        "Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q6_K.gguf",
        "Huihui-Qwen3.6-35B-A3B-abliterated-Q6_K.gguf",
        "Qwen3.8-27B-Heretic-Q5_K_M.gguf",
    ]
    for i, name in enumerate(names):
        _touch(models_dir / "Coding" / f"m{i}" / name)
    entries = registry.scan_models()
    assert len(entries) == len(names)
    assert all(m.engine == "llama" for m in entries), [(m.path, m.engine) for m in entries]


def test_gpt_oss_stays_on_llama(models_dir: Path):
    _touch(models_dir / "Reasoning" / "gpt-oss-20b-MXFP4.gguf")
    (m,) = registry.scan_models()
    assert m.engine == "llama"


# ---- runtime command lines ------------------------------------------------

def _entry(path: str, *, projector: str | None = None, mid: str = "m") -> registry.ModelEntry:
    return registry.ModelEntry(
        id=mid, name="x", path=path, quant="Q4_K_M", params="?", size_gb=1.0,
        engine="llama", engine_reason="", vision_projector_path=projector,
    )


def test_llama_args_include_mmproj_only_with_projector():
    with_vision = runtime.build_llama_args(
        _entry("/m/qwen.gguf", projector="/m/mmproj-F16.gguf"), llama=Path("llama-server"), threads=8
    )
    i = with_vision.index("--mmproj")
    assert with_vision[i + 1] == "/m/mmproj-F16.gguf"
    text_only = runtime.build_llama_args(_entry("/m/qwen.gguf"), llama=Path("llama-server"), threads=8)
    assert "--mmproj" not in text_only


def test_llama_args_moe_and_gpt_oss_flags():
    moe = runtime.build_llama_args(
        _entry("/m/Qwen3-Coder-30B-A3B.gguf"), llama=Path("llama-server"), threads=8
    )
    assert "--cpu-moe" in moe and moe[moe.index("-c") + 1] == "4096"
    oss = runtime.build_llama_args(_entry("/m/gpt-oss-20b.gguf"), llama=Path("llama-server"), threads=8)
    assert oss[oss.index("--reasoning") + 1] == "on"
    assert str(runtime.BACKEND_PORT) in oss


def test_colibri_args_use_backend_port():
    args = runtime.build_colibri_args(Path("model.colibri"), coli=Path("coli"), python="py")
    assert args[:3] == ["py", "coli", "serve"]
    assert args[args.index("--port") + 1] == str(runtime.BACKEND_PORT)


def test_release_backend_never_touches_unowned_processes(monkeypatch, tmp_path: Path):
    """No owned PID → nothing is killed, even if the port is busy."""
    monkeypatch.setenv("HIVI_DATA_ROOT", str(tmp_path))
    killed: list[int] = []
    monkeypatch.setattr(runtime, "_kill_tree", lambda pid: killed.append(pid))
    monkeypatch.setattr(runtime, "_port_owner_pid", lambda port: 4242)
    runtime.STATE.proc = None
    runtime.release_backend()
    assert killed == []


def test_busy_port_is_reported_not_killed(monkeypatch):
    monkeypatch.setattr(runtime, "_port_owner_pid", lambda port: 4242)
    with pytest.raises(RuntimeError, match="4242"):
        runtime._assert_port_free()


def test_old_colibri_is_rejected(tmp_path: Path):
    c = tmp_path / "c"
    c.mkdir()
    (c / "coli").write_text("#!/usr/bin/env python3\n")
    (c / "version.py").write_text('__version__ = "1.10.2"\n')
    with pytest.raises(RuntimeError, match="needs >= 2.1.0"):
        runtime._require_colibri(c / "coli")
    (c / "version.py").write_text('__version__ = "2.1.0"\n')
    runtime._require_colibri(c / "coli")  # accepted