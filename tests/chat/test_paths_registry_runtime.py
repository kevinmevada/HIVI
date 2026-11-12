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
    monkeypatch.setenv("HIVI_DATA_ROOT", str(tmp_path / "data"))  # engine pins live here
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
    assert by_name["base"].engine == "llama"  # vision needs --mmproj
    assert by_name["heretic"].engine == "colibri"  # text-only qwen3.6 → Colibrì


def test_plain_qwen3_text_model_keeps_colibri(models_dir: Path):
    _touch(models_dir / "General" / "plain" / "Qwen3.6-35B-A3B-Q5_K_M.gguf")
    (m,) = registry.scan_models()
    assert m.engine == "colibri"


def test_uncensored_builds_follow_their_family(models_dir: Path):
    """Abliterated/heretic qwen builds go to Colibrì like their base models."""
    names = [
        "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf",
        "Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q6_K.gguf",
        "Huihui-Qwen3.6-35B-A3B-abliterated-Q6_K.gguf",
        "Qwen3.8-27B-Heretic-Q5_K_M.gguf",
    ]
    for i, name in enumerate(names):
        _touch(models_dir / "Coding" / f"m{i}" / name)
    _touch(models_dir / "Reasoning" / "oss" / "gpt-oss-20b-abliterated-q8_0.gguf")
    entries = {Path(m.path).parent.name: m for m in registry.scan_models()}
    assert all(entries[f"m{i}"].engine == "colibri" for i in range(len(names)))
    assert entries["oss"].engine == "llama"  # architecture Colibrì doesn't run


def test_engine_pin_overrides_auto_choice(models_dir: Path):
    _touch(models_dir / "Coding" / "a" / "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf")
    (m,) = registry.scan_models()
    assert m.engine == "colibri"
    registry.set_engine_pin(m.id, "llama")
    (m,) = registry.scan_models()
    assert m.engine == "llama" and "pinned" in m.engine_reason
    registry.set_engine_pin(m.id, "auto")
    (m,) = registry.scan_models()
    assert m.engine == "colibri"
    with pytest.raises(ValueError):
        registry.set_engine_pin(m.id, "ollama")


def test_gpt_oss_stays_on_llama(models_dir: Path):
    _touch(models_dir / "Reasoning" / "gpt-oss-20b-MXFP4.gguf")
    (m,) = registry.scan_models()
    assert m.engine == "llama"


# ---- runtime command lines ------------------------------------------------

def _entry(path: str, *, projector: str | None = None, mid: str = "m",
           uncensored: bool = False) -> registry.ModelEntry:
    return registry.ModelEntry(
        id=mid, name="x", path=path, quant="Q4_K_M", params="?", size_gb=1.0,
        engine="llama", engine_reason="", vision_projector_path=projector,
        uncensored=uncensored,
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


def test_uncensored_llama_args_keep_model_defaults():
    normal = runtime.build_llama_args(_entry("/m/Qwen3.6-35B-A3B-Q5_K_M.gguf"),
                                      llama=Path("llama-server"), threads=8)
    assert normal[normal.index("--reasoning") + 1] == "off"
    assert "--repeat-penalty" in normal and "--frequency-penalty" in normal
    unc = runtime.build_llama_args(
        _entry("/m/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf", uncensored=True),
        llama=Path("llama-server"), threads=8)
    assert "--reasoning" not in unc  # thinking decided per request by HAIE
    assert unc[unc.index("-c") + 1] == "16384"
    assert "--repeat-penalty" not in unc and "--frequency-penalty" not in unc
    oss = runtime.build_llama_args(_entry("/m/gpt-oss-20b-abliterated-q8_0.gguf", uncensored=True),
                                   llama=Path("llama-server"), threads=8)
    assert oss[oss.index("--reasoning") + 1] == "on" and "--reasoning-budget" not in oss


def test_registry_flags_uncensored_chat_models(models_dir: Path):
    _touch(models_dir / "Coding" / "a" / "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf")
    _touch(models_dir / "General" / "b" / "Qwen3.6-35B-A3B-Q5_K_M.gguf")
    by_dir = {Path(m.path).parent.name: m for m in registry.scan_models()}
    assert by_dir["a"].uncensored is True and by_dir["b"].uncensored is False


def test_colibri_args_use_backend_port():
    args = runtime.build_colibri_args(Path("model.colibri"), coli=Path("coli"), python="py")
    assert args[:3] == ["py", "coli", "serve"]
    assert args[args.index("--port") + 1] == str(runtime.BACKEND_PORT)
    assert args[args.index("--policy") + 1] == "quality"  # keeps quantization + exact routing
    assert args[args.index("--ctx") + 1] == "8192"
    unc = runtime.build_colibri_args(Path("m.colibri"), coli=Path("coli"), python="py",
                                     uncensored=True)
    assert unc[unc.index("--ctx") + 1] == "16384"


# ---- llama router mode --------------------------------------------------------

def test_preset_options_translate_cli_args():
    m = _entry("C:/M/qwen.gguf", projector="C:/M/mmproj-F16.gguf", mid="qwen")
    opts = dict(runtime.preset_options(
        runtime.build_llama_args(m, llama=Path("llama-server"), threads=8)))
    assert opts["model"] == "C:/M/qwen.gguf"
    assert opts["mmproj"] == "C:/M/mmproj-F16.gguf"
    assert opts["jinja"] == "true" and opts["c"] == "8192" and opts["fa"] == "on"
    assert "host" not in opts and "port" not in opts  # the router owns these


def test_preset_file_has_one_section_per_model():
    text = runtime.build_llama_preset(
        [_entry("/m/a.gguf", mid="a"), _entry("/m/gpt-oss-20b.gguf", mid="oss", uncensored=True)],
        llama=Path("llama-server"), threads=6)
    assert "[a]" in text and "[oss]" in text
    oss = text.split("[oss]")[1]
    assert "reasoning = on" in oss and "model = /m/gpt-oss-20b.gguf" in oss
    assert "-" not in [line.split("=")[0].strip()[:1] for line in text.splitlines() if "=" in line]


def test_router_args(monkeypatch):
    monkeypatch.setattr(runtime, "MEMORY_MARGIN", "512")
    args = runtime.build_router_args(Path("llama-server"), Path("p.ini"), margin_ok=True)
    assert "-m" not in args and "--models-preset" in args and "--no-models-autoload" in args
    assert args[args.index("--models-memory-margin") + 1] == "512"
    old = runtime.build_router_args(Path("llama-server"), Path("p.ini"), margin_ok=False)
    assert "--models-memory-margin" not in old  # older builds without the flag


def test_paths_with_ini_comment_chars_skip_router():
    assert runtime._preset_safe(_entry("C:/M/qwen.gguf"))
    assert not runtime._preset_safe(_entry("C:/M/#new/qwen.gguf"))


def test_ram_evictions_oldest_first():
    big = _entry("/m/big.gguf", mid="big")
    big.size_gb = 20.0
    sizes = {"old": 7.0, "mid": 7.0, "new": 3.0}
    # 24 GB budget: 17 resident + 21 needed → drop oldest until it fits
    assert runtime.plan_evictions(big, ["old", "mid", "new"], sizes, 24.0) == ["old", "mid"]
    small = _entry("/m/s.gguf", mid="s")
    assert runtime.plan_evictions(small, ["old", "mid", "new"], sizes, 24.0) == []


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