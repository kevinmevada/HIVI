"""Single source of truth for HIVI filesystem locations.

Every path resolves in this order:
  1. an explicit environment variable (for per-machine overrides),
  2. configs/storage.yaml (for models_root / data_root),
  3. a default relative to the repository root.

Nothing here is tied to one user's machine, so the repo can move between
laptops, drives and user accounts without edits.
"""
from __future__ import annotations

import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

_REPO_FROM_HERE = Path(__file__).resolve().parents[2]  # src/hivi/paths.py -> repo


def repo_root() -> Path:
    """HIVI_ROOT, else the repo that contains this file."""
    env = os.environ.get("HIVI_ROOT")
    return Path(env).expanduser().resolve() if env else _REPO_FROM_HERE


def _resolve(value: str | os.PathLike[str], base: Path | None = None) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base or repo_root()) / p


@lru_cache(maxsize=4)
def _storage_config(root: str) -> dict[str, Any]:
    cfg = Path(root) / "configs" / "storage.yaml"
    if not cfg.is_file():
        return {}
    try:
        import yaml

        raw = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    storage = raw.get("storage") if isinstance(raw, dict) else None
    return storage if isinstance(storage, dict) else {}


def storage_config() -> dict[str, Any]:
    return _storage_config(str(repo_root()))


def models_root() -> Path:
    """HIVI_MODELS_ROOT, else storage.models_root, else <repo>/Models."""
    env = os.environ.get("HIVI_MODELS_ROOT")
    if env:
        return _resolve(env)
    return _resolve(storage_config().get("models_root") or "Models")


def data_root() -> Path:
    """HIVI_DATA_ROOT, else storage.data_root, else <repo>/data."""
    env = os.environ.get("HIVI_DATA_ROOT")
    if env:
        return _resolve(env)
    return _resolve(storage_config().get("data_root") or "data")


def inference_root() -> Path:
    return repo_root() / "src" / "hivi" / "inference"


def colibri_cli() -> Path:
    """HIVI_COLI, else the vendored/fetched Colibrì CLI under src/hivi/inference."""
    env = os.environ.get("HIVI_COLI")
    if env:
        return _resolve(env)
    return inference_root() / "colibri-LLM-main" / "c" / "coli"


def llama_server() -> Path:
    """HIVI_LLAMA_SERVER, else the bootstrapped binary, else llama-server on PATH."""
    env = os.environ.get("HIVI_LLAMA_SERVER")
    if env:
        return _resolve(env)
    exe = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    bootstrapped = data_root() / "P1_1" / "runtime" / "llama.cpp" / "bin" / exe
    if bootstrapped.is_file():
        return bootstrapped
    on_path = shutil.which("llama-server")
    return Path(on_path) if on_path else bootstrapped


def logs_dir() -> Path:
    return _resolve(os.environ.get("HIVI_LOG_DIR") or (data_root() / "logs"))


def python_exe() -> str:
    """Interpreter for helper subprocesses (Colibrì CLI, verification sandboxes)."""
    return os.environ.get("HIVI_PYTHON") or sys.executable or "python"


__all__ = [
    "colibri_cli",
    "data_root",
    "inference_root",
    "llama_server",
    "logs_dir",
    "models_root",
    "python_exe",
    "repo_root",
    "storage_config",
]
