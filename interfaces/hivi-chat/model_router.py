"""Resolve an OpenAI ``model`` field to a HIVI chat model.

A request's ``model`` may name:
  * a role from configs/model_roles.yaml (``main-chef``, ``security-auditor`` …),
  * an alias from the same file (``hivi-heavy``, ``hivi-review`` …),
  * or a registry id exactly as /api/models lists it.

Anything else resolves to None and the request goes to whatever is loaded,
which keeps the chat UI and existing tools working unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hivi_paths import paths
from registry import CHAT, ModelEntry, scan_models


@dataclass(frozen=True)
class Alias:
    match: tuple[str, ...]
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoleConfig:
    aliases: dict[str, Alias] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)


def _lower_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return ()
    return tuple(str(v).lower() for v in value if str(v).strip())


def parse_config(raw: Any) -> RoleConfig:
    if not isinstance(raw, dict):
        return RoleConfig()
    aliases: dict[str, Alias] = {}
    for name, spec in (raw.get("aliases") or {}).items():
        if isinstance(spec, dict):
            match = _lower_list(spec.get("match"))
            if match:
                aliases[str(name).lower()] = Alias(match, _lower_list(spec.get("exclude")))
    roles = {
        str(role).lower(): str(target).lower()
        for role, target in (raw.get("roles") or {}).items()
        if isinstance(target, str)
    }
    return RoleConfig(aliases, roles)


_CACHE: dict[str, tuple[float, RoleConfig]] = {}


def config_path() -> Path:
    return paths.repo_root() / "configs" / "model_roles.yaml"


def load_config(path: Path | None = None) -> RoleConfig:
    """Re-read when the file changes, so edits apply without a restart."""
    p = path or config_path()
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return RoleConfig()
    cached = _CACHE.get(str(p))
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        import yaml

        cfg = parse_config(yaml.safe_load(p.read_text(encoding="utf-8")))
    except Exception:
        cfg = RoleConfig()
    _CACHE[str(p)] = (mtime, cfg)
    return cfg


def _matches(model: ModelEntry, alias: Alias) -> bool:
    text = f"{model.path} {model.name}".lower().replace("\\", "/")
    return all(m in text for m in alias.match) and not any(x in text for x in alias.exclude)


@dataclass(frozen=True)
class Resolution:
    model: ModelEntry
    via: str  # "id" | "alias:<name>" | "role:<name>-><alias>"


def resolve(
    name: str | None,
    models: list[ModelEntry] | None = None,
    config: RoleConfig | None = None,
) -> Resolution | None:
    if not isinstance(name, str) or not name.strip():
        return None
    key = name.strip().lower()
    cfg = config if config is not None else load_config()
    chat = [m for m in (models if models is not None else scan_models()) if m.kind == CHAT]
    chat.sort(key=lambda m: (m.size_gb, m.name.lower()))

    for m in chat:
        if m.id == key:
            return Resolution(m, "id")

    via = None
    alias_name = key
    if key in cfg.roles:
        alias_name = cfg.roles[key]
        via = f"role:{key}->{alias_name}"
    alias = cfg.aliases.get(alias_name)
    if alias is None:
        # A role may also point straight at a registry id.
        if via is not None:
            for m in chat:
                if m.id == alias_name:
                    return Resolution(m, via)
        return None
    for m in chat:
        if _matches(m, alias):
            return Resolution(m, via or f"alias:{alias_name}")
    return None


def describe(models: list[ModelEntry] | None = None, config: RoleConfig | None = None) -> dict:
    """What every alias and role resolves to right now (for /api/roles)."""
    cfg = config if config is not None else load_config()
    models = models if models is not None else scan_models()
    out: dict[str, dict] = {"aliases": {}, "roles": {}}
    for name in cfg.aliases:
        r = resolve(name, models, cfg)
        out["aliases"][name] = None if r is None else {"id": r.model.id, "name": r.model.name}
    for role, target in cfg.roles.items():
        r = resolve(role, models, cfg)
        out["roles"][role] = {
            "alias": target,
            "model": None if r is None else {"id": r.model.id, "name": r.model.name},
        }
    return out
