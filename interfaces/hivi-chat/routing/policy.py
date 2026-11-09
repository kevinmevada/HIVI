"""Load the small, explicit routing policy used by A1."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RoutingPolicy:
    strategy: str = "capability"
    fallback_model: str | None = None
    max_size_gb: float | None = None


def load_policy(path: Path) -> RoutingPolicy:
    """Load and validate routing.yaml."""
    if not path.exists():
        raise FileNotFoundError(f"routing policy not found: {path}")
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("routing", {}), dict):
        raise ValueError(f"invalid routing policy shape: {path}")
    config = raw["routing"]
    strategy = config.get("strategy", "capability")
    fallback = config.get("fallback_model")
    max_size = config.get("max_size_gb")
    if strategy != "capability":
        raise ValueError(f"unsupported routing strategy: {strategy!r}")
    if fallback is not None and not isinstance(fallback, str):
        raise ValueError("routing fallback_model must be a string or null")
    if max_size is not None and (isinstance(max_size, bool) or not isinstance(max_size, (int, float))):
        raise ValueError("routing max_size_gb must be numeric or null")
    return RoutingPolicy(strategy, fallback, float(max_size) if max_size is not None else None)
