"""Derive routing capabilities from the existing registry metadata."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from registry import ModelEntry


def model_capabilities(model: ModelEntry) -> frozenset[str]:
    """Map a registry entry to capabilities used by the A1 selector."""
    text = f"{model.name} {model.path} {model.params}".lower()
    capabilities = {"general"}
    if any(term in text for term in ("coder", "coding", "deepseek")):
        capabilities.add("coding")
    if any(term in text for term in ("reason", "27b", "30b", "gpt-oss")):
        capabilities.add("reasoning")
    if any(term in text for term in ("vision", "vl", "llava")):
        capabilities.add("vision")
    if "creative" in text:
        capabilities.add("creative")
    return frozenset(capabilities)
