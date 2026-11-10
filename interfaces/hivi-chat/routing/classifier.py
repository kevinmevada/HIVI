"""Classify a prompt into the capabilities it requires."""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class PromptClassification:
    task: str
    capabilities: frozenset[str]
    complexity: str


_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("coding", ("code", "coding", "debug", "implement", "refactor", "python", "javascript"), ("coding",)),
    ("reasoning", ("prove", "analyze", "reason", "trade-off", "architecture", "why"), ("reasoning",)),
    ("creative", ("write a story", "poem", "creative", "brainstorm", "imagine"), ("creative",)),
    ("vision", ("image", "picture", "screenshot", "diagram", "visual"), ("vision",)),
)


def classify_prompt(prompt: str, *, has_image: bool = False) -> PromptClassification:
    """Return a stable, dependency-free classification for a user prompt."""
    normalized = re.sub(r"\s+", " ", prompt.strip().lower())
    capabilities: set[str] = set()
    task = "general"
    if has_image:
        # An attached image needs a vision model whatever the text says.
        capabilities.add("vision")
        task = "vision"
    for candidate, terms, required in _RULES:
        if any(term in normalized for term in terms):
            if task == "general":
                task = candidate
            capabilities.update(required)
            break
    word_count = len(normalized.split())
    complexity = "high" if word_count >= 120 or len(capabilities) > 1 else "standard"
    return PromptClassification(task, frozenset(capabilities), complexity)
