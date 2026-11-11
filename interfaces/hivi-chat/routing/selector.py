"""Select the best existing registry model for a classified prompt."""
from __future__ import annotations

from dataclasses import dataclass

from hivi_paths import paths
from registry import ModelEntry, scan_models

from .capabilities import model_capabilities
from .classifier import PromptClassification, classify_prompt
from .policy import RoutingPolicy, load_policy


@dataclass(frozen=True)
class RouteDecision:
    model: ModelEntry | None
    classification: PromptClassification
    score: int
    reason: str


def select_model(
    prompt: str,
    models: list[ModelEntry] | None = None,
    policy: RoutingPolicy | None = None,
    *,
    has_image: bool = False,
) -> RouteDecision:
    """Choose a model without starting or stopping any runtime process."""
    classification = classify_prompt(prompt, has_image=has_image)
    candidates = models if models is not None else scan_models()
    active_policy = policy or load_policy(paths.repo_root() / "configs" / "routing.yaml")
    eligible = [
        model for model in candidates
        # Embedding / reranker files are retrieval services, never chat answers.
        if getattr(model, "kind", "chat") == "chat"
        and (active_policy.max_size_gb is None or model.size_gb <= active_policy.max_size_gb)
    ]
    if not eligible:
        return RouteDecision(None, classification, 0, "no eligible models")

    scored = []
    for model in eligible:
        capabilities = model_capabilities(model)
        score = len(classification.capabilities & capabilities) * 100
        score += min(int(model.size_gb), 50)
        scored.append((score, model))
    scored.sort(key=lambda item: (-item[0], item[1].size_gb, item[1].name.lower()))
    score, selected = scored[0]
    if classification.capabilities and score < 100 and active_policy.fallback_model:
        selected = next((m for m in eligible if m.id == active_policy.fallback_model), selected)
        score = 0 if selected.id == active_policy.fallback_model else score
    matched = ", ".join(sorted(classification.capabilities)) or "general"
    return RouteDecision(selected, classification, score, f"{classification.task} task; matched {matched}")
