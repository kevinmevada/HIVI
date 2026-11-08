"""Engine-specific request field filters (llama vs Colibrì)."""
from __future__ import annotations

from typing import Any

from .types import SampleParams


def apply_params_to_body(
    body: dict[str, Any],
    params: SampleParams,
    engine: str | None,
) -> dict[str, Any]:
    """Inject params only where the client left them unset. Explicit client wins."""
    out = dict(body)
    eng = (engine or "").lower()

    def missing(key: str) -> bool:
        return key not in out or out[key] is None

    if missing("temperature"):
        out["temperature"] = params.temperature
    if missing("top_p"):
        out["top_p"] = params.top_p
    if missing("max_tokens"):
        out["max_tokens"] = params.max_tokens

    if eng == "llama":
        if missing("top_k"):
            out["top_k"] = params.top_k
        if params.thinking_budget_tokens > 0 and missing("thinking_budget_tokens"):
            out["thinking_budget_tokens"] = params.thinking_budget_tokens
        if params.reasoning_effort and missing("reasoning_effort"):
            # gpt-oss / reasoning templates; harmless if ignored
            out["reasoning_effort"] = params.reasoning_effort

    return out
