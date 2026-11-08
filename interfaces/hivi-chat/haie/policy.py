"""Map understanding → continuous sampling params (V1.2 deliverable-aware budgets)."""
from __future__ import annotations

from .types import SampleParams, Understanding

MAX_TOKENS_CEILING = 8192

# temp / base max_tokens — simple math stays small; proofs can grow via complexity
_PROFILE = {
    "greeting": (0.50, 384),
    "coding": (0.15, 1024),
    "explanation": (0.35, 2048),
    "architecture": (0.30, 4096),  # V1.2: start 4k; complexity/est can raise to ceiling
    "math": (0.10, 512),  # V1.2: was 4096 — too wasteful for arithmetic
    "research": (0.20, 6144),
    "brainstorm": (0.80, 4096),
    "chat": (0.50, 1024),
}

_REASON_BUDGET = {
    "none": 0,
    "low": 256,
    "medium": 1024,
    "high": 2048,
}
_REASON_EFFORT = {
    "none": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
}


def _intent_top_p(intent: str) -> float:
    return {
        "coding": 0.9,
        "math": 0.85,
        "explanation": 0.92,
        "architecture": 0.9,
        "brainstorm": 0.95,
        "research": 0.92,
        "greeting": 0.9,
        "chat": 0.92,
    }.get(intent, 0.92)


def _intent_top_k(intent: str) -> int:
    return {
        "coding": 40,
        "math": 30,
        "explanation": 50,
        "architecture": 50,
        "brainstorm": 80,
        "research": 50,
        "greeting": 40,
        "chat": 50,
    }.get(intent, 50)


def derive_params(u: Understanding) -> SampleParams:
    u = u.normalized()
    intents = u.intents

    temps = [_PROFILE.get(i, (0.35, 1024))[0] for i in intents]
    maxes = [_PROFILE.get(i, (0.35, 1024))[1] for i in intents]
    temperature = sum(temps) / len(temps)
    if any(i in ("architecture", "coding", "math", "research") for i in intents):
        temperature = min(temperature, 0.35)
        if "architecture" in intents:
            temperature = min(temperature, 0.32)
        if "coding" in intents or "math" in intents:
            temperature = min(temperature, 0.18)
    temperature = max(0.05, min(0.95, temperature))

    top_p = sum(_intent_top_p(i) for i in intents) / len(intents)
    top_p = max(0.7, min(0.98, top_p))
    top_k = int(round(sum(_intent_top_k(i) for i in intents) / len(intents)))
    top_k = max(10, min(100, top_k))

    max_tokens = max(maxes)

    # Math: keep small unless complexity/proof-like
    if "math" in intents and len(intents) == 1:
        if u.complexity <= 2:
            max_tokens = min(max_tokens, 512)
        elif u.complexity >= 4:
            max_tokens = max(max_tokens, 4096)
        else:
            max_tokens = max(min(max_tokens, 2048), 1024)

    # Coding complexity bumps
    if "coding" in intents and u.complexity >= 4:
        max_tokens = max(max_tokens, 8192)
    elif "coding" in intents and u.complexity >= 3:
        max_tokens = max(max_tokens, 4096)

    # Classifier estimate wins when present (clamped), but don't inflate trivial math
    if u.expected_output_tokens is not None:
        est = int(u.expected_output_tokens)
        if "math" in intents and u.complexity <= 2:
            est = min(est, 512)
        max_tokens = max(256, min(MAX_TOKENS_CEILING, max(max_tokens // 2, est)))

    # Length hint — skip inflating simple math/greeting
    trivial = set(intents).issubset({"math", "greeting", "chat"}) and u.complexity <= 2
    if not trivial:
        length_floor = {"short": 384, "medium": 1024, "long": 2560, "xlarge": 4096}.get(
            u.length, 1024
        )
        max_tokens = max(max_tokens, length_floor)

    # Deliverable floors last — classifier est must not starve architecture/research
    dlow = (u.deliverable or "").lower()
    if "architecture" in intents:
        max_tokens = max(max_tokens, 4096 if u.complexity >= 3 else 3072)
    if "research" in intents:
        max_tokens = max(max_tokens, 3072 if u.complexity >= 3 else 2048)
    if any(k in dlow for k in ("compare", "comparison", "versus", "vs", "moe", "dense")):
        max_tokens = max(max_tokens, 2048)

    max_tokens = max(128, min(MAX_TOKENS_CEILING, int(max_tokens)))

    reasoning = u.reasoning
    if u.complexity >= 4 and reasoning in ("none", "low"):
        reasoning = "medium"
    if "math" in intents and reasoning == "none" and u.complexity >= 3:
        reasoning = "low"
    if "math" in intents and u.complexity <= 2:
        reasoning = "none"
        think = 0
    else:
        think = _REASON_BUDGET.get(reasoning, 256)
        if think + 256 > max_tokens:
            think = max(0, max_tokens // 4)

    effort = _REASON_EFFORT.get(reasoning, "low")
    return SampleParams(
        temperature=round(temperature, 3),
        top_p=round(top_p, 3),
        top_k=top_k,
        max_tokens=max_tokens,
        thinking_budget_tokens=int(think),
        reasoning_effort=effort,
    )
