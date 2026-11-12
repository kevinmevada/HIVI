"""HIVI Adaptive Intelligence Engine (HAIE) V1.2 — quality + latency; params-only."""
from __future__ import annotations

import time
from typing import Any

from .adapters import apply_params_to_body
from .eval_store import log_plan
from .fast_path import extract_last_user, try_fast_path
from .heuristic import try_heuristic_path
from .policy import adapt_for_uncensored, adapt_thinking, derive_params
from .semantic import classify_semantic
from .types import PlanMeta, Understanding

_MODEL_CACHE: dict[str, tuple[float, str]] = {}
_MODEL_CACHE_TTL = 60.0


def _enrich_from_text(u: Understanding, user_text: str) -> Understanding:
    """Light keyword enrichment so deliverable floors apply when the classifier underspecifies."""
    text = (user_text or "").lower()
    intents = list(u.intents)
    changed = False
    if any(
        k in text
        for k in ("compare", "comparison", " versus ", " vs ", "moe", "dense", "tradeoff", "trade-off")
    ):
        if "research" not in intents:
            intents.append("research")
            changed = True
        if "explanation" not in intents and "architecture" not in intents:
            intents.append("explanation")
            changed = True
    if any(k in text for k in ("architecture", "design a ", "system design", "high-level design")):
        if "architecture" not in intents:
            intents.append("architecture")
            changed = True
    if not changed:
        return u
    u.intents = intents
    return u.normalized()


def _safe_fallback(user_text: str = "") -> Understanding:
    """Conservative defaults; soft multi-intent hints when semantic is unavailable."""
    text = (user_text or "").lower()
    intents: list[str] = []
    if any(k in text for k in ("explain", "why", "how does", "what does")):
        intents.append("explanation")
    if any(
        k in text
        for k in ("fix", "implement", "write", "optimize", "code", "function", "debug", "concurrency")
    ):
        intents.append("coding")
    if any(k in text for k in ("architecture", "design", "structure")):
        intents.append("architecture")
    if any(k in text for k in ("brainstorm", "explore", "alternative")):
        intents.append("brainstorm")
    if any(k in text for k in ("calculate", "compute", "prove")) or any(
        c.isdigit() for c in text
    ) and any(op in text for op in ("+", "*", "/", "=")):
        intents.append("math")
    if not intents:
        intents = ["chat"]
    complexity = 3 if len(intents) > 1 else 2
    return Understanding(
        intents=intents,
        complexity=complexity,
        deliverable="reply",
        reasoning="medium" if len(intents) > 1 or complexity >= 3 else "low",
        length="medium",
        ambiguity=0.5,
        confidence=0.3,
        path="fallback",
    ).normalized()


def _resolve_openai_model(backend_base: str) -> str | None:
    now = time.time()
    cached = _MODEL_CACHE.get(backend_base)
    if cached and now - cached[0] < _MODEL_CACHE_TTL:
        return cached[1]
    try:
        import httpx

        with httpx.Client(timeout=2.0) as client:
            r = client.get(f"{backend_base.rstrip('/')}/v1/models")
            if r.status_code >= 400:
                return None
            data = r.json()
            rows = data.get("data") or []
            if rows and isinstance(rows[0], dict) and rows[0].get("id"):
                mid = str(rows[0]["id"])
                _MODEL_CACHE[backend_base] = (now, mid)
                return mid
    except Exception:
        return None
    return None


def _recent_context(body: dict[str, Any], limit: int = 4) -> str:
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        return ""
    bits: list[str] = []
    for msg in messages[-limit:]:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            bits.append(f"{role}: {content[:200]}")
    return "\n".join(bits)


def plan_request(
    body: dict[str, Any],
    *,
    engine: str | None,
    backend_base: str,
    model_id: str | None = None,
    uncensored: bool = False,
) -> tuple[dict[str, Any], PlanMeta]:
    """Understand the turn and inject dynamic sampling params (client values win)."""
    user = extract_last_user(body)
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    history_len = len(messages)

    understanding = try_fast_path(user, history_len=history_len)
    classify_ms: float | None = None
    classify_prompt_ms: float | None = None
    classify_predicted_ms: float | None = None
    if understanding is None:
        # Clear single-intent → heuristic (no LLM round-trip)
        understanding = try_heuristic_path(user)
    if understanding is None:
        model = str(body.get("model") or "local")
        if model in ("local", "", "default") and model_id:
            model = _resolve_openai_model(backend_base) or model_id
        elif model in ("local", "", "default"):
            model = _resolve_openai_model(backend_base) or model
        t0 = time.perf_counter()
        result = classify_semantic(
            backend_base=backend_base,
            model=model,
            user_text=user,
            recent_context=_recent_context(body),
        )
        classify_ms = (time.perf_counter() - t0) * 1000.0
        if result is None:
            understanding = _safe_fallback(user)
        else:
            understanding = result.understanding
            understanding.path = "semantic"
            classify_prompt_ms = result.prompt_ms
            classify_predicted_ms = result.predicted_ms
            if result.total_ms is not None:
                classify_ms = result.total_ms
    understanding = _enrich_from_text(understanding, user)

    params = derive_params(understanding)
    if uncensored:
        params = adapt_for_uncensored(params, understanding)
    if (engine or "").lower() == "colibri":
        params = adapt_thinking(params, understanding)
    adapted = apply_params_to_body(body, params, engine)
    meta = PlanMeta(
        path=understanding.path,
        intents=list(understanding.intents),
        complexity=understanding.complexity,
        deliverable=understanding.deliverable,
        reasoning=understanding.reasoning,
        length=understanding.length,
        ambiguity=understanding.ambiguity,
        confidence=understanding.confidence,
        params={
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "max_tokens": params.max_tokens,
            "thinking_budget_tokens": params.thinking_budget_tokens,
            "reasoning_effort": params.reasoning_effort,
            "enable_thinking": params.enable_thinking,
        },
        engine=engine,
        classify_ms=classify_ms,
        classify_prompt_ms=classify_prompt_ms,
        classify_predicted_ms=classify_predicted_ms,
    )
    try:
        log_plan(
            model_id=model_id,
            engine=engine,
            meta=meta.as_dict(),
            user_preview=user,
        )
    except Exception:
        pass
    return adapted, meta


__all__ = ["plan_request", "PlanMeta", "Understanding"]
