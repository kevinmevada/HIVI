"""Semantic classify via the already-loaded backend model (V1.2 latency-tuned)."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .harmony import extract_final_text
from .types import Understanding

# Lean prompt — fewer prefill tokens. Compact keys preferred (parser expands).
CLASSIFIER_SYSTEM = (
    "Classify request. Reply with ONE JSON object only (no markdown, no prose). "
    'Prefer short keys: {"i":["coding"],"p":"coding","c":3,"d":"code","n":2048,"r":true,"f":0.85}. '
    "i=intents∈coding|explanation|architecture|math|research|brainstorm|greeting|chat; "
    "c=complexity1-5; n=expected_output_tokens; r=reasoning_required; f=confidence."
)

# Keep classify short — on MoE 30B a long classify steals the whole turn.
_TIMEOUT = httpx.Timeout(6.0, connect=2.0)
_CLASSIFY_MAX_TOKENS = 96
_CLASSIFY_DEADLINE_S = 5.5
_FOLLOWUP_CHARS = 160
_CONTEXT_CHARS = 300
_VALID_INTENTS = {
    "coding",
    "explanation",
    "architecture",
    "math",
    "research",
    "brainstorm",
    "greeting",
    "chat",
}


@dataclass
class ClassifyResult:
    understanding: Understanding
    prompt_ms: float | None = None
    predicted_ms: float | None = None
    total_ms: float | None = None


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    t = extract_final_text(text) or text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _expand_compact(obj: dict[str, Any]) -> dict[str, Any]:
    """Accept short keys {i,c,d,n,p,r,f} or full schema keys."""
    out = dict(obj)
    if "i" in out and "task_intents" not in out and "intents" not in out:
        out["task_intents"] = out["i"]
    if "p" in out and "primary_intent" not in out:
        out["primary_intent"] = out["p"]
    if "c" in out and "complexity" not in out:
        out["complexity"] = out["c"]
    if "d" in out and "deliverable" not in out:
        out["deliverable"] = out["d"]
    if "n" in out and "expected_output_tokens" not in out:
        out["expected_output_tokens"] = out["n"]
    if "r" in out and "reasoning_required" not in out:
        out["reasoning_required"] = out["r"]
    if "f" in out and "confidence" not in out:
        out["confidence"] = out["f"]
    return out


def _validate(obj: dict[str, Any]) -> dict[str, Any] | None:
    obj = _expand_compact(obj)
    intents_raw = obj.get("task_intents") or obj.get("intents")
    if isinstance(intents_raw, str):
        intents_raw = [intents_raw]
    if not isinstance(intents_raw, list) or not intents_raw:
        return None
    intents = [
        str(x).lower().strip()
        for x in intents_raw
        if str(x).lower().strip() in _VALID_INTENTS
    ]
    if not intents:
        return None
    primary = str(obj.get("primary_intent") or intents[0]).lower().strip()
    if primary not in _VALID_INTENTS:
        primary = intents[0]
    if primary not in intents:
        intents = [primary] + intents
    try:
        complexity = int(obj.get("complexity") or 2)
    except Exception:
        return None
    complexity = max(1, min(5, complexity))
    try:
        confidence = float(
            obj.get("confidence") if obj.get("confidence") is not None else 0.6
        )
    except Exception:
        confidence = 0.6
    confidence = max(0.0, min(1.0, confidence))
    reasoning_required = bool(obj.get("reasoning_required", complexity >= 3))
    try:
        expected = int(obj.get("expected_output_tokens") or 1024)
    except Exception:
        expected = 1024
    expected = max(128, min(8192, expected))
    deliverable = str(obj.get("deliverable") or "reply")[:80]
    if expected <= 512:
        length = "short"
    elif expected <= 2048:
        length = "medium"
    elif expected <= 4096:
        length = "long"
    else:
        length = "xlarge"
    if not reasoning_required:
        reasoning = "none"
    elif complexity >= 4:
        reasoning = "high"
    elif complexity >= 3:
        reasoning = "medium"
    else:
        reasoning = "low"
    return {
        "intents": intents,
        "complexity": complexity,
        "deliverable": deliverable,
        "reasoning": reasoning,
        "length": length,
        "expected_output_tokens": expected,
        "ambiguity": max(0.0, 1.0 - confidence),
        "confidence": confidence,
    }


def _from_validated(v: dict[str, Any]) -> Understanding:
    return Understanding(
        intents=list(v["intents"]),
        complexity=int(v["complexity"]),
        deliverable=str(v["deliverable"]),
        reasoning=str(v["reasoning"]),
        length=str(v["length"]),
        ambiguity=float(v["ambiguity"]),
        confidence=float(v["confidence"]),
        path="semantic",
        expected_output_tokens=int(v["expected_output_tokens"]),
    ).normalized()


def _message_text(data: dict[str, Any]) -> str:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):
        content = " ".join(
            str(p.get("text") or "") for p in content if isinstance(p, dict)
        )
    final = extract_final_text(content if isinstance(content, str) else "")
    if final:
        return final
    if isinstance(content, str) and content.strip() and "<|channel|>" not in content:
        return content.strip()
    return final or ""


def _timings(data: dict[str, Any]) -> tuple[float | None, float | None]:
    t = data.get("timings") or {}
    prompt_ms = t.get("prompt_ms")
    predicted_ms = t.get("predicted_ms")
    try:
        prompt_ms = float(prompt_ms) if prompt_ms is not None else None
    except Exception:
        prompt_ms = None
    try:
        predicted_ms = float(predicted_ms) if predicted_ms is not None else None
    except Exception:
        predicted_ms = None
    return prompt_ms, predicted_ms


# Short-key example kept for optional probe only — not used as primary (empty/final
# regressions on gpt-oss). Parser still accepts compact keys via _expand_compact.
_COMPACT_SYSTEM = (
    "Classify. JSON only in final answer. Short keys OK: "
    '{"i":["coding","explanation"],"p":"coding","c":3,"d":"explain_and_optimize","n":2048,"r":true,"f":0.9}'
)


def _one_classify_call(
    *,
    client: httpx.Client,
    url: str,
    model: str,
    prompt: str,
    system: str = CLASSIFIER_SYSTEM,
    max_tokens: int = _CLASSIFY_MAX_TOKENS,
    prefill: bool = True,
) -> ClassifyResult | None:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    # Prefill steers the model straight into JSON (fewer wasted analysis tokens).
    if prefill:
        messages.append({"role": "assistant", "content": '{"i":'})
    payload = {
        "model": model or "local",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        "reasoning_effort": "none",
        "reasoning_format": "none",
        # Uncensored models now run with thinking on; the classifier must not.
        "chat_template_kwargs": {"enable_thinking": False},
        "enable_thinking": False,  # Colibrì reads this one, not chat_template_kwargs
    }
    t0 = time.perf_counter()
    r = client.post(url, json=payload)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    if r.status_code >= 400:
        return None
    data = r.json()
    prompt_ms, predicted_ms = _timings(data)
    text = _message_text(data)
    # With prefill, content may be a JSON suffix — reassemble
    raw = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
    candidates = [text]
    if isinstance(raw, str):
        candidates.append(extract_final_text(raw) or raw)
        if prefill and not raw.strip().startswith("{"):
            candidates.append('{"i":' + raw.strip())
            candidates.append('{"i":' + (extract_final_text(raw) or raw).strip())
    obj = None
    for cand in candidates:
        obj = _extract_json(cand)
        if obj:
            break
    if not obj:
        return None
    validated = _validate(obj)
    if not validated:
        return None
    return ClassifyResult(
        understanding=_from_validated(validated),
        prompt_ms=prompt_ms,
        predicted_ms=predicted_ms,
        total_ms=wall_ms,
    )


def classify_semantic(
    *,
    backend_base: str,
    model: str,
    user_text: str,
    recent_context: str = "",
) -> ClassifyResult | None:
    """POST directly to backend. One short attempt; bail fast on timeout.

    Returns ClassifyResult with prompt_ms / predicted_ms when llama timings exist.
    """
    user = user_text.strip()
    # Never send long history into classify — prefill dominates on MoE. A short
    # follow-up gets only the previous exchange.
    prompt = user[:500]
    if recent_context and len(user) < _FOLLOWUP_CHARS:
        prompt = (
            f"Context (do not classify):\n{recent_context[:_CONTEXT_CHARS]}\n\n"
            f"Request to classify:\n{prompt}"
        )
    url = f"{backend_base.rstrip('/')}/v1/chat/completions"
    t_deadline = time.perf_counter() + _CLASSIFY_DEADLINE_S
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            try:
                result = _one_classify_call(
                    client=client,
                    url=url,
                    model=model,
                    prompt=prompt,
                    system=CLASSIFIER_SYSTEM,
                    max_tokens=_CLASSIFY_MAX_TOKENS,
                    prefill=True,
                )
                if result is not None:
                    return result
            except Exception:
                pass
            # One non-prefill retry only if we still have budget
            if time.perf_counter() < t_deadline:
                try:
                    return _one_classify_call(
                        client=client,
                        url=url,
                        model=model,
                        prompt=prompt,
                        system=CLASSIFIER_SYSTEM,
                        max_tokens=_CLASSIFY_MAX_TOKENS,
                        prefill=False,
                    )
                except Exception:
                    return None
    except Exception:
        return None
    return None
