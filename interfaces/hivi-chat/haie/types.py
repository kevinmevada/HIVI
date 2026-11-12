"""Shared HAIE types."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VALID_INTENTS = frozenset(
    {
        "greeting",
        "coding",
        "explanation",
        "architecture",
        "math",
        "research",
        "brainstorm",
        "chat",
    }
)
REASONING_LEVELS = ("none", "low", "medium", "high")
LENGTH_LEVELS = ("short", "medium", "long", "xlarge")


@dataclass
class Understanding:
    intents: list[str] = field(default_factory=lambda: ["chat"])
    complexity: int = 2  # 1–5
    deliverable: str = "reply"
    reasoning: str = "low"  # none|low|medium|high
    length: str = "medium"  # short|medium|long|xlarge
    ambiguity: float = 0.3
    confidence: float = 0.5
    path: str = "fallback"  # fast|semantic|fallback
    expected_output_tokens: int | None = None  # classifier estimate when present

    def normalized(self) -> Understanding:
        intents = [i for i in self.intents if i in VALID_INTENTS]
        if not intents:
            intents = ["chat"]
        complexity = max(1, min(5, int(self.complexity or 2)))
        reasoning = self.reasoning if self.reasoning in REASONING_LEVELS else "low"
        length = self.length if self.length in LENGTH_LEVELS else "medium"
        ambiguity = max(0.0, min(1.0, float(self.ambiguity or 0.0)))
        confidence = max(0.0, min(1.0, float(self.confidence or 0.0)))
        expected = self.expected_output_tokens
        if expected is not None:
            expected = max(64, min(8192, int(expected)))
        return Understanding(
            intents=intents,
            complexity=complexity,
            deliverable=(self.deliverable or "reply")[:80],
            reasoning=reasoning,
            length=length,
            ambiguity=ambiguity,
            confidence=confidence,
            path=self.path,
            expected_output_tokens=expected,
        )


@dataclass
class SampleParams:
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    thinking_budget_tokens: int
    reasoning_effort: str  # low|medium|high|none
    enable_thinking: bool | None = None  # None = leave the model/server default


@dataclass
class PlanMeta:
    path: str
    intents: list[str]
    complexity: int
    deliverable: str
    reasoning: str
    length: str
    ambiguity: float
    confidence: float
    params: dict[str, Any]
    engine: str | None = None
    classify_ms: float | None = None
    classify_prompt_ms: float | None = None
    classify_predicted_ms: float | None = None

    def to_headers(self) -> dict[str, str]:
        h = {
            "X-HIVI-HAIE-Path": self.path,
            "X-HIVI-HAIE-Intents": ",".join(self.intents),
            "X-HIVI-HAIE-Complexity": str(self.complexity),
            "X-HIVI-HAIE-Deliverable": self.deliverable,
            "X-HIVI-HAIE-Reasoning": self.reasoning,
            "X-HIVI-HAIE-Length": self.length,
            "X-HIVI-HAIE-Confidence": f"{self.confidence:.2f}",
            "X-HIVI-HAIE-Temp": str(self.params.get("temperature", "")),
            "X-HIVI-HAIE-Max-Tokens": str(self.params.get("max_tokens", "")),
            "X-HIVI-HAIE-Think": str(self.params.get("thinking_budget_tokens", "")),
            "X-HIVI-HAIE-Thinking": {True: "on", False: "off"}.get(
                self.params.get("enable_thinking"), ""
            ),
            # Compat aliases for older UI snippets
            "X-HIVI-Adaptive-Bucket": ",".join(self.intents) or self.path,
            "X-HIVI-Adaptive-Temp": str(self.params.get("temperature", "")),
            "X-HIVI-Adaptive-Max-Tokens": str(self.params.get("max_tokens", "")),
            "X-HIVI-Adaptive-Think": str(self.params.get("thinking_budget_tokens", "")),
        }
        if self.classify_ms is not None:
            h["X-HIVI-HAIE-Classify-Ms"] = str(int(self.classify_ms))
        if self.classify_prompt_ms is not None:
            h["X-HIVI-HAIE-Classify-Prompt-Ms"] = str(int(self.classify_prompt_ms))
        if self.classify_predicted_ms is not None:
            h["X-HIVI-HAIE-Classify-Gen-Ms"] = str(int(self.classify_predicted_ms))
        return h

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
