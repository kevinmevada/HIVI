"""Medium-confidence heuristic planner — skip slow LLM classify when intent is clear."""
from __future__ import annotations

import re

from .types import Understanding

_MULTI = re.compile(
    r"\b("
    r"explain\s+and|and\s+also|then\s+also|compare|versus|\bvs\.?\b|"
    r"brainstorm|research|trade-?off|multi"
    r")\b",
    re.I,
)
_CODE = re.compile(
    r"\b("
    r"write|implement|code|function|class|fix|debug|refactor|optimize|"
    r"python|typescript|javascript|golang|rust|sql|regex|api|endpoint|"
    r"dockerfile|makefile|pytest|unit\s+test|middleware|parser|script"
    r")\b",
    re.I,
)
_EXPLAIN = re.compile(
    r"\b(explain|what\s+is|what\s+are|how\s+does|how\s+do|why\s+does|walk\s+through|summarize)\b",
    re.I,
)
_ARCH = re.compile(
    r"\b(design|architect(?:ure)?|system\s+design|high-?level\s+design|propose\s+a)\b",
    re.I,
)
_MATH = re.compile(
    r"\b(prove|derive|calculate|compute|complexity|big-?o|integral|equation)\b",
    re.I,
)
_RESEARCH = re.compile(
    r"\b(research|survey|compare|versus|\bvs\.?\b|trade-?off|benchmark)\b",
    re.I,
)


def try_heuristic_path(user_text: str) -> Understanding | None:
    """Return Understanding when intent is clear enough to skip LLM classify.

    Used after fast-path and before LLM semantic classify so MoE backends
    (e.g. Qwen3-Coder-30B) do not pay ~3–10s classify latency on routine asks.
    """
    text = (user_text or "").strip()
    if not text or len(text) > 400:
        return None
    if text.count("?") > 1:
        return None
    low = text.lower()
    n = len(text)

    # Common multi-intent patterns (still no LLM)
    if re.search(r"\bexplain\s+and\s+(optimize|implement|fix|refactor)\b", low):
        return Understanding(
            intents=["explanation", "coding"],
            complexity=3,
            deliverable="explain_and_optimize",
            reasoning="medium",
            length="medium",
            ambiguity=0.2,
            confidence=0.8,
            path="heuristic",
            expected_output_tokens=2048,
        ).normalized()
    if re.search(r"\b(implement|write|code).{0,40}\b(and|then).{0,20}\b(explain|compare)\b", low):
        return Understanding(
            intents=["coding", "explanation"],
            complexity=3,
            deliverable="code_and_explain",
            reasoning="medium",
            length="medium",
            ambiguity=0.25,
            confidence=0.75,
            path="heuristic",
            expected_output_tokens=2048,
        ).normalized()
    if re.search(r"\bcompare\b.+\b(moe|dense|model|approach)", low) or (
        "compare" in low and (" vs " in low or "versus" in low)
    ):
        return Understanding(
            intents=["research", "explanation"],
            complexity=3,
            deliverable="comparison",
            reasoning="medium",
            length="long",
            ambiguity=0.25,
            confidence=0.75,
            path="heuristic",
            expected_output_tokens=3072,
        ).normalized()

    # Ambiguous leftovers → semantic
    if _MULTI.search(text):
        return None

    intents: list[str] = []
    if _CODE.search(text):
        intents.append("coding")
    if _EXPLAIN.search(text):
        intents.append("explanation")
    if _ARCH.search(text):
        intents.append("architecture")
    if _MATH.search(text) and "coding" not in intents:
        intents.append("math")
    if _RESEARCH.search(text):
        intents.append("research")

    # Only skip LLM when exactly one clear intent
    if len(intents) != 1:
        return None

    intent = intents[0]
    if intent == "coding":
        hard = bool(
            re.search(r"\b(complex|concurrency|thread|race|deadlock|distributed)\b", low)
        )
        cx = 3 if (n > 120 or hard) else 2
        return Understanding(
            intents=["coding"],
            complexity=cx,
            deliverable="code",
            reasoning="medium" if cx >= 3 else "low",
            length="medium",
            ambiguity=0.25,
            confidence=0.75,
            path="heuristic",
            expected_output_tokens=2048 if cx >= 3 else 1024,
        ).normalized()
    if intent == "explanation":
        return Understanding(
            intents=["explanation"],
            complexity=3,
            deliverable="explanation",
            reasoning="medium",
            length="medium",
            ambiguity=0.25,
            confidence=0.75,
            path="heuristic",
            expected_output_tokens=2048,
        ).normalized()
    if intent == "architecture":
        return Understanding(
            intents=["architecture"],
            complexity=4,
            deliverable="design",
            reasoning="high",
            length="long",
            ambiguity=0.3,
            confidence=0.7,
            path="heuristic",
            expected_output_tokens=4096,
        ).normalized()
    if intent == "math":
        return Understanding(
            intents=["math"],
            complexity=3,
            deliverable="solution",
            reasoning="medium",
            length="medium",
            ambiguity=0.3,
            confidence=0.7,
            path="heuristic",
            expected_output_tokens=1024,
        ).normalized()
    if intent == "research":
        return Understanding(
            intents=["research", "explanation"],
            complexity=3,
            deliverable="comparison",
            reasoning="medium",
            length="long",
            ambiguity=0.3,
            confidence=0.7,
            path="heuristic",
            expected_output_tokens=3072,
        ).normalized()
    return None
