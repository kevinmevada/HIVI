"""Instant rule-based classifier: the default HAIE path (no LLM round-trip).

The LLM classifier (semantic.py) cost a second model call before every
ambiguous message: 2–6 s on an offloaded MoE, plus it evicted llama.cpp's
cached conversation from the single slot, so the real reply then re-read the
whole chat. This module decides intent, complexity and length from the text in
well under a millisecond, and covers the cases the old fast/heuristic paths
sent to the LLM: follow-ups ("make it async") inherit the previous request's
intent, writing tasks (emails, cover letters) get their own profile.

Set HIVI_HAIE_SEMANTIC=1 to put the LLM classifier back in front.
"""
from __future__ import annotations

import re
from typing import Any

from .fast_path import is_acknowledgement
from .types import Understanding

_I = re.I

# Code blocks, stack traces and code-shaped text are coding whatever the words say.
_CODE_SHAPE = re.compile(
    r"```|Traceback \(most recent|\b\w+(Error|Exception)\b|^\s*(def|class|import|from|const|let|"
    r"function|public|SELECT|INSERT|UPDATE|CREATE)\b|=>|\)\s*\{|;\s*$|</?\w+>",
    re.M,
)
_PATTERNS: dict[str, re.Pattern[str]] = {
    "coding": re.compile(
        r"\b(code|coding|function|method|class|script|program|bug|debug|fix|error|exception|"
        r"traceback|stack ?trace|compile|refactor|optimi[sz]e|implement|regex|api|endpoint|sql|"
        r"query|database|schema|python|typescript|javascript|java|golang|rust|c\+\+|c#|bash|"
        r"powershell|react|component|html|css|frontend|backend|node|npm|pip|venv|conda|docker|"
        r"container|kubernetes|k8s|git|github|repo|commit|deploy|server|async|await|thread|"
        r"logging|unit ?tests?|pytest|dataframe|pandas|numpy|json|yaml|csv|parser|cli|"
        r"fastapi|flask|django|llama\.cpp|gguf|cuda)\b",
        _I,
    ),
    "architecture": re.compile(
        r"\b(architecture|architect|system design|design (a|an|the|my)\b|high-?level design|"
        r"scalab\w+|microservices?|infrastructure|data model|pipeline design|tech stack)\b",
        _I,
    ),
    "math": re.compile(
        r"\b(solve|calculate|compute|prove|proof|derive|equation|integral|derivative|"
        r"probability|percent(age)?|matrix|eigen\w*|statistics?)\b|\d\s*%\s*of\b|"
        r"\d\s*[-+*/^×÷]\s*\d|\b[a-z]\s*=\s*\d|\d+\s*[a-z]\s*[-+]\s*\d",
        _I,
    ),
    "research": re.compile(
        r"\b(compare|comparison|vs\.?|versus|pros and cons|trade-?offs?|better to use|"
        r"which is better|should i (use|learn|pick|choose|go with)|benchmarks?|alternatives to)\b",
        _I,
    ),
    "explanation": re.compile(
        r"\b(explain|what is|what are|what's|whats|difference between|how does|how do|how is|"
        r"why does|why is|why do|walk me through|meaning of|teach me|help me understand|"
        r"tell me about(?! (yourself|you)\b))\b",
        _I,
    ),
    "writing": re.compile(
        r"\b(email|e-mail|cover letter|letter|essay|rewrite|rephrase|reword|proofread|draft|"
        r"summari[sz]e|summary|translate|caption|tweet|linkedin post|post for|bio|resume|cv|"
        r"message to|reply to|story|poem|blog|paragraph|tone)\b",
        _I,
    ),
    "brainstorm": re.compile(
        r"\b(ideas?|brainstorm|names? for|come up with|suggest(ions)?|inspiration|"
        r"options for|ways to)\b",
        _I,
    ),
}
# When two intents tie, the earlier one leads (decides temperature).
_ORDER = ("coding", "architecture", "math", "research", "explanation", "writing", "brainstorm")
_HARD = re.compile(
    r"\b(complex|concurrency|race condition|deadlock|distributed|production|scalab\w+|"
    r"end-to-end|full app|entire|from scratch|multi-?thread\w*|performance|optimi[sz]e)\b",
    _I,
)
_CASUAL = re.compile(
    r"^\s*(lol|lmao|haha+|hehe|nice|cool|wow|damn|bruh|omg|yay|nope|no|nah|hmm+|ok+|okay)\b",
    _I,
)
_FOLLOWUP = re.compile(
    r"^\s*(now|also|and|but|then|ok(ay)?|yes|yeah|yep|sure|do it|go ahead|make|add|change|"
    r"convert|rewrite|redo|try|again|same|use|switch|remove|rename|shorten|expand|simplify|"
    r"improve|continue|more|another)\b|\b(it|this|that|these|those|them|same|again|instead|"
    r"above|previous)\b",
    _I,
)
_GREETING = re.compile(
    r"^\s*(hi|hey|hello|yo|sup|hiya|good (morning|afternoon|evening|night))\b[\w\s!.?,']{0,30}$",
    _I,
)
# Small talk that happens to contain question words ("what's up", "how are you").
_SMALL_TALK = re.compile(
    r"^\s*(what'?s up|wh?assup|how are you|how'?s it going|how'?s your day|how have you been|"
    r"who are you|what'?s your name|tell me about (yourself|you))\b[\w\s!.?,']{0,30}$",
    _I,
)

_LENGTH = {
    "coding": "medium",
    "architecture": "long",
    "math": "short",
    "research": "long",
    "explanation": "medium",
    "writing": "medium",
    "brainstorm": "medium",
    "chat": "short",
}


def _intents(text: str) -> list[str]:
    scores = {name: len(p.findall(text)) for name, p in _PATTERNS.items()}
    if _CODE_SHAPE.search(text):
        scores["coding"] += 3
    # "write an email" / "draft a letter": writing, even though "write" sounds like code
    if scores["writing"] and not _CODE_SHAPE.search(text):
        scores["coding"] = max(0, scores["coding"] - 1)
    hits = [i for i in _ORDER if scores[i] > 0]
    hits.sort(key=lambda i: -scores[i])  # stable: ties keep _ORDER
    return hits[:2]


def _previous_user(messages: list[Any]) -> str:
    seen_current = False
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        if not seen_current:
            seen_current = True  # skip the message being classified
            continue
        content = msg.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content if isinstance(content, str) else ""
    return ""


def _understanding(text: str, intents: list[str], path: str) -> Understanding:
    n = len(text)
    lead = intents[0]
    complexity = {"architecture": 4, "research": 3, "explanation": 3, "chat": 1}.get(lead, 2)
    # A second intent adds work, except explanation: "how do I…" alongside a task
    # is how people ask, not a harder request.
    if len(intents) > 1 and intents[1] != "explanation" and lead != "explanation":
        complexity += 1
    if n > 300:
        complexity += 1
    if _HARD.search(text):
        complexity += 1
    if lead == "chat" and n > 120:
        complexity = 2
    complexity = max(1, min(5, complexity))
    reasoning = "high" if complexity >= 4 else "medium" if complexity == 3 else "low"
    if lead == "chat":
        reasoning = "none"
    length = _LENGTH.get(lead, "medium")
    if complexity >= 4 and length != "long":
        length = "long"
    return Understanding(
        intents=intents,
        complexity=complexity,
        deliverable={"coding": "code", "writing": "text", "math": "solution"}.get(lead, "reply"),
        reasoning=reasoning,
        length=length,
        ambiguity=0.3,
        confidence=0.7,
        path=path,
    ).normalized()


def classify(user_text: str, messages: list[Any] | None = None) -> Understanding:
    """Always returns an Understanding; never calls a model."""
    text = (user_text or "").strip()
    messages = messages or []
    if not text:
        return _understanding("", ["chat"], "rules")
    if is_acknowledgement(text) or (_GREETING.match(text) and len(text) < 40):
        u = _understanding(text, ["greeting"], "fast")
        u.complexity, u.reasoning, u.length = 1, "none", "short"
        if is_acknowledgement(text):
            u.deliverable, u.expected_output_tokens = "acknowledgement", 64
        return u

    if _SMALL_TALK.match(text):
        return _understanding(text, ["chat"], "fast")

    intents = _intents(text)
    if intents:
        return _understanding(text, intents, "rules")

    # Short and pointing back ("make it async", "yes do it", "same for the other one"):
    # it continues the previous request, so it gets that request's intent.
    if len(text) < 160 and not _CASUAL.match(text) and _FOLLOWUP.search(text):
        prev = _previous_user(messages)
        prev_intents = _intents(prev) if prev else []
        if prev_intents:
            return _understanding(text, prev_intents, "rules-followup")

    return _understanding(text, ["chat"], "rules")
