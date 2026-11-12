"""Heuristic fast path — ONLY obvious low-complexity turns (V1.1 restricted)."""
from __future__ import annotations

import re

from .types import Understanding

# Fast path is intentionally narrow. Keyword recognition alone is NOT enough.
_GREET = re.compile(r"^\s*(hi|hey|hello|yo|thanks|thank you|ok|okay)\b[\s!.?]*$", re.I)
_SIMPLE_MATH = re.compile(
    r"^\s*(what\s+is\s+|calculate\s+|compute\s+)?\s*\d+\s*[+\-*/x×]\s*\d+\s*\??\s*$",
    re.I,
)
_SIMPLE_CODE = re.compile(
    r"^\s*(write|create|make|implement)\s+(a\s+)?(simple\s+|basic\s+|one[- ]line\s+)?"
    r"(hello\s*world|hi\s*world|fizzbuzz|python\s+hello)"
    r".{0,80}$",
    re.I,
)
_SIMPLE_CODE_ALT = re.compile(
    r"^\s*(write|create)\s+(a\s+)?(hello\s*world).{0,60}$",
    re.I,
)

# Pure acknowledgements ("thank you", "perfect", "works, thanks bro", "badhiya"):
# the whole message is made of these words. Anything else ("great, now add
# logging") is a real request and goes through the normal path.
_ACK_CORE = frozenset("""
thanks thank thankyou thanku thnx thx ty tysm tyvm cheers appreciate appreciated
perfect great nice awesome amazing cool excellent fantastic brilliant superb sweet
good gg noted understood gotcha got works worked working
solved fixed love lovely wow neat perfectly nicely flawlessly
shukriya dhanyavad dhanyawad badhiya badiya mast sahi
""".split())
# "yes" / "ok" alone often answer "want me to…?" and mean "go ahead": only
# filler here, so "ok thanks" is an acknowledgement but "ok" alone is not.
_ACK_FILLER = frozenset("""
ok okay okk k kk alright yes yep yeah yup done accha acha achha theek thik
you u so much very a lot lots again man bro bruh dude buddy boss sir mate it its it's
that that's thats this is was now all fine job work answer one really totally
haha hehe lol bhai yaar hai h ho gaya gya ji the
""".split())


_QUESTION_START = frozenset(
    "is are was were does do did can could should will would what why how which who when where".split()
)


def is_acknowledgement(text: str) -> bool:
    words = re.findall(r"[a-z']+", (text or "").lower())
    if not words or len(words) > 8 or len(text) > 80 or "?" in text:
        return False
    if words[0] in _QUESTION_START:
        return False
    return all(w in _ACK_CORE or w in _ACK_FILLER for w in words) and any(
        w in _ACK_CORE for w in words
    )


# Anything matching these must go to semantic (even if also matches a code verb)
_FORCE_SEMANTIC = re.compile(
    r"\b("
    r"explain\s+and|optimize|refactor(?:\s+and)?|design|architecture|architect|compare|versus|vs\.?|"
    r"debug|concurrency|trading\s+engine|research|brainstorm|explore|"
    r"multi[- ]?intent|and\s+also|then\s+also|"
    r"moe|dense\s+model|system\s+design|plan\s+(a|the)|"
    r"propose\s+a|propose\s+an|walk\s+through|derive|prove|"
    r"why\s+does|how\s+does|how\s+do\b"
    r")\b",
    re.I,
)
# Verbs/phrasing that mean this is a task — never treat as short chat
_TASKISH = re.compile(
    r"\b("
    r"write|fix|code|explain|design|build|implement|debug|compare|"
    r"create|add|parse|refactor|optimize|architect|propose|survey|"
    r"derive|prove|migrate|generate|tune|profile|implement|"
    r"why|how|what\s+is|what\s+are|research|summarize|"
    r"make|change|convert|rewrite|translate|port|update|remove|rename|replace|"
    r"redo|continue|shorten|expand|simplify|improve|extend|test|document"
    r")\b",
    re.I,
)
# A short message that points back at earlier turns depends on context.
_REFERS_BACK = re.compile(
    r"\b(it|this|that|these|those|them|same|again|instead|above|previous)\b",
    re.I,
)
_FENCE = re.compile(r"```")
_MULTI_Q = re.compile(r"\?.*\?")


def extract_last_user(body: dict) -> str:
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text") or ""))
            return " ".join(parts)
    return ""


def try_fast_path(user_text: str, history_len: int = 0) -> Understanding | None:
    """Return Understanding only for clearly simple turns; else None → semantic."""
    text = (user_text or "").strip()
    if not text:
        return Understanding(
            intents=["chat"],
            complexity=1,
            deliverable="reply",
            reasoning="none",
            length="short",
            ambiguity=0.1,
            confidence=0.9,
            path="fast",
        ).normalized()

    n = len(text)

    if is_acknowledgement(text):
        # Short reply only. With a large budget, models re-send their previous
        # answer in full after "thank you" / "perfect".
        return Understanding(
            intents=["greeting"],
            complexity=1,
            deliverable="acknowledgement",
            reasoning="none",
            length="short",
            ambiguity=0.05,
            confidence=0.95,
            path="fast",
            expected_output_tokens=64,
        ).normalized()

    # Hard force-semantic gates (do this BEFORE keyword fast matches)
    if _FORCE_SEMANTIC.search(text):
        return None
    if _MULTI_Q.search(text) or text.count("?") > 1:
        return None
    if _FENCE.search(text) or n > 220 or history_len >= 8:
        return None
    if " and " in text.lower() and any(
        w in text.lower() for w in ("explain", "optimize", "design", "implement", "compare")
    ):
        return None

    if _GREET.match(text) and n < 40:
        return Understanding(
            intents=["greeting"],
            complexity=1,
            deliverable="greeting",
            reasoning="none",
            length="short",
            ambiguity=0.05,
            confidence=0.95,
            path="fast",
        ).normalized()

    if _SIMPLE_MATH.match(text):
        return Understanding(
            intents=["math"],
            complexity=1,
            deliverable="solution",
            reasoning="none",
            length="short",
            ambiguity=0.1,
            confidence=0.9,
            path="fast",
            expected_output_tokens=256,
        ).normalized()

    if _SIMPLE_CODE.match(text) or _SIMPLE_CODE_ALT.match(text):
        return Understanding(
            intents=["coding"],
            complexity=1,
            deliverable="code",
            reasoning="low",
            length="short",
            ambiguity=0.1,
            confidence=0.9,
            path="fast",
            expected_output_tokens=512,
        ).normalized()

    # Short plain chat with no task verbs → fast chat
    # P1 finding: previous verb list was too narrow ("Create…", "Add…", "Why…")
    # silently took the chat fast-path and starved coding/explanation profiles.
    if n < 60 and not _TASKISH.search(text) and not (
        history_len >= 2 and _REFERS_BACK.search(text)
    ):
        return Understanding(
            intents=["chat"],
            complexity=1,
            deliverable="reply",
            reasoning="none",
            length="short",
            ambiguity=0.2,
            confidence=0.8,
            path="fast",
        ).normalized()

    # Everything else → semantic
    return None
