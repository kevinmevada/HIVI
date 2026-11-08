"""Extract user-facing text from gpt-oss Harmony / think-tagged model output."""
from __future__ import annotations

import re

_FINAL = re.compile(
    r"<\|channel\|>final<\|message\|>([\s\S]*?)(?=(?:<\|end\|>|<\|start\|>|$))",
    re.I,
)
_ANALYSIS = re.compile(
    r"<\|channel\|>analysis<\|message\|>([\s\S]*?)(?=(?:<\|end\|>|<\|start\|>|<\|channel\|>|$))",
    re.I,
)
_COMMENTARY = re.compile(
    r"<\|channel\|>commentary\b[^|]*<\|message\|>([\s\S]*?)(?=(?:<\|end\|>|<\|start\|>|<\|channel\|>|$))",
    re.I,
)
_TAGS = re.compile(r"<\|[^|>]+\|>")
_THINK_OPEN = re.compile(r"<think>", re.I)
_THINK_CLOSE = re.compile(r"</think>", re.I)
_THINK_BLOCK = re.compile(r"<think>[\s\S]*?</think>", re.I)


def extract_final_text(text: str | None) -> str:
    """Return only the final/user-facing answer; never analysis channel body."""
    if not text:
        return ""
    t = str(text)
    m = _FINAL.search(t)
    if m:
        return _clean(m.group(1))
    # Drop analysis / commentary; keep any remaining plain text
    t = _ANALYSIS.sub("", t)
    t = _COMMENTARY.sub("", t)
    t = _THINK_BLOCK.sub("", t)
    t = _THINK_OPEN.sub("", t)
    t = _THINK_CLOSE.sub("", t)
    t = _TAGS.sub("", t)
    return _clean(t)


def _clean(t: str) -> str:
    t = _TAGS.sub("", t)
    return t.strip()
