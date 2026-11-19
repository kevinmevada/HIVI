"""Answers that finish: continue a reply that stopped at its length limit.

The proxy streams a reply; when a part ends with finish_reason "length" (the
answer budget or the context window ran out), it asks the model to go on and
keeps streaming into the same reply, until the model ends the answer itself.
The client just sees one long answer, like Claude.

Two ways to go on:
- "prefill" (llama.cpp): the answer so far is sent as the last, assistant,
  message and llama.cpp resumes the exact text, mid-line if needed.
- "ask" (Colibrì, or a llama.cpp build that refuses prefill on a thinking
  model): the answer so far, then a short user message asking to go on. Models
  tend to reopen the code block they were in; that repeat is removed.

When the answer itself outgrows the window, context_budget slides it: the
question and the end of the answer stay, the start is left out (the user still
has all of it).
"""
from __future__ import annotations

import codecs
import json
import re
from typing import Any

MAX_PARTS = 60   # a runaway model stops eventually; real answers never get near this
CONTINUE_ASK = (
    "Continue exactly where your previous message stopped. Do not repeat anything and do "
    "not restart the code block; just carry on with the next characters."
)
_FENCE = re.compile(r"^[ \t]*```", re.MULTILINE)
_REOPENED_FENCE = re.compile(r"^\s*```[\w+#.\-]*[^\S\n]*(?:[\w.\-/]+\.[A-Za-z0-9]{1,8})?[^\S\n]*\n")


def inside_code(text: str) -> bool:
    return len(_FENCE.findall(text)) % 2 == 1


def join(prev: str, nxt: str, mode: str) -> str:
    """Append a continuation. In "ask" mode drop a code fence the model reopened."""
    if not prev or mode != "ask":
        return prev + nxt
    if inside_code(prev):
        nxt = _REOPENED_FENCE.sub("", nxt, count=1)
    elif not prev.endswith((" ", "\n")):
        nxt = nxt.lstrip(" ")
        nxt = " " + nxt if nxt and not nxt.startswith(("\n", ".", ",", ";", ":", ")")) else nxt
    return prev + nxt


def looping(text: str) -> bool:
    """True when the end of ``text`` is one 40-600 character stretch repeated 5+ times
    (a stuck model), outside code (code repeats lines legitimately)."""
    if not text or inside_code(text):
        return False
    tail = text[-3200:]
    for size in range(40, 601):
        if size * 5 > len(tail):
            break
        unit = tail[-size:]
        if all(tail[-(k + 1) * size: -k * size] == unit for k in range(1, 5)):
            return True
    return False


def continuation_messages(messages: list[Any], answer: str, mode: str) -> list[Any]:
    """The conversation plus the answer so far (and, in "ask" mode, the request to go on)."""
    out = [*messages, {"role": "assistant", "content": answer}]
    if mode == "ask":
        out.append({"role": "user", "content": CONTINUE_ASK})
    return out


def continuation_body(base: dict[str, Any], answer: str, mode: str) -> dict[str, Any]:
    body = dict(base)
    body["messages"] = continuation_messages(list(base.get("messages") or []), answer, mode)
    body["stream"] = True
    body.pop("max_tokens", None)      # the budget gives it the room left
    body.pop("return_progress", None)
    # Keep going with the answer, not new reasoning about it.
    kwargs = body.get("chat_template_kwargs")
    body["chat_template_kwargs"] = {**(kwargs if isinstance(kwargs, dict) else {}),
                                    "enable_thinking": False}
    if "enable_thinking" in body:
        body["enable_thinking"] = False
    body.pop("thinking_budget_tokens", None)
    return body


def context_exceeded(status: int, raw: bytes) -> tuple[int, int] | None:
    """(prompt tokens, window) from llama.cpp's "exceeds the available context size"
    error, else None."""
    if status != 400:
        return None
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return None
    err = data.get("error") if isinstance(data, dict) else None
    if not isinstance(err, dict):
        return None
    text = f"{err.get('type', '')} {err.get('message', '')}".lower()
    if "exceed" not in text or "context" not in text:
        return None
    try:
        return int(err.get("n_prompt_tokens") or 0), int(err.get("n_ctx") or 0)
    except (TypeError, ValueError):
        return None


def prefill_refused(status: int, raw: bytes) -> bool:
    """llama.cpp builds that cannot resume a thinking model's answer say so with a 400."""
    if status < 400:
        return False
    text = raw.decode("utf-8", errors="replace").lower()
    return "prefill" in text or "continue_final_message" in text or "assistant" in text


class SSEEvents:
    """Split an SSE byte stream into complete events (``data: ...`` payloads)."""

    def __init__(self) -> None:
        self._buf = ""
        # Incremental: a character split across two network chunks stays whole.
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, chunk: bytes) -> list[str]:
        self._buf += self._dec.decode(chunk).replace("\r\n", "\n")
        out = []
        while "\n\n" in self._buf:
            block, self._buf = self._buf.split("\n\n", 1)
            data = "\n".join(line[5:].lstrip() for line in block.split("\n")
                             if line.startswith("data:"))
            if data:
                out.append(data)
        return out

    def rest(self) -> list[str]:
        block, self._buf = self._buf, ""
        data = "\n".join(line[5:].lstrip() for line in block.split("\n") if line.startswith("data:"))
        return [data] if data else []
