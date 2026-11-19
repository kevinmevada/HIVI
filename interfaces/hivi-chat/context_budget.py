"""Fit each chat request inside the loaded model's context window.

A request is prompt (system + history + new message) plus the answer. When the
total is bigger than the window, llama.cpp rejects the request ("exceeds the
available context size") or cuts the answer off, and Colibrì can stall. This
module runs on every chat request, before it reaches the backend:

1. Estimates the prompt in tokens (conservative, no model call).
2. Drops the oldest turns when they no longer fit, never the system prompt and
   never the newest message.
3. Caps max_tokens to the room left, and switches thinking off when the room
   is too small for reasoning plus an answer.
4. Reports a message that is too long by itself, so the user gets a clear
   answer instead of an error or a hang.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pasted

# Smallest answer worth sending a request for, and the slack for the chat
# template's own tokens and the estimate's error.
MIN_ANSWER = 768
MARGIN = 128
# Below this much room, thinking would eat the whole budget.
THINK_ROOM = 3072
# Per-message template overhead (role markers, separators).
_PER_MESSAGE = 6
# An image costs roughly this many tokens in Qwen-VL / Gemma vision encoders.
IMAGE_TOKENS = 1024

_NON_ASCII = re.compile(r"[^\x00-\x7f]")


def estimate_tokens(text: str) -> int:
    """Tokens for ``text``, erring high.

    English prose is about 4 characters per token and code about 3; Hindi and
    other non-Latin scripts are often one token per character or two. Counting
    ASCII at 3.2 characters per token and every other character as one token
    stays above real counts for all of these, so a "fits" verdict is safe.
    """
    if not text:
        return 0
    other = len(_NON_ASCII.findall(text))
    return int((len(text) - other) / 3.2) + other + 1


def message_tokens(msg: Any) -> int:
    if not isinstance(msg, dict):
        return 0
    content = msg.get("content")
    total = _PER_MESSAGE
    if isinstance(content, str):
        total += estimate_tokens(content)
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                total += estimate_tokens(str(part.get("text") or ""))
            elif part.get("type") in ("image_url", "input_image", "image"):
                total += IMAGE_TOKENS
    reasoning = msg.get("reasoning_content")
    if isinstance(reasoning, str):
        total += estimate_tokens(reasoning)
    return total


def prompt_tokens(messages: list[Any]) -> int:
    return sum(message_tokens(m) for m in messages) + 8


@dataclass
class Fit:
    ctx: int
    prompt: int = 0           # estimated prompt tokens after trimming
    dropped: int = 0          # old messages left out
    max_tokens: int | None = None
    thinking_off: bool = False
    too_long: bool = False    # the newest message alone does not fit
    slid: bool = False        # a long answer's start was left out so it can go on
    newest: int = 0           # estimated tokens of the newest message
    notes: list[str] = field(default_factory=list)

    def headers(self) -> dict[str, str]:
        h = {"X-HIVI-Context": f"{self.prompt}/{self.ctx}"}
        if self.dropped:
            h["X-HIVI-Context-Dropped"] = str(self.dropped)
        if self.thinking_off:
            h["X-HIVI-Context-Thinking"] = "off"
        if self.too_long:
            h["X-HIVI-Context-Too-Long"] = str(self.newest)
        if self.slid:
            h["X-HIVI-Context-Slid"] = "1"
        return h


def _thinking_on(body: dict[str, Any]) -> bool:
    if body.get("enable_thinking") is True:
        return True
    kwargs = body.get("chat_template_kwargs")
    return isinstance(kwargs, dict) and kwargs.get("enable_thinking") is True


def _thinking_off(body: dict[str, Any]) -> None:
    if "enable_thinking" in body:
        body["enable_thinking"] = False
    kwargs = body.get("chat_template_kwargs")
    if isinstance(kwargs, dict):
        body["chat_template_kwargs"] = {**kwargs, "enable_thinking": False}
    body.pop("thinking_budget_tokens", None)
    if body.get("reasoning_effort") not in (None, "none"):
        body["reasoning_effort"] = "low"


def _text_tail(text: str, tokens: int) -> str:
    """Roughly the last ``tokens`` tokens of ``text``, starting at a line break."""
    chars = max(400, int(tokens * 3.0))
    if len(text) <= chars:
        return text
    tail = text[-chars:]
    nl = tail.find("\n")
    return tail[nl + 1:] if 0 <= nl < 300 else tail


def _slide(body: dict[str, Any], system: list[Any], rest: list[Any], budget: int,
           fit: Fit) -> bool:
    """Make room for a long answer to go on: keep the question and the end of the
    answer so far, leave out the start (the chat already shows all of it)."""
    a = next((i for i in range(len(rest) - 1, -1, -1)
              if isinstance(rest[i], dict) and rest[i].get("role") == "assistant"), None)
    if a is None or not isinstance(rest[a].get("content"), str):
        return False
    partial = rest[a]["content"]
    after = rest[a + 1:]                       # "please continue" (engines without prefill)
    q = next((rest[i] for i in range(a - 1, -1, -1)
              if isinstance(rest[i], dict) and rest[i].get("role") == "user"), None)
    base = prompt_tokens(system) + sum(message_tokens(m) for m in after)
    if q is not None and base + message_tokens(q) > budget // 2:
        # A huge question (attachments): keep the user's own words, or its end.
        content = q.get("content")
        text = content if isinstance(content, str) else " ".join(
            str(p.get("text") or "") for p in content or [] if isinstance(p, dict))
        short = pasted.question_of(text).strip() or text
        q = {**q, "content": _text_tail(short, budget // 8)}
    if q is not None:
        base += message_tokens(q)
    free = budget - base
    if free < 2 * MIN_ANSWER:
        return False
    tail = _text_tail(partial, free // 2)      # half for the answer so far, half to go on
    msgs = [*system, *([q] if q is not None else []), {**rest[a], "content": tail}, *after]
    body["messages"] = msgs
    fit.slid = True
    fit.too_long = False
    fit.dropped = len(rest) - (len(msgs) - len(system))
    fit.prompt = prompt_tokens(msgs)
    fit.notes.append("long answer: kept its last part so it can go on")
    return True


def fit_request(body: dict[str, Any], ctx: int | None, *, fill: bool = False,
                continuing: bool = False) -> Fit | None:
    """Trim and cap ``body`` in place so it fits ``ctx``. None when ctx is unknown.

    ``fill``: give the answer all the room left (code and long writing), instead
    of HAIE's estimate. A local model costs nothing per token, and it still stops
    on its own when it is done; the cap only decides where a long file gets cut.
    """
    messages = body.get("messages")
    if not ctx or ctx < 1024 or not isinstance(messages, list) or not messages:
        return None
    fit = Fit(ctx=ctx)
    budget = ctx - MARGIN
    # A continuation should get a real stretch of new text each time, not crumbs.
    want_answer = max(MIN_ANSWER, budget // 4) if continuing else MIN_ANSWER

    system = [m for m in messages if isinstance(m, dict) and m.get("role") == "system"]
    rest = [m for m in messages if not (isinstance(m, dict) and m.get("role") == "system")]
    if not rest:
        return None
    newest = rest[-1]
    fit.newest = message_tokens(newest)
    fixed = prompt_tokens(system) + fit.newest

    if fixed + want_answer > budget:
        # Even with no history the newest message (plus the system prompt) leaves
        # no room for an answer. A long answer that is still going slides instead.
        fit.too_long = True
        fit.prompt = fixed
        if continuing and _slide(body, system, rest, budget, fit):
            body["max_tokens"] = max(MIN_ANSWER, budget - fit.prompt)
            fit.max_tokens = body["max_tokens"]
        return fit

    # Keep the most recent turns that fit, walking backwards.
    history = rest[:-1]
    kept: list[Any] = []
    used = fixed
    for msg in reversed(history):
        cost = message_tokens(msg)
        if used + cost + want_answer > budget:
            break
        kept.append(msg)
        used += cost
    kept.reverse()
    # Don't open the kept history on an assistant turn: the model would see a
    # reply with no question.
    while kept and isinstance(kept[0], dict) and kept[0].get("role") == "assistant":
        used -= message_tokens(kept.pop(0))
    fit.dropped = len(history) - len(kept)
    if fit.dropped:
        body["messages"] = [*system, *kept, newest]
        fit.notes.append(f"left out {fit.dropped} older message(s)")
    fit.prompt = used

    if continuing:
        # A continuation is useless without the answer so far and its question.
        final = body["messages"]
        a = next((m for m in reversed(rest) if isinstance(m, dict) and m.get("role") == "assistant"), None)
        q = None
        if a is not None:
            before_a = rest[:next(i for i, m in enumerate(rest) if m is a)]
            q = next((m for m in reversed(before_a) if isinstance(m, dict) and m.get("role") == "user"), None)
        missing = [m for m in (a, q) if m is not None and not any(x is m for x in final)]
        if missing and _slide(body, system, rest, budget, fit):
            body["max_tokens"] = max(MIN_ANSWER, budget - fit.prompt)
            fit.max_tokens = body["max_tokens"]
            return fit

    room = budget - used
    if _thinking_on(body) and room < THINK_ROOM:
        _thinking_off(body)
        fit.thinking_off = True
        fit.notes.append("thinking off: not enough room")
    asked = body.get("max_tokens")
    if fill or not isinstance(asked, int) or asked <= 0 or asked > room:
        body["max_tokens"] = room
    fit.max_tokens = body["max_tokens"]
    return fit


def too_long_reply(fit: Fit, model_name: str | None, *, window: int | None = None,
                   scale: float = 1.0) -> str:
    """What HIVI says when one message is bigger than the window (in the model's
    own tokens: ``window`` is its real size, ``scale`` the learned token ratio)."""
    window = window or fit.ctx
    newest = int(fit.newest * scale)
    words = int(window * 0.7)  # ~0.75 words per token, rounded down
    who = f"**{model_name}**" if model_name else "The loaded model"
    return (
        f"That message is about **{newest:,} tokens**, and {who} reads at most "
        f"**{window:,} tokens** at once (that has to cover your message, my instructions "
        f"and my answer). I can't take it in one go without losing part of it.\n\n"
        f"Ways to get it through:\n"
        f"- Send it in parts of roughly {words // 2:,} words and tell me what to do once "
        f"they're all in.\n"
        f"- Paste only the part that matters (the error, the section, the function).\n"
        f"- Load a model with a bigger window: MoE models (30B-A3B) run with 32K here.\n"
        f"- Or raise the window yourself: set `HIVI_CTX` (llama.cpp) or `HIVI_COLIBRI_CTX` "
        f"(Colibrì), for example `32768`, and restart HIVI. A bigger window uses more "
        f"memory and can make replies a little slower."
    )


def trim_for_signals(text: str, head: int = 1200, tail: int = 600) -> str:
    """The user's own framing of a long paste: its start and end.

    Mood and memory recall should read what the user said, not a 40-page log
    they pasted (whose 'error' and 'failed' are not the user's mood), and must
    not scan megabytes of text on the server's event loop.
    """
    # Attached documents are material, not the user's words (pasted.py).
    t = pasted.question_of(text or "")
    # Code fences and their contents are material too.
    t = re.sub(r"```.*?(```|$)", " ", t, flags=re.DOTALL)
    if len(t) <= head + tail + 50:
        return t
    return f"{t[:head]}\n…\n{t[-tail:]}"
