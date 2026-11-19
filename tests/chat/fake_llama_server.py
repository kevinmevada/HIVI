#!/usr/bin/env python3
"""A stand-in for llama-server that enforces the rules HIVI has to live with.

Used by the long-context stress test (test_stress_long.py), launched by HIVI's
own runtime exactly like the real binary (same flags). It does not run a
model; it plays one whose output can be checked byte for byte:

- Tokens are counted like a real BPE tokenizer would (words, symbols and
  whitespace runs; non-Latin text at about 2 characters per token).
- The window (-c) is enforced: a prompt that does not fit gets llama.cpp's
  400 "exceeds the available context size" error with n_prompt_tokens/n_ctx,
  and generation stops with finish_reason "length" at max_tokens or at the end
  of the window.
- A last assistant message is continued, not answered (assistant prefill).
- One slot with a prompt cache: reused prefix tokens are reported as cache_n.

The "model" answers every request with one long, deterministic program (its
size set by "LINES=<n>" in the question, default 300), starting with what it
read in any attached documents. When continuing, it finds where the answer
so far ends in that program and carries on from there, so a correctly joined
answer equals the program exactly.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_PIECE = re.compile(r" ?[A-Za-z_]+| ?\d{1,3}| ?[^\sA-Za-z_\d]|\s+")
STATS: list[dict] = []
_LOCK = threading.Lock()
_SLOT: dict[str, list[str]] = {"tokens": []}


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for p in _PIECE.findall(text):
        if p.isascii():
            while len(p) > 8:          # long identifiers split like BPE does
                out.append(p[:6])
                p = p[6:]
            out.append(p)
        else:
            out.extend(p[i:i + 2] for i in range(0, len(p), 2))
    return out


def render(messages: list[dict], prefill: bool) -> str:
    text = ""
    for i, m in enumerate(messages):
        c = m.get("content")
        if isinstance(c, list):
            c = "".join(str(p.get("text") or "") for p in c if isinstance(p, dict))
        last = i == len(messages) - 1
        if prefill and last:
            text += f"<|im_start|>assistant\n{c}"
        else:
            text += f"<|im_start|>{m.get('role')}\n{c}<|im_end|>\n"
    if not prefill:
        text += "<|im_start|>assistant\n"
    return text


def program(lines: int, seed: int) -> str:
    rows = ["import math", "import sys", "", ""]
    for i in range(lines):
        rows.append(f"def step_{seed}_{i}(x):")
        rows.append(f"    return math.floor(x * {i + 1} + {seed % 97}) % 1000003")
        rows.append("")
    rows += ["", "if __name__ == \"__main__\":",
             f"    print(sum(step_{seed}_{i}(7) for i in range(3)))"]
    return "\n".join(rows) + "\n"


def target_answer(messages: list[dict]) -> str:
    users = [m for m in messages if m.get("role") == "user"]
    last = users[-1] if users else {"content": ""}
    c = last.get("content")
    if isinstance(c, list):
        c = "".join(str(p.get("text") or "") for p in c if isinstance(p, dict))
    c = str(c or "")
    m = re.search(r"LINES=(\d+)", c)
    lines = int(m.group(1)) if m else 300
    read = ""
    d = re.search(r"<document_content>\n(.*?)\n</document_content>", c, re.DOTALL)
    if d:
        doc = d.group(1).split("\n")
        read = f"I read all {len(doc)} lines of the attachment; the last one is `{doc[-1][:60]}`.\n\n"
    question = c.split("</documents>")[-1]      # the same however much of the attachment is left
    seed = sum(map(ord, re.sub(r"\s+", "", question)[-40:])) % 9973
    return f"{read}Here is the complete program.\n\n```python app.py\n{program(lines, seed)}```\n\nThat is the whole file.\n"


def answer_from(messages: list[dict], prefill: bool) -> tuple[str, str]:
    """(full target answer, the part still to write)."""
    convo = [m for m in messages if m.get("role") != "system"]
    if prefill:
        partial = str(convo[-1].get("content") or "")
        target = target_answer(convo[:-1])
    elif convo and convo[-1].get("role") == "user" and "Continue exactly where" in str(convo[-1].get("content")):
        partial = str(convo[-2].get("content") or "")
        target = target_answer(convo[:-2])
        # models asked to go on tend to reopen the fence they were in: do that too
        rest = _rest(target, partial)
        if partial.count("```") % 2 == 1:
            rest = "```python\n" + rest
        return target, rest
    else:
        return (t := target_answer(convo)), t
    return target, _rest(target, partial)


def _rest(target: str, partial: str) -> str:
    if not partial:
        return target
    probe = partial[-300:]
    i = target.rfind(probe)
    if i < 0:
        return "\n[the model lost its place]\n"
    return target[i + len(probe):]


class Handler(BaseHTTPRequestHandler):
    ctx = 8192
    model = "fake"

    def log_message(self, *a):
        pass

    def _json(self, code: int, obj) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._json(200, {"status": "ok"})
        if self.path.startswith("/v1/models"):
            return self._json(200, {"data": [{"id": self.model, "object": "model"}]})
        if self.path.startswith("/props"):
            return self._json(200, {"default_generation_settings": {"n_ctx": self.ctx}})
        if self.path.startswith("/__stats"):
            with _LOCK:
                return self._json(200, {"requests": STATS, "ctx": self.ctx})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if not self.path.endswith("chat/completions"):
            return self._json(404, {"error": "not found"})
        messages = body.get("messages") or []
        prefill = bool(messages) and messages[-1].get("role") == "assistant"
        if prefill and "thinking" in self.model:
            # older llama.cpp builds refuse to resume a thinking model's answer
            return self._json(400, {"error": {"code": 400, "type": "invalid_request_error",
                                              "message": "Assistant response prefill is incompatible with enable_thinking."}})
        prompt = tokenize(render(messages, prefill))
        n_prompt = len(prompt)
        stat = {"n_prompt": n_prompt, "prefill": prefill, "max_tokens": body.get("max_tokens"),
                "stream": bool(body.get("stream")), "messages": len(messages)}
        if n_prompt >= self.ctx:
            stat["error"] = "exceed_context_size_error"
            with _LOCK:
                STATS.append(stat)
            return self._json(400, {"error": {
                "code": 400, "type": "exceed_context_size_error",
                "message": "the request exceeds the available context size, try increasing it",
                "n_prompt_tokens": n_prompt, "n_ctx": self.ctx}})
        with _LOCK:
            old = _SLOT["tokens"]
            cache = 0
            for a, b in zip(old, prompt, strict=False):
                if a != b:
                    break
                cache += 1
        _target, rest = answer_from(messages, prefill)
        out_tokens = tokenize(rest)
        limit = self.ctx - n_prompt
        if isinstance(body.get("max_tokens"), int) and body["max_tokens"] > 0:
            limit = min(limit, body["max_tokens"])
        gen = out_tokens[:limit]
        finish = "stop" if len(gen) == len(out_tokens) else "length"
        with _LOCK:
            _SLOT["tokens"] = prompt + gen
            stat.update(cache_n=cache, predicted_n=len(gen), finish=finish)
            STATS.append(stat)
        timings = {"cache_n": cache, "prompt_n": n_prompt - cache, "predicted_n": len(gen),
                   "predicted_per_second": 40.0}
        text = "".join(gen)
        if not body.get("stream"):
            return self._json(200, {"choices": [{"index": 0, "finish_reason": finish,
                                                 "message": {"role": "assistant", "content": text}}],
                                     "timings": timings,
                                     "usage": {"prompt_tokens": n_prompt, "completion_tokens": len(gen)}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(obj) -> None:
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())

        send({"choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]})
        for i in range(0, len(gen), 16):
            send({"choices": [{"index": 0, "delta": {"content": "".join(gen[i:i + 16])},
                               "finish_reason": None}]})
        send({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "timings": timings})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main() -> int:
    argv = sys.argv[1:]
    if "--help" in argv:
        print("fake llama-server for HIVI tests (no router: --models-preset unsupported)")
        return 0
    # llama.cpp flags like -ctk would fool argparse's "-c" prefix matching: read exact flags.
    opts = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("-")}
    args = argparse.Namespace(m=opts.get("-m", "fake.gguf"), host=opts.get("--host", "127.0.0.1"),
                              port=int(opts.get("--port", "8080")), c=int(opts.get("-c", "8192")))
    Handler.ctx = args.c
    Handler.model = args.m.replace("\\", "/").split("/")[-1].removesuffix(".gguf")
    ThreadingHTTPServer.daemon_threads = True
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"fake llama-server on {args.host}:{args.port}, ctx {args.c}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    time.sleep(0)
    sys.exit(main())
