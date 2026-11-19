"""Stress-test HIVI with the model you have loaded: very long prompts, very long code.

Run it while HIVI is open with a model loaded:

    python scripts/stress_long_context.py              # all tests
    python scripts/stress_long_context.py --functions 150 --only big-code

Every test goes through HIVI's own chat endpoint, exactly like the app. Each one
checks that nothing errors, that the answer is complete (not cut off), and what
it cost in time. Generated files land in stress_out/ so you can open them.

Tests
  big-code     one Python file with N functions (default 60); must parse, every
               function present, closed code block. Long answers run through
               HIVI's automatic continuation.
  needle       a log filling ~70% of the window with one failure hidden in the
               middle; the model must find it (reads the whole paste?).
  follow-up    a second question on the same paste; should start much faster
               (the paste stays in llama.cpp's prompt cache).
  over-window  a paste bigger than the window; must get HIVI's clear "too long"
               reply, never an error.
  long-chat    40 earlier turns of history plus a coding request; must not overflow.
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
import time
import uuid
from pathlib import Path

try:
    import httpx
except ImportError:  # pragma: no cover - HIVI's environment has it
    sys.exit("httpx is missing: run this with HIVI's Python (the one that runs server.py).")

OUT = Path("stress_out")
SYSTEM = {"role": "system", "content": "You are HIVI, a local assistant."}


def wrap(name: str, text: str, question: str) -> str:
    """Attachments the way the chat UI sends them (see interfaces/hivi-chat/pasted.py)."""
    return (f'<documents>\n<document index="1">\n<source>{name}</source>\n<document_content>\n'
            f"{text}\n</document_content>\n</document>\n</documents>\n\n{question}")


class Hivi:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.chat = f"stress-{uuid.uuid4().hex[:8]}"

    def status(self) -> dict:
        return httpx.get(f"{self.base}/api/status", timeout=10).json()

    def ask(self, messages: list[dict], *, timeout: float = 3600) -> dict:
        """Stream one reply; returns text, finish, ttft, seconds and HIVI's headers."""
        headers = {"X-HIVI-Chat": self.chat, "X-HIVI-Chat-Title": "stress test"}
        body = {"model": "local", "stream": True, "messages": [SYSTEM, *messages]}
        t0 = time.perf_counter()
        ttft = None
        text, finish = "", None
        with httpx.Client(timeout=httpx.Timeout(timeout, connect=10)) as c, \
                c.stream("POST", f"{self.base}/v1/chat/completions", json=body, headers=headers) as r:
            if r.status_code != 200:
                return {"error": f"HTTP {r.status_code}: {r.read().decode()[:300]}",
                        "seconds": time.perf_counter() - t0}
            hdrs = dict(r.headers)
            for line in r.iter_lines():
                if not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                try:
                    obj = json.loads(line[5:])
                except ValueError:
                    continue
                ch = (obj.get("choices") or [{}])[0]
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece and ttft is None:
                    ttft = time.perf_counter() - t0
                text += piece
                finish = ch.get("finish_reason") or finish
        return {"text": text, "finish": finish, "ttft": ttft, "seconds": time.perf_counter() - t0,
                "headers": hdrs}


def code_blocks(text: str) -> list[tuple[str, str]]:
    return re.findall(r"```[^\n]*?([\w./-]+\.py)?[^\n]*\n(.*?)```", text, re.S)


def log_lines(n: int, needle_at: int, rng: random.Random) -> tuple[str, str]:
    rows = []
    for i in range(n):
        rows.append(f"2026-11-18 12:{i // 60 % 60:02d}:{i % 60:02d}.{rng.randint(0, 999):03d} "
                    f"worker={1000 + i} job={rng.randint(1, 99999)} status=ok latency_ms={rng.randint(3, 900)}")
    needle = f"worker={1000 + needle_at}"
    rows[needle_at] = (f"2026-11-18 12:{needle_at // 60 % 60:02d}:{needle_at % 60:02d}.000 {needle} "
                       "job=4242 status=FAILED reason=disk full on /var/spool")
    return "\n".join(rows), needle


def test_big_code(h: Hivi, functions: int) -> tuple[bool, str, dict]:
    q = (f"Write one complete Python file named inventory_sim.py. It must define exactly "
         f"{functions} top-level functions named step_001 to step_{functions:03d}; each takes an "
         "int and returns an int, has a docstring and at least 3 lines of real logic. End with "
         "a main() that calls all of them and an if __name__ == '__main__' guard. Write every "
         "function in full; do not skip or abbreviate any.")
    r = h.ask([{"role": "user", "content": q}])
    if "error" in r:
        return False, r["error"], r
    blocks = [code for _name, code in code_blocks(r["text"])]
    if not blocks:
        return False, "no code block in the answer", r
    code = max(blocks, key=len)
    OUT.mkdir(exist_ok=True)
    (OUT / "inventory_sim.py").write_text(code, encoding="utf-8")
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"code does not parse: line {e.lineno}: {e.msg}", r
    names = re.findall(r"^def (step_\d{3})\(", code, re.M)
    missing = sorted({f"step_{i:03d}" for i in range(1, functions + 1)} - set(names))
    dupes = len(names) - len(set(names))
    if r["finish"] != "stop":
        return False, f"answer ended with finish={r['finish']}", r
    if missing:
        return False, f"{len(missing)} functions missing (first: {missing[:3]})", r
    lines = code.count("\n") + 1
    note = f"{lines} lines, {len(set(names))} functions" + (f", {dupes} repeated" if dupes else "")
    return True, note, r


def test_needle(h: Hivi, window: int) -> tuple[bool, str, dict, list[dict]]:
    rng = random.Random(7)
    n = max(50, int(window * 0.7 / 30))       # ~30 tokens per log line
    log, needle = log_lines(n, n // 2 + 7, rng)
    msgs = [{"role": "user", "content": wrap("server.log", log,
                                             "One job failed in this log. Which worker, and why? "
                                             "Quote the line.")}]
    r = h.ask(msgs)
    if "error" in r:
        return False, r["error"], r, msgs
    ok = needle in r["text"] and "disk full" in r["text"].lower()
    msgs += [{"role": "assistant", "content": r["text"]}]
    return ok, (f"{n} log lines; found it" if ok else f"{n} log lines; did not find {needle}"), r, msgs


def test_follow_up(h: Hivi, msgs: list[dict], first: dict) -> tuple[bool, str, dict]:
    r = h.ask([*msgs, {"role": "user", "content": "Which worker had the highest latency_ms? Just the line."}])
    if "error" in r:
        return False, r["error"], r
    a, b = first.get("ttft") or 0, r.get("ttft") or 0
    faster = b < a * 0.6 if a > 2 else True
    return faster, f"first answer started after {a:.1f}s, this one after {b:.1f}s", r


def test_over_window(h: Hivi, window: int) -> tuple[bool, str, dict]:
    rng = random.Random(3)
    log, _ = log_lines(int(window * 1.6 / 30), 5, rng)
    r = h.ask([{"role": "user", "content": wrap("huge.log", log, "Summarize this.")}])
    if "error" in r:
        return False, r["error"], r
    ok = "reads at most" in r["text"] and r["finish"] == "stop"
    return ok, "clear 'too long' reply" if ok else "unexpected reply: " + r["text"][:120], r


def test_long_chat(h: Hivi) -> tuple[bool, str, dict]:
    msgs = []
    for i in range(40):
        msgs += [{"role": "user", "content": f"Turn {i}: tell me about topic {i}. " + "context " * 120},
                 {"role": "assistant", "content": f"About topic {i}: " + "explanation " * 160}]
    msgs.append({"role": "user", "content": "Now write a Python function is_prime(n) as a file named primes.py."})
    r = h.ask(msgs)
    if "error" in r:
        return False, r["error"], r
    ok = "def is_prime" in r["text"] and r["finish"] == "stop"
    dropped = r["headers"].get("x-hivi-context-dropped", "0")
    return ok, f"{dropped} old messages left out to fit; answer {'complete' if ok else 'wrong'}", r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:7860")
    ap.add_argument("--functions", type=int, default=60)
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["big-code", "needle", "follow-up", "over-window", "long-chat"])
    args = ap.parse_args()
    h = Hivi(args.base)
    try:
        st = h.status()
    except httpx.HTTPError:
        print(f"HIVI is not answering at {args.base}. Open HIVI first.")
        return 2
    if st.get("status") != "ready" or not st.get("model_id"):
        print("Load a chat model in HIVI first.")
        return 2
    window = int(st.get("context_size") or 8192)
    print(f"Model: {st.get('name')}  ·  engine {st.get('engine')}  ·  window {window:,} tokens\n")
    want = set(args.only or ["big-code", "needle", "follow-up", "over-window", "long-chat"])
    rows = []

    def run(name, fn):
        print(f"→ {name} …", flush=True)
        t0 = time.perf_counter()
        try:
            ok, note, r = fn()
        except Exception as e:  # noqa: BLE001 - report, keep going
            ok, note, r = False, f"crashed: {e!r}", {}
        secs = time.perf_counter() - t0
        ttft = r.get("ttft")
        rows.append((name, ok, secs, ttft, len(r.get("text") or ""), note))
        print(f"  {'PASS' if ok else 'FAIL'}  {secs:6.1f}s  {note}\n", flush=True)

    needle_state: dict = {}
    if "big-code" in want:
        run("big-code", lambda: test_big_code(h, args.functions))
    if "needle" in want or "follow-up" in want:
        def needle():
            ok, note, r, msgs = test_needle(h, window)
            needle_state.update(r=r, msgs=msgs)
            return ok, note, r
        run("needle", needle)
    if "follow-up" in want and needle_state.get("msgs"):
        run("follow-up", lambda: test_follow_up(h, needle_state["msgs"], needle_state["r"]))
    if "over-window" in want:
        run("over-window", lambda: test_over_window(h, window))
    if "long-chat" in want:
        run("long-chat", lambda: test_long_chat(h))

    print(f"{'test':<12} {'result':<6} {'time':>8} {'1st word':>9} {'chars':>8}  notes")
    for name, ok, secs, ttft, chars, note in rows:
        t = f"{ttft:.1f}s" if ttft else "—"
        print(f"{name:<12} {'PASS' if ok else 'FAIL':<6} {secs:7.1f}s {t:>9} {chars:>8,}  {note}")
    if OUT.exists():
        print(f"\nGenerated files: {OUT.resolve()}")
    return 0 if all(r[1] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
