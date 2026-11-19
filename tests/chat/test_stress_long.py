"""Stress test: very long prompts and very long code answers through all of HIVI.

HIVI's server runs as a real process and its runtime launches
fake_llama_server.py exactly as it launches llama-server (same flags, the
window read from the model file). The fake enforces llama.cpp's rules (window
size, 400 on overflow, length cut-offs, prefill, prompt cache) and writes a
deterministic program, so a long answer joined across many parts can be checked
line by line.
"""
from __future__ import annotations

import ast
import json
import os
import re
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

HERE = Path(__file__).resolve().parent
APP = HERE.parents[1] / "interfaces" / "hivi-chat"
FAKE = HERE / "fake_llama_server.py"

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="fake server is launched by shebang")


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gguf(path: Path, ctx: int) -> None:
    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = [s("general.architecture") + struct.pack("<I", 8) + s("qwen3"),
          s("qwen3.context_length") + struct.pack("<I", 4) + struct.pack("<I", ctx)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                     + struct.pack("<Q", len(kv)) + b"".join(kv) + b"\0" * 2048)


@pytest.fixture(scope="module")
def hivi(tmp_path_factory):
    root = tmp_path_factory.mktemp("stress")
    models = root / "Models"
    _gguf(models / "Coding" / "small" / "fake-coder-8k-Q4_K_M.gguf", 8192)
    _gguf(models / "Coding" / "big" / "fake-coder-32k-Q4_K_M.gguf", 32768)
    _gguf(models / "Coding" / "think" / "fake-thinking-8k-Q4_K_M.gguf", 8192)
    port, backend = _port(), _port()
    env = {**os.environ, "HIVI_DATA_ROOT": str(root / "data"), "HIVI_MODELS_ROOT": str(models),
           "HIVI_LLAMA_SERVER": str(FAKE), "HIVI_BACKEND_PORT": str(backend),
           "HIVI_LLAMA_ROUTER": "0", "PYTHONUNBUFFERED": "1"}
    env.pop("HIVI_CTX", None)
    log = open(root / "hivi.log", "w")
    proc = subprocess.Popen([sys.executable, "server.py", "--port", str(port)], cwd=APP, env=env,
                            stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(f"{base}/api/status", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    models_list = httpx.get(f"{base}/api/models", timeout=10).json()["models"]
    ids = {m["id"]: m for m in models_list}
    yield {"base": base, "backend": f"http://127.0.0.1:{backend}", "ids": ids, "log": root / "hivi.log"}
    proc.terminate()
    proc.wait(10)
    log.close()


def _load(h, needle: str) -> dict:
    mid = next(i for i in h["ids"] if needle in i)
    r = httpx.post(f"{h['base']}/api/select", json={"id": mid}, timeout=60)
    assert r.status_code == 200, r.text
    st = httpx.get(f"{h['base']}/api/status").json()
    assert st["status"] == "ready", st
    return st


def _chat(h, messages, *, chat="c1", stream=True, prefill=False):
    headers = {"X-HIVI-Chat": chat, "X-HIVI-Chat-Title": "stress"}
    if prefill:
        headers["X-HIVI-Prefill"] = "1"
    body = {"model": "local", "stream": stream, "messages": messages}
    with httpx.Client(timeout=300) as c:
        if not stream:
            r = c.post(f"{h['base']}/v1/chat/completions", json=body, headers=headers)
            data = r.json()
            return r, data["choices"][0]["message"]["content"], data["choices"][0].get("finish_reason")
        text, finish = "", None
        with c.stream("POST", f"{h['base']}/v1/chat/completions", json=body, headers=headers) as r:
            if r.status_code != 200:
                return r, r.read().decode(), None
            for line in r.iter_lines():
                if not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                obj = json.loads(line[5:])
                ch = (obj.get("choices") or [{}])[0]
                text += (ch.get("delta") or {}).get("content") or ""
                finish = ch.get("finish_reason") or finish
        return r, text, finish


def _stats(h) -> list[dict]:
    return httpx.get(f"{h['backend']}/__stats", timeout=5).json()["requests"]


def _check_program(text: str, lines: int) -> str:
    """The joined answer holds one complete, valid program with every function once."""
    assert "lost its place" not in text
    blocks = re.findall(r"```python app\.py\n(.*?)```", text, re.DOTALL)
    assert len(blocks) == 1, f"expected one code block, got {len(blocks)}"
    code = blocks[0]
    ast.parse(code)
    nums = [int(n) for n in re.findall(r"^def step_\d+_(\d+)\(", code, re.MULTILINE)]
    assert nums == list(range(lines)), f"functions out of order or missing ({len(nums)} of {lines})"
    assert text.rstrip().endswith("That is the whole file.")
    return code


SYSTEM = {"role": "system", "content": "You are HIVI."}


def test_huge_code_answer_in_a_small_window_finishes(hivi):
    """~60K tokens of code through an 8K window: many parts, the window slides, and
    the joined file is complete and valid."""
    st = _load(hivi, "fake-coder-8k")
    assert st["context_size"] == 8192          # read from the model file
    before = len(_stats(hivi))
    r, text, finish = _chat(hivi, [SYSTEM, {"role": "user", "content":
                                            "Write the complete simulator as one file. LINES=2000"}])
    assert r.status_code == 200 and finish == "stop"
    assert r.headers.get("x-hivi-continues") == "server"
    _check_program(text, 2000)
    reqs = _stats(hivi)[before:]
    assert len(reqs) >= 8                       # it really took many parts
    assert all("error" not in q for q in reqs)
    assert all(q["n_prompt"] + (q["predicted_n"] or 0) <= 8192 for q in reqs)
    assert all(q["prefill"] for q in reqs[1:])  # llama.cpp resumed the exact text


def test_non_streaming_clients_get_the_whole_answer_too(hivi):
    _load(hivi, "fake-coder-8k")
    r, text, finish = _chat(hivi, [SYSTEM, {"role": "user", "content": "Write it. LINES=700"}],
                            stream=False, chat="c2")
    assert r.status_code == 200
    _check_program(text, 700)


def test_without_prefill_the_model_is_asked_to_go_on(hivi):
    """A build that refuses prefill: HIVI asks the model to continue and removes the
    code fence it reopens, so the file still comes out as one clean block."""
    _load(hivi, "fake-thinking")
    before = len(_stats(hivi))
    r, text, finish = _chat(hivi, [SYSTEM, {"role": "user", "content": "Write it all. LINES=900"}],
                            chat="c3")
    assert finish == "stop"
    _check_program(text, 900)
    reqs = _stats(hivi)[before:]
    assert any(not q["prefill"] and q["messages"] >= 4 for q in reqs)   # the "continue" turns


def test_long_paste_is_read_whole_and_then_cached(hivi):
    """~17K tokens of attachment in a 32K window: the model sees every line, and the
    follow-up question reuses the cached prompt instead of reading it all again."""
    import pasted
    _load(hivi, "fake-coder-32k")
    log = "\n".join(f"2026-11-18 12:{i // 60 % 60:02d}:{i % 60:02d} worker={i} status=ok latency_ms={i % 997}"
                    for i in range(600))
    log += "\n2026-11-18 13:00:00 worker=9999 status=FAILED reason=db timeout"
    first_q = pasted.wrap([pasted.Doc("Pasted text", log)], "what failed? LINES=20")
    convo = [SYSTEM, {"role": "user", "content": first_q}]
    # the UI reads ahead while the question is typed
    pre = _chat(hivi, [SYSTEM, {"role": "user", "content": pasted.wrap([pasted.Doc("Pasted text", log)])}],
                chat="c4", prefill=True)[0]
    assert pre.status_code == 204 and pre.headers["x-hivi-prefill"] == "read"
    r, text, finish = _chat(hivi, convo, chat="c4")
    assert finish == "stop"
    assert "I read all 601 lines" in text and "worker=9999 status=FAILED" in text
    stats = _stats(hivi)
    real = stats[-1]
    assert real["cache_n"] >= real["n_prompt"] * 0.9, real   # read-ahead already did the reading
    convo += [{"role": "assistant", "content": text},
              {"role": "user", "content": "and which worker was slowest? LINES=10"}]
    _, _, finish = _chat(hivi, convo, chat="c4")
    assert finish == "stop"
    follow = _stats(hivi)[-1]
    assert follow["cache_n"] >= real["n_prompt"] * 0.9, follow   # the attachment was not read again


def test_paste_bigger_than_the_window_gets_a_clear_answer(hivi):
    import pasted
    _load(hivi, "fake-coder-32k")
    doc = "\n".join(f"line {i}: " + "lorem ipsum dolor sit amet " * 4 for i in range(6000))
    r, text, finish = _chat(hivi, [SYSTEM, {"role": "user", "content":
                                            pasted.wrap([pasted.Doc("Pasted text", doc)], "summarize")}],
                            chat="c5")
    assert r.status_code == 200 and "32,768 tokens" in text and "HIVI_CTX" in text


def test_tokenizer_that_counts_more_than_hivi_expects(hivi):
    """Dense symbols count ~3x HIVI's estimate. The backend's own count is learned and the
    request is refitted instead of failing."""
    import pasted
    _load(hivi, "fake-coder-32k")
    dense = ";".join(f"a{i % 10}={i % 7}" for i in range(9000))   # HIVI ~17K, real ~45K tokens
    r, text, finish = _chat(hivi, [SYSTEM, {"role": "user", "content":
                                            pasted.wrap([pasted.Doc("data.txt", dense)], "parse this")}],
                            chat="c6")
    assert r.status_code == 200
    assert "32,768 tokens" in text          # honestly too big, said plainly (no error)
    log = Path(hivi["log"]).read_text()
    assert "budget scaled by" in log
    # after learning, a request that does fit goes straight through
    smaller = ";".join(f"a{i % 10}={i % 7}" for i in range(3000))
    before = len(_stats(hivi))
    r, text, finish = _chat(hivi, [SYSTEM, {"role": "user", "content":
                                            pasted.wrap([pasted.Doc("data.txt", smaller)], "parse this LINES=5")}],
                            chat="c7")
    assert finish == "stop" and "I read all" in text
    assert all("error" not in q for q in _stats(hivi)[before:])


def test_long_chat_history_never_overflows(hivi):
    _load(hivi, "fake-coder-8k")
    convo = [SYSTEM]
    for i in range(30):
        convo += [{"role": "user", "content": f"question {i}: " + "details " * 300},
                  {"role": "assistant", "content": f"answer {i}: " + "words " * 300}]
    convo.append({"role": "user", "content": "now the final file please LINES=150"})
    before = len(_stats(hivi))
    r, text, finish = _chat(hivi, convo, chat="c8")
    assert finish == "stop"
    _check_program(text, 150)
    assert all("error" not in q for q in _stats(hivi)[before:])
