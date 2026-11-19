"""Long prompts: every chat request is fitted inside the loaded model's window."""
from __future__ import annotations

import json
import time

import context_budget as cb
import httpx
import pytest
import server
import soul
from fastapi.responses import Response

LONG = "word " * 40000          # ~200K chars: far past any 16K window
MEDIUM = "line of pasted log text\n" * 900   # ~21K chars, ~7K tokens


def _body(*turns: tuple[str, str], max_tokens: int | None = None, **extra) -> dict:
    msgs = [{"role": "system", "content": "You are HIVI."}]
    msgs += [{"role": r, "content": c} for r, c in turns]
    body = {"model": "m", "stream": True, "messages": msgs, **extra}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return body


# ------------------------------------------------------------------ budget -- #

def test_estimate_errs_high_for_prose_code_and_hindi():
    assert cb.estimate_tokens("") == 0
    prose = "The quick brown fox jumps over the lazy dog. " * 100
    assert cb.estimate_tokens(prose) >= len(prose) / 4          # prose ~4 chars/token
    code = "def f(x):\n    return x * 2  # double\n" * 100
    assert cb.estimate_tokens(code) >= len(code) / 3.5
    hindi = "मुझे कल की मीटिंग के बारे में बताओ " * 50
    assert cb.estimate_tokens(hindi) >= len(hindi) * 0.8         # non-Latin ~1 token/char


def test_small_request_untouched_except_the_cap():
    body = _body(("user", "hi"), max_tokens=4096)
    fit = cb.fit_request(body, 16384)
    assert fit and not fit.too_long and fit.dropped == 0
    assert body["max_tokens"] == 4096 and len(body["messages"]) == 2


def test_max_tokens_capped_to_the_room_left():
    body = _body(("user", MEDIUM), max_tokens=12288)
    fit = cb.fit_request(body, 8192)
    assert not fit.too_long
    assert fit.prompt + body["max_tokens"] <= 8192 - cb.MARGIN
    assert body["max_tokens"] >= cb.MIN_ANSWER


def test_old_turns_dropped_newest_and_system_kept():
    turns = []
    for i in range(12):
        turns += [("user", f"question {i} " + "x" * 3000), ("assistant", f"answer {i} " + "y" * 3000)]
    turns.append(("user", "and now the last question"))
    body = _body(*turns)
    fit = cb.fit_request(body, 8192)
    msgs = body["messages"]
    assert fit.dropped > 0 and not fit.too_long
    assert msgs[0]["role"] == "system" and msgs[-1]["content"] == "and now the last question"
    assert msgs[1]["role"] == "user"   # never opens on an orphan reply
    assert cb.prompt_tokens(msgs) + cb.MIN_ANSWER <= 8192
    assert "answer 11" in msgs[-2]["content"]   # the most recent turns are the ones kept


def test_thinking_switched_off_when_the_room_is_tight():
    filler = "z" * int((8192 - 2500) * 3.2)
    body = _body(("user", filler), chat_template_kwargs={"enable_thinking": True},
                 thinking_budget_tokens=2048)
    fit = cb.fit_request(body, 8192)
    assert fit.thinking_off and body["chat_template_kwargs"]["enable_thinking"] is False
    assert "thinking_budget_tokens" not in body
    colibri = _body(("user", filler), enable_thinking=True, reasoning_effort="high")
    cb.fit_request(colibri, 8192)
    assert colibri["enable_thinking"] is False and colibri["reasoning_effort"] == "low"
    roomy = _body(("user", "short"), enable_thinking=True)
    assert not cb.fit_request(roomy, 16384).thinking_off and roomy["enable_thinking"] is True


def test_a_single_message_bigger_than_the_window_is_reported():
    body = _body(("user", LONG))
    fit = cb.fit_request(body, 16384)
    assert fit.too_long and fit.newest > 16384
    text = cb.too_long_reply(fit, "Qwen3 8B")
    assert "16,384" in text and "HIVI_CTX" in text and "Qwen3 8B" in text
    assert fit.headers()["X-HIVI-Context-Too-Long"] == str(fit.newest)


def test_images_count_toward_the_window():
    img = {"role": "user", "content": [{"type": "text", "text": "what is this"},
                                       {"type": "image_url", "image_url": {"url": "data:,"}}]}
    assert cb.message_tokens(img) >= cb.IMAGE_TOKENS


def test_unknown_window_leaves_the_request_alone():
    body = _body(("user", LONG))
    assert cb.fit_request(body, None) is None and len(body["messages"]) == 2


def test_signals_read_the_users_words_not_the_paste():
    pasted = "please look at this, im stuck\n```\n" + ("ERROR failed crashed broken\n" * 5000) + "```\nthanks"
    trimmed = cb.trim_for_signals(pasted)
    assert "ERROR" not in trimmed and "stuck" in trimmed and "thanks" in trimmed
    assert len(cb.trim_for_signals("a" * 100000)) < 2000
    # a log full of error words is not the user being frustrated
    assert soul.read_mood(cb.trim_for_signals("can you check this log\n```\n" + "FAILED ERROR!!!\n" * 300 + "```")).label == "neutral"


def test_signal_trim_is_fast_on_huge_text():
    t0 = time.perf_counter()
    cb.trim_for_signals("```" + "x" * 2_000_000)
    cb.estimate_tokens("y" * 2_000_000)
    assert time.perf_counter() - t0 < 1.0


# ------------------------------------------------------------------ proxy -- #

def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://hivi")


@pytest.fixture()
def backend(monkeypatch):
    sent: list[dict] = []

    async def stream(request, path, body, extra=None, on_close=None, **kw):
        sent.append({"body": json.loads(body), "headers": dict(extra or {})})
        if on_close:
            on_close()
        return Response(content=b"data: [DONE]\n\n", media_type="text/event-stream",
                        headers=extra or {})

    monkeypatch.setattr(server, "_proxy_stream", stream)
    monkeypatch.setattr(server, "_adapt_chat_body", lambda raw: (raw, {}))
    monkeypatch.setattr(server.model_router, "resolve", lambda name: None)
    monkeypatch.setattr(server, "backend_model_name", lambda: None)
    monkeypatch.setattr(server.STATE, "context_size", 8192)
    monkeypatch.setattr(server.STATE, "engine", "llama")
    monkeypatch.setattr(server.STATE, "name", "Test 8B")
    return sent


async def test_proxy_caps_answer_to_the_room_left(backend):
    async with _client() as c:
        r = await c.post("/v1/chat/completions", json=_body(("user", MEDIUM), max_tokens=12288))
    assert r.status_code == 200
    sent = backend[0]["body"]
    assert sent["max_tokens"] < 12288 and "return_progress" not in sent
    assert r.headers["X-HIVI-Context"].endswith("/8192")
    assert r.headers["X-HIVI-HAIE-Max-Tokens"] == str(sent["max_tokens"])


async def test_proxy_answers_itself_when_one_message_is_too_long(backend):
    async with _client() as c:
        r = await c.post("/v1/chat/completions", json=_body(("user", LONG)))
    assert backend == []          # never reaches the model
    assert r.status_code == 200 and "8,192" in r.text and "HIVI_CTX" in r.text
    assert int(r.headers["X-HIVI-Context-Too-Long"]) > 8192
    assert server.GATE.foreground == 0   # the gate was released




# ------------------------------------------------------- long answers -- #

def test_fill_gives_code_the_whole_window():
    body = _body(("user", "write the full app"), max_tokens=2048)
    fit = cb.fit_request(body, 16384, fill=True)
    assert body["max_tokens"] == fit.max_tokens > 14000


async def test_coding_answers_get_the_room_left(backend, monkeypatch):
    monkeypatch.setattr(server, "_adapt_chat_body",
                        lambda raw: (raw, {"X-HIVI-HAIE-Intents": "coding"}))
    async with _client() as c:
        await c.post("/v1/chat/completions", json=_body(("user", "write a snake game"), max_tokens=2048))
        monkeypatch.setattr(server, "_adapt_chat_body",
                            lambda raw: (raw, {"X-HIVI-HAIE-Intents": "chat"}))
        await c.post("/v1/chat/completions", json=_body(("user", "how are you"), max_tokens=1024))
    assert backend[0]["body"]["max_tokens"] > 7000      # 8192 window minus the prompt
    assert backend[1]["body"]["max_tokens"] == 1024     # chat keeps HAIE's budget


async def test_continuation_resumes_the_partial_answer(backend, monkeypatch, tmp_path):
    import personal_memory as pm
    monkeypatch.setenv("HIVI_DATA_ROOT", str(tmp_path / "data"))
    store = pm.MemoryStore(tmp_path / "data" / "memory" / "memory.db")
    pm.set_store(store)
    scheduled = []
    monkeypatch.setattr(server, "_schedule_extraction", lambda *a: scheduled.append(a))
    partial = "```python\ndef main():\n    for i in range(10):\n"
    body = _body(("user", "write a long script"), ("assistant", partial), max_tokens=512)
    async with _client() as c:
        r = await c.post("/v1/chat/completions", json=body,
                         headers={"X-HIVI-Continue": "1", "X-HIVI-Chat": "c9"})
    assert r.status_code == 200
    sent = backend[0]["body"]
    assert sent["messages"][-1] == {"role": "assistant", "content": partial}   # llama.cpp prefill
    assert sent["max_tokens"] > 7000      # a continuation always gets the room left
    assert scheduled == []                # not mined for memory
    store.close()
    pm.set_store(None)


# --------------------------------------------- answers that never stop -- #

def _gguf(path, arch="qwen3", ctx=40960, *, vocab=50):
    """A minimal GGUF header: architecture, a vocab array, then context_length."""
    import struct

    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = [s("general.architecture") + struct.pack("<I", 8) + s(arch),
          s("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<I", 8)
          + struct.pack("<Q", vocab) + b"".join(s(f"tok{i}") for i in range(vocab)),
          s(f"{arch}.context_length") + struct.pack("<I", 4) + struct.pack("<I", ctx)]
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                     + struct.pack("<Q", len(kv)) + b"".join(kv))
    return path


def test_gguf_context_length_is_read_from_the_header(tmp_path):
    import gguf_context
    f = _gguf(tmp_path / "m.gguf", ctx=40960, vocab=2000)
    assert gguf_context.read_context_length(f) == 40960
    assert gguf_context.trained_context(f) == 40960
    (tmp_path / "x.gguf").write_bytes(b"nope")
    assert gguf_context.trained_context(tmp_path / "x.gguf") is None
    assert gguf_context.trained_context(tmp_path / "missing.gguf") is None


def test_window_is_big_but_never_past_training(tmp_path, monkeypatch):
    import registry
    import runtime
    monkeypatch.delenv("HIVI_CTX", raising=False)

    def entry(path):
        return registry.ModelEntry(id="m", name="x", path=str(path), quant="Q4_K_M", params="?",
                                   size_gb=1.0, engine="llama", engine_reason="")
    long_trained = _gguf(tmp_path / "qwen3-8b.gguf", ctx=131072)
    short_trained = _gguf(tmp_path / "old-7b.gguf", ctx=4096)
    moe = _gguf(tmp_path / "Qwen3.6-35B-A3B.gguf", arch="qwen35moe", ctx=262144)
    assert runtime.context_for(entry(long_trained)) == runtime.DEFAULT_CTX == 32768
    assert runtime.context_for(entry(short_trained)) == 4096
    assert runtime.context_for(entry(moe)) == runtime.MOE_CTX == 65536


def test_a_long_answer_slides_instead_of_stopping():
    question = "write the complete game engine in one file"
    partial = "".join(f"def f{i}(x):\n    return x * {i}\n\n" for i in range(3000))   # ~30K tokens
    body = _body(("user", "old question " * 400), ("assistant", "old answer " * 400),
                 ("user", question), ("assistant", partial))
    fit = cb.fit_request(body, 16384, continuing=True)
    msgs = body["messages"]
    assert not fit.too_long and fit.slid
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[1]["content"] == question                      # the question stays
    assert partial.endswith(msgs[2]["content"])                # the end of the answer so far
    tail = msgs[2]["content"]
    assert partial[len(partial) - len(tail) - 1] == "\n"      # cut at a line break
    assert cb.prompt_tokens(msgs) + body["max_tokens"] <= 16384
    assert body["max_tokens"] >= 4000                          # a real stretch to go on with
    assert fit.headers()["X-HIVI-Context-Slid"] == "1"


def test_ask_mode_keeps_the_answer_it_continues():
    question = "write a long report"
    partial = "Paragraph text that goes on. " * 4000
    body = _body(("user", question), ("assistant", partial),
                 ("user", "Continue exactly where your previous message stopped."))
    fit = cb.fit_request(body, 16384, continuing=True)
    roles = [m["role"] for m in body["messages"]]
    assert fit.slid and roles == ["system", "user", "assistant", "user"]
    assert body["messages"][1]["content"] == question


def test_huge_attachment_question_is_shrunk_when_sliding():
    import pasted
    doc = pasted.wrap([pasted.Doc("Pasted text", "log line\n" * 30000)], "refactor this whole thing")
    body = _body(("user", doc), ("assistant", "x = 1\n" * 20000))
    fit = cb.fit_request(body, 16384, continuing=True)
    assert fit.slid and body["messages"][1]["content"] == "refactor this whole thing"
