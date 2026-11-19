"""Long pastes as documents (Claude.ai style), a cacheable prompt, and read-ahead."""
from __future__ import annotations

import json

import chat_history
import context_budget as cb
import httpx
import memory_engine
import pasted
import personal_memory as pm
import pytest
import server
from fastapi.responses import Response
from haie import plan_request

RealClient = httpx.AsyncClient   # the tests patch httpx.AsyncClient for the backend
CODE = "def handler(event):\n    raise ValueError('bad')\n" * 400
LOG = "2026-11-18 ERROR worker failed, connection refused!!!\n" * 300


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVI_DATA_ROOT", str(tmp_path / "data"))
    st = pm.MemoryStore(tmp_path / "data" / "memory" / "memory.db")
    pm.set_store(st)
    monkeypatch.setattr(memory_engine, "EXTRACTOR", memory_engine.Extractor())
    monkeypatch.setattr(memory_engine, "RECALL", memory_engine.Recall())
    server._PROFILES.clear()
    yield st
    st.close()
    pm.set_store(None)


def test_wrap_and_split_round_trip():
    docs = [pasted.Doc("Pasted text", LOG), pasted.Doc("app.py", CODE)]
    text = pasted.wrap(docs, "why does it fail?")
    assert text.startswith("<documents>\n<document index=\"1\">\n<source>Pasted text</source>")
    back, question = pasted.split(text)
    assert [d.name for d in back] == ["Pasted text", "app.py"]
    assert back[0].text == LOG and back[1].text == CODE and question == "why does it fail?"
    only = pasted.wrap(docs[:1])
    assert pasted.split(only)[1] == "" and len(pasted.split(only)[0]) == 1
    assert pasted.split("just a question") == ([], "just a question")
    # the read-ahead text is an exact prefix of the real message
    assert text.startswith(pasted.wrap(docs)) and pasted.block_end(text) == len(pasted.wrap(docs))


def test_haie_reads_the_question_not_the_attachment():
    body = {"messages": [{"role": "user", "content": pasted.wrap(
        [pasted.Doc("Pasted text", CODE)], "summarize this in two lines")}]}
    _, meta = plan_request(body, engine="llama", backend_base="http://127.0.0.1:9", model_id=None)
    assert "writing" in meta.intents and "coding" not in meta.intents
    doc_only = {"messages": [{"role": "user", "content": pasted.wrap([pasted.Doc("a.py", CODE)])}]}
    _, meta = plan_request(doc_only, engine="llama", backend_base="http://127.0.0.1:9", model_id=None)
    assert "coding" in meta.intents and meta.length != "short"


def test_mood_memory_and_titles_ignore_the_attachment():
    msg = pasted.wrap([pasted.Doc("Pasted text", LOG)], "what does this log mean")
    assert "ERROR" not in cb.trim_for_signals(msg)
    assert chat_history.make_title([{"role": "user", "content": msg}]) == "what does this log mean"
    assert chat_history.make_title([{"role": "user", "content": pasted.wrap(
        [pasted.Doc("server.log", LOG)])}]) == "server.log"


async def test_a_bare_attachment_is_not_mined_for_personal_facts(store):
    calls = []

    class Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def post(self, *a, **k):
            calls.append(k)
            raise httpx.ConnectError("no")

    memory_engine.httpx.AsyncClient, real = Client, memory_engine.httpx.AsyncClient
    try:
        ev = await memory_engine.EXTRACTOR.run(
            backend_base="http://x", model="m", user_text=pasted.wrap([pasted.Doc("Pasted text", LOG)]),
            reply_text="It is a log.", chat_id="c", chat_title="t")
    finally:
        memory_engine.httpx.AsyncClient = real
    assert ev is None and calls == []


CTX = {"hivi_chat": True, "chat_id": "c1", "title": None, "temporary": False, "internal": False}


async def test_system_prompt_stays_identical_turn_to_turn(store):
    store.update_settings({"user_name": "Lakshya"})
    store.add("Roommate is Ayush", category="people")
    store.add("Deadline for the gait paper is Friday", category="projects")
    first = {"messages": [{"role": "user", "content": pasted.wrap(
        [pasted.Doc("Pasted text", LOG)], "deadline friday hai, what is broken? im stressed")}]}
    out1, h1 = await server._prepare_chat(first, CTX)
    turn = out1["messages"][1]["content"]
    # documents first, then HIVI's note, then the user's own words last
    assert turn.startswith(pasted.wrap([pasted.Doc("Pasted text", LOG)]))
    assert turn.index("</documents>") < turn.index("<hivi_context>") < turn.index("what is broken")
    assert "Right now:" in turn and h1["X-HIVI-Mood"].startswith("stressed")
    # a fact learned in the background mid-chat does not change this chat's system prompt
    store.add("Prefers PowerShell one-liners", category="preferences", origin="auto")
    second = {"messages": [*first["messages"], {"role": "assistant", "content": "The db is down."},
                           {"role": "user", "content": "ok thanks"}]}
    out2, _ = await server._prepare_chat(second, CTX)
    assert out2["messages"][0]["content"] == out1["messages"][0]["content"]
    # and the earlier turn goes out exactly as stored (no note), so it stays cached
    assert out2["messages"][1]["content"] == first["messages"][0]["content"]
    # an explicit edit on the memory screen does show up on the next message
    async with RealClient(transport=httpx.ASGITransport(app=server.app), base_url="http://h") as c:
        await c.post("/api/memory", json={"text": "Lives in Chicago"})
    out3, _ = await server._prepare_chat(second, CTX)
    assert "Lives in Chicago" in out3["messages"][0]["content"]


# ------------------------------------------------------------- read-ahead -- #

@pytest.fixture()
def llama_ready(monkeypatch):
    monkeypatch.setattr(server.STATE, "engine", "llama")
    monkeypatch.setattr(server.STATE, "status", "ready")
    monkeypatch.setattr(server.STATE, "model_id", "m")
    monkeypatch.setattr(server.STATE, "name", "Test 8B")
    monkeypatch.setattr(server.STATE, "context_size", 16384)
    monkeypatch.setattr(server, "_current_backend_model", lambda: "m")
    monkeypatch.setattr(server.model_router, "resolve", lambda name: None)
    monkeypatch.setattr(server, "backend_model_name", lambda: None)
    monkeypatch.setattr(server, "_adapt_chat_body", lambda raw: (raw, {}))
    sent = {"prefill": [], "chat": []}

    class Backend:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def post(self, url, json=None, **k):
            sent["prefill"].append(json)
            return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    async def stream(request, path, body, extra=None, on_close=None, **kw):
        sent["chat"].append(json.loads(body))
        if on_close:
            on_close()
        return Response(content=b"data: [DONE]\n\n", media_type="text/event-stream")

    monkeypatch.setattr(server.httpx, "AsyncClient", Backend)
    monkeypatch.setattr(server, "_proxy_stream", stream)
    return sent


HEADERS = {"X-HIVI-Chat": "c1", "X-HIVI-Chat-Title": "t"}


async def test_read_ahead_shares_its_prefix_with_the_real_message(store, llama_ready):
    store.add("Roommate is Ayush", category="people")
    history = [{"role": "system", "content": "You are HIVI."},
               {"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hey."}]
    docs = [pasted.Doc("Pasted text", LOG)]
    async with RealClient(transport=httpx.ASGITransport(app=server.app), base_url="http://h") as c:
        r = await c.post("/v1/chat/completions", headers={**HEADERS, "X-HIVI-Prefill": "1"}, json={
            "model": "m", "stream": True, "messages": [*history, {"role": "user", "content": pasted.wrap(docs)}]})
        assert r.status_code == 204 and r.headers["X-HIVI-Prefill"] == "read"
        await c.post("/v1/chat/completions", headers=HEADERS, json={
            "model": "m", "stream": True,
            "messages": [*history, {"role": "user", "content": pasted.wrap(docs, "why is the worker failing")}]})
    pre, real = llama_ready["prefill"][0], llama_ready["chat"][0]
    assert pre["max_tokens"] == 1 and pre["stream"] is False and pre["cache_prompt"] is True
    # identical system prompt and history, and the real message extends the read-ahead one
    assert pre["messages"][:-1] == real["messages"][:-1]
    assert real["messages"][-1]["content"].startswith(pre["messages"][-1]["content"])
    assert server.GATE.background == 0


async def test_read_ahead_skipped_when_it_cannot_help(store, llama_ready, monkeypatch):
    body = {"model": "m", "messages": [{"role": "user", "content": pasted.wrap([pasted.Doc("p", LOG)])}]}
    monkeypatch.setattr(server.STATE, "engine", "colibri")   # no shared prompt cache to warm
    async with RealClient(transport=httpx.ASGITransport(app=server.app), base_url="http://h") as c:
        r = await c.post("/v1/chat/completions", headers={**HEADERS, "X-HIVI-Prefill": "1"}, json=body)
        assert r.status_code == 204 and r.headers["X-HIVI-Prefill"] == "skipped"
        monkeypatch.setattr(server.STATE, "engine", "llama")
        monkeypatch.setattr(server.STATE, "context_size", 4096)
        huge = {"model": "m", "messages": [{"role": "user", "content": pasted.wrap([pasted.Doc("p", LOG * 20)])}]}
        r = await c.post("/v1/chat/completions", headers={**HEADERS, "X-HIVI-Prefill": "1"}, json=huge)
        assert r.headers["X-HIVI-Prefill"] == "skipped"
    assert llama_ready["prefill"] == []
