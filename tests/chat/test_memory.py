"""Stage 2: personal memory, mood, loyalty prompt, backups (MEM-1..3, SOUL-1/2, BAK-1)."""
from __future__ import annotations

import asyncio
import json

import backups
import httpx
import memory_engine
import personal_memory as pm
import pytest
import server
import soul


@pytest.fixture(autouse=True)
def fresh_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVI_DATA_ROOT", str(tmp_path / "data"))
    st = pm.MemoryStore(tmp_path / "data" / "memory" / "memory.db")
    pm.set_store(st)
    monkeypatch.setattr(memory_engine, "EXTRACTOR", memory_engine.Extractor())
    monkeypatch.setattr(memory_engine, "RECALL", memory_engine.Recall())
    monkeypatch.setattr(backups, "_LAST", {"at": None, "path": None, "error": None})
    server._PROFILES.clear()
    yield st
    st.close()
    pm.set_store(None)


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://hivi")


# ---------------------------------------------------------------- store ---- #

def test_commands_english_and_hinglish():
    assert pm.parse_command("remember that I prefer PowerShell one-liners") == (
        "remember", "I prefer PowerShell one-liners")
    assert pm.parse_command("Hivi, forget that I prefer powershell one-liners")[0] == "forget"
    assert pm.parse_command("yaad rakhna ki mera roommate Ayush hai") == ("remember", "Mera roommate Ayush hai")
    assert pm.parse_command("mujhe chai pasand hai, ye yaad rakhna") == ("remember", "Mujhe chai pasand hai")
    assert pm.parse_command("bhool jao ki main Chicago mein rehta hoon")[0] == "forget"
    # conversation, not commands
    assert pm.parse_command("remember when we talked about the router?") is None
    assert pm.parse_command("do you remember my name?") is None
    assert pm.parse_command("forget it") is None
    assert pm.parse_command("I can't remember the command") is None


def test_secrets_never_stored(fresh_store):
    for text in ("my password is hunter2", "card 4111 1111 1111 1111 expires 09/27",
                 "api key sk-abcdefghijklmnopqrstuvwxyz123456", "ssn 123-45-6789",
                 "aadhaar 1234 5678 9012", "ghp_abcdefghijklmnopqrstuvwxyz1234"):
        assert pm.is_sensitive(text), text
        assert fresh_store.add(text)[0] == "sensitive"
    assert not pm.is_sensitive("Roommate is Ayush, born 1999")
    assert not pm.is_sensitive("Uses the Visa ending in 4417 for groceries")
    assert fresh_store.count() == 0


def test_dedupe_update_forget_undo(fresh_store):
    st = fresh_store
    assert st.add("I prefer PowerShell one-liners over scripts")[0] == "added"
    assert st.add("Prefers PowerShell one-liners over scripts")[0] == "duplicate"
    status, mem = st.add("Roommate is Ayush")
    assert status == "added" and mem["category"] == "people"
    assert st.update(mem["id"], "Roommate is Ayush, also in CSP 554")[0] == "updated"
    assert st.get(mem["id"])["text"] == "Roommate is Ayush, also in CSP 554"
    removed = st.forget("powershell one-liners")
    assert [m["text"] for m in removed] == ["I prefer PowerShell one-liners over scripts"]
    assert st.forget("something completely unrelated") == []
    batch = st.new_batch()
    st.add("Lives in Chicago", batch=batch)
    st.add("Studies at IIT", batch=batch)
    assert st.count() == 3
    assert st.undo(batch) == 2
    assert st.count() == 1
    assert st.delete_all() == 1


def test_export_import_roundtrip(fresh_store):
    st = fresh_store
    st.add("Lives in Chicago", category="about you")
    st.add("Roommate is Ayush", category="people")
    md = st.export_markdown()
    js = st.export_json()
    st.delete_all()
    assert st.import_items(pm.parse_import(md))["added"] == 2
    assert st.import_items(pm.parse_import(json.dumps(js)))["duplicate"] == 2
    cats = {m["text"]: m["category"] for m in st.all()}
    assert cats["Roommate is Ayush"] == "people"


def test_profile_and_recall_budget(fresh_store):
    st = fresh_store
    for i in range(80):
        st.add(f"Fact number {i} about a project called alpha{i}", category="projects")
    st.add("Name is Lakshya, based in Chicago", category="about you")
    chosen = pm.profile_memories(st.all())
    assert chosen[0]["category"] == "about you"
    assert sum(len(m["text"]) + 3 for m in chosen) <= pm.PROFILE_CHARS
    hits = pm.recall_lexical("tell me about project alpha7", st.all())
    assert hits and hits[0][1]["text"].endswith("alpha7")
    assert pm.recall_lexical("hello", st.all()) == []


def test_settings_are_typed(fresh_store):
    st = fresh_store
    out = st.update_settings({"memory_enabled": 0, "user_name": "  Lakshya ", "backup_keep": "900",
                              "unknown": 1})
    assert out["memory_enabled"] is False
    assert out["user_name"] == "Lakshya"
    assert out["backup_keep"] == 365
    assert "unknown" not in out


# ------------------------------------------------------------ extraction -- #

def test_parse_candidates_tolerates_model_noise():
    text = ('<think>hmm</think>```json\n{"memories":[{"text":"studies at IIT","category":"about"},'
            '{"text":"x"},{"text":"Roommate is Ayush","category":"people","replaces":"3"}]}\n```')
    out = memory_engine.parse_candidates(text)
    assert [c["text"] for c in out] == ["Studies at IIT", "Roommate is Ayush"]
    assert out[0]["category"] == "about you" and out[1]["replaces"] == 3
    assert memory_engine.parse_candidates("no json here") == []
    assert memory_engine.parse_candidates('{"memories":[]}') == []


def test_apply_candidates_replaces_and_blocks(fresh_store):
    st = fresh_store
    _, old = st.add("Lives in Chicago", category="about you")
    cands = memory_engine.parse_candidates(json.dumps({"memories": [
        {"text": "Lives in Chicago, new apartment on a different lease", "category": "about you",
         "replaces": old["id"]},
        {"text": "Lives in Chicago", "category": "about you"},
        {"text": "API key is sk-abcdefghijklmnopqrstuvwxyz123456", "category": "other"},
        {"text": "Roommate is Ayush", "category": "people"},
    ]}))
    ev = memory_engine.apply_candidates(st, cands, chat_id="c1", chat_title="t")
    assert [m["text"] for m in ev.added] == ["Roommate is Ayush"]
    assert ev.updated[0]["previous"] == "Lives in Chicago"
    assert ev.blocked == 1
    assert st.count() == 2
    st.undo(ev.batch)
    assert [m["text"] for m in st.all()] == ["Lives in Chicago"]


# -------------------------------------------------------------------- soul -- #

@pytest.mark.parametrize(("text", "label"), [
    ("fix this.. breaks formatting AGAIN!! wtf", "frustrated"),
    ("deadline kal hai aur kuch kaam nahi kar raha, bahut pareshan hoon", "stressed"),
    ("got the offer letter!!! finally", "excited"),
    ("haven't slept, its 3am", "tired"),
    ("feeling pretty down today, didnt get the job", "down"),
    ("i am not stressed, just curious how this works", "neutral"),
    ("Explain how transformers work", "neutral"),
])
def test_mood_reading(text, label):
    assert soul.read_mood(text).label == label


def test_system_prompt_layers():
    p = soul.build_system_prompt(profile="- Roommate is Ayush", mood=soul.read_mood("ugh broken again"),
                                 personality=True)
    assert soul.LOYALTY_RULES in p and soul.PERSONALITY in p
    assert "Roommate is Ayush" in p and "Right now:" in p
    plain = soul.build_system_prompt(personality=False, mood=soul.read_mood("hi"))
    assert soul.PERSONALITY not in plain and "Right now:" not in plain


# ------------------------------------------------------------------ server -- #

def _fake_backend(monkeypatch, replies: list[dict], seen: list[dict]):
    class Resp:
        def __init__(self, body):
            self.status_code = 200
            self.headers = {"content-type": "application/json"}
            self._body = body

        def json(self):
            return self._body

        @property
        def body(self):
            return json.dumps(self._body).encode()

    class Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def post(self, url, json=None, headers=None):
            seen.append({"url": url, "body": json, "headers": headers or {}})
            return Resp(replies.pop(0) if replies else {"choices": [{"message": {"content": ""}}]})

    monkeypatch.setattr(memory_engine.httpx, "AsyncClient", Client)


async def test_remember_and_forget_commands_bypass_the_model(fresh_store):
    headers = {"X-HIVI-Chat": "c1", "X-HIVI-Chat-Title": "test"}
    async with _client() as client:
        r = await client.post("/v1/chat/completions", headers=headers, json={
            "model": "local", "stream": False,
            "messages": [{"role": "user", "content": "remember that I prefer PowerShell one-liners"}]})
        assert r.status_code == 200
        assert "remember" in r.json()["choices"][0]["message"]["content"]
        assert r.headers["X-HIVI-HAIE-Path"] == "memory"
        assert fresh_store.count() == 1
        ev = (await client.get("/api/memory/events?chat=c1")).json()["event"]
        assert ev["added"][0]["text"] == "I prefer PowerShell one-liners"
        r = await client.post("/v1/chat/completions", headers=headers, json={
            "model": "local", "stream": True,
            "messages": [{"role": "user", "content": "forget that I prefer powershell one-liners"}]})
        assert r.headers["content-type"].startswith("text/event-stream")
        assert "Forgotten" in r.text
        assert fresh_store.count() == 0
        r = await client.post("/v1/chat/completions", headers=headers, json={
            "model": "local", "stream": False,
            "messages": [{"role": "user", "content": "remember that my password is hunter2"}]})
        assert "stays out of memory" in r.json()["choices"][0]["message"]["content"]


async def test_prepare_chat_builds_prompt_with_memory_mood_and_recall(fresh_store):
    st = fresh_store
    st.update_settings({"user_name": "Lakshya"})
    st.add("Roommate is Ayush", category="people")
    for i in range(200):  # far more than the profile block can hold
        st.add(f"Project note {i} about module beta{i}", category="projects")
    body = {"messages": [{"role": "system", "content": "You are HIVI."},
                         {"role": "user", "content": "what should I do about module beta42, deadline kal hai pareshan hoon"}]}
    ctx = {"hivi_chat": True, "chat_id": "c1", "title": None, "temporary": False, "internal": False}
    out, headers = await server._prepare_chat(body, ctx)
    system = out["messages"][0]["content"]
    turn = out["messages"][1]["content"]
    assert out["messages"][0]["role"] == "system" and len(out["messages"]) == 2
    assert system.startswith("You are HIVI.")
    assert "Lakshya" in system and "Roommate is Ayush" in system
    # per-turn signals live in the newest message, so the system prompt can stay cached
    assert "beta42" not in system and "Right now:" not in system
    assert turn.startswith("<hivi_context>") and "beta42" in turn and "Right now:" in turn
    assert turn.endswith("pareshan hoon") and headers["X-HIVI-Recalled"] == "1"
    assert headers["X-HIVI-Mood"].startswith("stressed")
    assert headers["X-HIVI-Memory"] == "on"
    # temporary chat: nothing personal, nothing recalled
    out, headers = await server._prepare_chat(body, {**ctx, "temporary": True})
    assert "Roommate" not in out["messages"][0]["content"] and headers["X-HIVI-Memory"] == "temporary"
    # memory off keeps the character but no facts
    st.update_settings({"memory_enabled": False})
    out, headers = await server._prepare_chat(body, ctx)
    assert "Roommate" not in out["messages"][0]["content"] and headers["X-HIVI-Memory"] == "off"


async def test_extraction_runs_after_reply_and_is_cancelled_by_next_message(fresh_store, monkeypatch):
    seen: list[dict] = []
    _fake_backend(monkeypatch, [
        {"choices": [{"message": {"content": json.dumps({"memories": [
            {"text": "Studies data science at IIT Chicago", "category": "about you"}]})}}]},
    ], seen)
    ext = memory_engine.EXTRACTOR
    ev = await ext.run(backend_base="http://x", model="m", user_text="I study data science at IIT Chicago",
                       reply_text="Nice.", chat_id="c1", chat_title="t")
    assert ev.added[0]["text"] == "Studies data science at IIT Chicago"
    assert seen[0]["headers"]["X-HIVI-HAIE-Skip"] == "1"
    assert seen[0]["body"]["enable_thinking"] is False
    assert fresh_store.all()[0]["origin"] == "auto"
    # cancelled: a new message outranks learning from the last one
    started = asyncio.Event()

    async def slow_idle():
        started.set()
        await asyncio.sleep(5)

    task = ext.schedule(ext.run(backend_base="http://x", model="m", user_text="another durable fact here",
                                reply_text="ok", chat_id="c1", chat_title="t", wait_idle=slow_idle))
    await started.wait()
    ext.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fresh_store.count() == 1
    # auto-memory off: no model call
    fresh_store.update_settings({"auto_memory": False})
    assert await ext.run(backend_base="http://x", model="m", user_text="I live in Chicago now",
                         reply_text="ok", chat_id="c1", chat_title="t") is None
    assert len(seen) == 1


async def test_memory_api_crud_export_import(fresh_store):
    async with _client() as client:
        r = await client.post("/api/memory", json={"text": "Lives in Chicago"})
        mid = r.json()["memory"]["id"]
        assert (await client.post("/api/memory", json={"text": "my password is hunter2"})).status_code == 400
        assert (await client.patch(f"/api/memory/{mid}", json={"text": "Lives in Chicago, Illinois"})).status_code == 200
        assert (await client.get("/api/memory?q=chicago")).json()["count"] == 1
        md = (await client.get("/api/memory/export?format=md")).text
        assert "- Lives in Chicago, Illinois" in md
        assert (await client.delete(f"/api/memory/{mid}")).status_code == 200
        r = await client.post("/api/memory/import", content=md, headers={"content-type": "text/plain"})
        assert r.json()["added"] == 1
        r = await client.post("/api/memory/delete-all", json={"confirm": "DELETE"})
        assert r.json()["count"] == 1
        r = await client.post("/api/memory/settings", json={"user_name": "Lakshya", "mood_enabled": False})
        assert r.json()["settings"]["user_name"] == "Lakshya"
        r = await client.post("/api/feedback", json={"rating": -1, "prompt": "p", "reply": "r", "better": "say it short"})
        assert r.json()["stats"] == {"total": 1, "up": 0, "down": 1, "rewrites": 1}
        assert "say it short" in (await client.get("/api/feedback/export")).text


# ----------------------------------------------------------------- backups -- #

async def test_backup_snapshot_restore_and_retention(fresh_store, tmp_path, monkeypatch):
    st = fresh_store
    st.update_settings({"backup_dir": str(tmp_path / "usb"), "backup_keep": 2})
    st.add("Lives in Chicago")
    async with _client() as client:
        r = await client.post("/api/backups", json={"reason": "test"})
        assert r.status_code == 200 and r.json()["count"] == 1
        first = r.json()["snapshot"]["name"]
        assert (tmp_path / "usb" / first / "memory.db").is_file()
        assert (tmp_path / "usb" / first / "memory.md").read_text().count("- Lives in Chicago") == 1
        st.add("Roommate is Ayush")
        await asyncio.sleep(1.1)  # snapshot names are per-second
        await client.post("/api/backups", json={"reason": "two"})
        await asyncio.sleep(1.1)
        await client.post("/api/backups", json={"reason": "three"})
        names = [s["name"] for s in (await client.get("/api/backups")).json()["snapshots"]]
        assert len(names) == 2 and first not in names  # retention keeps the newest 2
        assert (await client.post("/api/backups/restore", json={"name": names[-1]})).status_code == 400
        r = await client.post("/api/backups/restore", json={"name": names[-1], "confirm": "RESTORE"})
        assert r.status_code == 200 and r.json()["memories"] == 2
        assert (await client.post("/api/backups/restore",
                                  json={"name": "../etc", "confirm": "RESTORE"})).status_code == 404
    assert pm.store().count() == 2
    assert backups.due() is False


# ------------------------------------------------------------ chat history -- #

@pytest.fixture
def chat_store(tmp_path):
    import chat_history
    st = chat_history.ChatStore(tmp_path / "chats.db")
    chat_history.set_store(st)
    yield st
    st.close()
    chat_history.set_store(None)


async def test_chat_history_save_search_rename_delete(chat_store):
    async with _client() as client:
        msgs = [{"role": "user", "content": "how do I fix the header overlap in CSS?"},
                {"role": "assistant", "content": "Use container queries on the status area."}]
        r = await client.put("/api/chats/c_abc123", json={"messages": msgs, "model": "qwen"})
        assert r.status_code == 200 and r.json()["title"] == "how do I fix the header overlap in CSS?"
        msgs += [{"role": "user", "content": [{"type": "text", "text": "and with an image?"},
                                              {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
                 {"role": "assistant", "content": "Same idea."}]
        await client.put("/api/chats/c_abc123", json={"messages": msgs})
        await client.put("/api/chats/c_def456", json={"messages": [
            {"role": "user", "content": "plan my week for the exam"}, {"role": "assistant", "content": "Sure."}]})
        assert (await client.put("/api/chats/bad id!", json={"messages": msgs})).status_code == 400
        d = (await client.get("/api/chats")).json()
        assert d["count"] == 2 and d["chats"][0]["id"] == "c_def456"   # newest first
        hits = (await client.get("/api/chats?q=container")).json()["chats"]
        assert [h["id"] for h in hits] == ["c_abc123"] and "container" in hits[0]["snippet"]
        full = (await client.get("/api/chats/c_abc123")).json()
        assert full["message_count"] == 4 and full["messages"][2]["content"][1]["type"] == "image_url"
        await client.patch("/api/chats/c_def456", json={"title": "Exam week", "pinned": True})
        d = (await client.get("/api/chats")).json()
        assert d["chats"][0]["title"] == "Exam week" and d["chats"][0]["pinned"] is True
        assert (await client.get("/api/chats?q=exam")).json()["chats"][0]["title"] == "Exam week"
        assert (await client.delete("/api/chats/c_abc123")).status_code == 200
        assert (await client.delete("/api/chats/c_abc123")).status_code == 404
        r = await client.post("/api/chats/delete-all", json={"confirm": "DELETE"})
        assert r.json()["count"] == 1 and chat_store.count() == 0
