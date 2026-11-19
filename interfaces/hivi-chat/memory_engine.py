"""MEM-2 / MEM-3: automatic memory extraction and recall around each chat message.

The chat model is only used *after* a reply has finished and the GPU is idle
(extraction). Recall before a reply never touches the chat model: it uses the
embedding side service when it is already running, else a word-overlap match.
Both are quick, so a message is never held back by memory.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import pasted
import personal_memory as pm
from haie.harmony import extract_final_text

EXTRACT_SYSTEM = (
    "You maintain a short list of facts about one user for a local assistant. Read the "
    "exchange and the facts already saved. Return JSON only: "
    '{"memories":[{"text":"...","category":"about you|preferences|people|projects|other",'
    '"replaces":null}]}. '
    "Save only durable facts the USER stated about themselves: who they are, where they "
    "live or study or work, people in their life, their preferences and how they like "
    "things done, projects and goals. Write each as one short third-person line "
    "('Prefers X', 'Roommate is Ayush'). Skip: anything the assistant said, one-off tasks, "
    "passing mentions, mood, opinions about the assistant, and always skip passwords, "
    "card or ID numbers and API keys. If a saved fact changed, set \"replaces\" to that "
    "fact's id and write the new version. If nothing durable was said, return "
    '{"memories":[]}. No prose.'
)
_MAX_PER_PASS = 4
_TIMEOUT = httpx.Timeout(90.0, connect=5.0)
_EMBED_TIMEOUT = httpx.Timeout(20.0, connect=3.0)


@dataclass
class MemoryEvent:
    """What one pass changed, for the "Memory updated" chip."""
    batch: str
    added: list[dict[str, Any]] = field(default_factory=list)
    updated: list[dict[str, Any]] = field(default_factory=list)
    removed: list[dict[str, Any]] = field(default_factory=list)
    blocked: int = 0
    at: float = field(default_factory=time.time)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch": self.batch,
            "added": self.added,
            "updated": self.updated,
            "removed": self.removed,
            "blocked": self.blocked,
            "at": self.at,
        }


# --------------------------------------------------------------------------- #
# Parsing the model's answer
# --------------------------------------------------------------------------- #

def _json_obj(text: str) -> dict | None:
    if not text:
        return None
    t = extract_final_text(text) or text
    t = re.sub(r"<think>[\s\S]*?</think>", "", t).strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    for cand in (t, (re.search(r"\{[\s\S]*\}", t) or [None])[0]):
        if not cand:
            continue
        try:
            obj = json.loads(cand)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_candidates(text: str) -> list[dict[str, Any]]:
    obj = _json_obj(text)
    if not obj:
        return []
    rows = obj.get("memories")
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows[:_MAX_PER_PASS * 2]:
        if isinstance(row, str):
            row = {"text": row}
        if not isinstance(row, dict):
            continue
        txt = pm.clean_text(row.get("text", ""))
        if len(txt) < 4 or len(pm.tokens(txt)) < 1:
            continue
        rep = row.get("replaces")
        try:
            rep = int(rep) if rep not in (None, "", "null") else None
        except (TypeError, ValueError):
            rep = None
        out.append({"text": txt, "category": pm.clean_category(row.get("category")),
                    "replaces": rep})
        if len(out) >= _MAX_PER_PASS:
            break
    return out


def _exchange_text(user_text: str, reply_text: str) -> str:
    u = (user_text or "").strip()[:2500]
    a = (reply_text or "").strip()[:1200]
    return f"USER:\n{u}\n\nASSISTANT:\n{a}"


def build_extract_messages(user_text: str, reply_text: str, existing: list[dict[str, Any]],
                           user_name: str = "") -> list[dict[str, str]]:
    known = "\n".join(f"[{m['id']}] {m['text']}" for m in existing) or "(none)"
    who = f"The user's name is {user_name}. " if user_name else ""
    prompt = (f"{who}Facts already saved (id in brackets):\n{known}\n\n"
              f"Exchange:\n{_exchange_text(user_text, reply_text)}\n\nJSON:")
    return [{"role": "system", "content": EXTRACT_SYSTEM}, {"role": "user", "content": prompt}]


# --------------------------------------------------------------------------- #
# Applying candidates (dedupe / replace), shared by the pass and by tests
# --------------------------------------------------------------------------- #

def apply_candidates(store: pm.MemoryStore, candidates: list[dict[str, Any]], *,
                     chat_id: str | None, chat_title: str | None) -> MemoryEvent:
    event = MemoryEvent(batch=store.new_batch())
    for cand in candidates:
        text = cand["text"]
        if pm.is_sensitive(text):
            event.blocked += 1
            continue
        target = store.get(cand["replaces"]) if cand.get("replaces") else None
        if target is None:
            dup = store.find_duplicate(text)
            if dup is not None:
                continue
            # A changed version of an existing fact on the same subject replaces it.
            for rel in store.related(text, limit=3):
                if pm.lexical_similarity(text, rel["text"]) >= 0.55:
                    target = rel
                    break
        if target is not None:
            status, mem = store.update(target["id"], text, cand["category"], batch=event.batch)
            if status == "updated" and mem:
                event.updated.append({**mem, "previous": target["text"]})
            continue
        status, mem = store.add(text, category=cand["category"], origin="auto",
                                chat_id=chat_id, chat_title=chat_title, batch=event.batch)
        if status == "added" and mem:
            event.added.append(mem)
        elif status == "sensitive":
            event.blocked += 1
    return event


# --------------------------------------------------------------------------- #
# Embeddings for recall (only when the side service is up; never starts it)
# --------------------------------------------------------------------------- #

async def embed(base: str, texts: list[str], model: str = "local") -> list[list[float]] | None:
    if not texts:
        return []
    try:
        async with httpx.AsyncClient(timeout=_EMBED_TIMEOUT) as client:
            r = await client.post(f"{base.rstrip('/')}/v1/embeddings",
                                  json={"model": model, "input": texts})
        if r.status_code >= 400:
            return None
        data = r.json().get("data") or []
        vecs = [row.get("embedding") for row in sorted(data, key=lambda d: d.get("index", 0))]
        if len(vecs) != len(texts) or not all(isinstance(v, list) and v for v in vecs):
            return None
        return [[float(x) for x in v] for v in vecs]
    except (httpx.HTTPError, ValueError, OSError):
        return None


class Recall:
    """Picks the few memories that matter for one message, inside a token budget."""

    def __init__(self, store_fn=pm.store) -> None:
        self._store_fn = store_fn
        self.embed_base: str | None = None   # set by the server when the service is ready
        self.embed_model = "local"

    async def for_message(self, text: str) -> list[dict[str, Any]]:
        store = self._store_fn()
        items = store.all()
        if not items or len(pm.tokens(text)) == 0:
            return []
        scored: list[tuple[float, dict]] = []
        if self.embed_base:
            vectors = store.vectors()
            missing = [m for m in items if m["id"] not in vectors][:32]
            if missing:
                vecs = await embed(self.embed_base, [m["text"] for m in missing], self.embed_model)
                if vecs:
                    for m, v in zip(missing, vecs, strict=False):
                        store.set_vector(m["id"], v, self.embed_model)
                        vectors[m["id"]] = v
            q = await embed(self.embed_base, [text[:1000]], self.embed_model)
            if q:
                scored = pm.recall_vectors(q[0], items, vectors)
        if not scored:
            scored = pm.recall_lexical(text, items)
        return pm.fit_budget([m for _, m in scored])


# --------------------------------------------------------------------------- #
# The extraction pass
# --------------------------------------------------------------------------- #

class Extractor:
    """Runs one extraction per finished reply, cancelled by the next message."""

    def __init__(self, store_fn=pm.store) -> None:
        self._store_fn = store_fn
        self._task: asyncio.Task | None = None
        self.last_event: MemoryEvent | None = None
        self.events: dict[str, MemoryEvent] = {}   # by chat id, for the chip
        self.model_calls = 0

    def cancel(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    def busy(self) -> bool:
        return bool(self._task and not self._task.done())

    def schedule(self, run) -> asyncio.Task:
        self.cancel()
        self._task = asyncio.get_running_loop().create_task(run)
        return self._task

    async def run(self, *, backend_base: str, model: str, user_text: str, reply_text: str,
                  chat_id: str | None, chat_title: str | None, user_name: str = "",
                  wait_idle=None, skip_headers: dict[str, str] | None = None) -> MemoryEvent | None:
        store = self._store_fn()
        if not store.setting("memory_enabled") or not store.setting("auto_memory"):
            return None
        # The model only ever sees the first 2500 characters (_exchange_text); matching
        # a 200K-character paste against every memory would stall the event loop.
        # Attached documents are what the user pasted, not facts about the user.
        user_text = pasted.question_of(user_text or "")[:2500]
        if len(pm.tokens(user_text)) < 2:
            return None
        if wait_idle is not None:
            await wait_idle()
        existing = store.related(user_text, limit=8)
        messages = build_extract_messages(user_text, reply_text, existing, user_name)
        payload = {
            "model": model or "local",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 400,
            "stream": False,
            "reasoning_effort": "none",
            "reasoning_format": "none",
            "chat_template_kwargs": {"enable_thinking": False},
            "enable_thinking": False,
        }
        headers = {"X-HIVI-HAIE-Skip": "1", "X-HIVI-Internal": "memory"}
        headers.update(skip_headers or {})
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                self.model_calls += 1
                r = await client.post(f"{backend_base.rstrip('/')}/v1/chat/completions",
                                      json=payload, headers=headers)
            if r.status_code >= 400:
                return None
            data = r.json()
        except (httpx.HTTPError, ValueError, OSError):
            return None
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content")
        if isinstance(content, list):
            content = " ".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
        candidates = parse_candidates(content if isinstance(content, str) else "")
        if not candidates:
            return None
        event = apply_candidates(store, candidates, chat_id=chat_id, chat_title=chat_title)
        if event.changed:
            self.last_event = event
            if chat_id:
                self.events[chat_id] = event
            store.prune_history()
        return event


RECALL = Recall()
EXTRACTOR = Extractor()
