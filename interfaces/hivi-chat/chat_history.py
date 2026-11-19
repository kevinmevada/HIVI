"""MEM-4: chat history, saved on this PC and searchable.

One SQLite file, ``<data_root>/memory/chats.db``. A chat is saved by the page
after every finished reply (``PUT /api/chats/{id}``), so a crash loses at most
the reply in flight. Temporary chats are never sent here.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import pasted
from hivi_paths import paths

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    model TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    messages TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS chats_updated ON chats(updated_at DESC);
CREATE VIRTUAL TABLE IF NOT EXISTS chats_fts USING fts5(id UNINDEXED, title, body);
"""
_MAX_MESSAGES_BYTES = 12 * 1024 * 1024  # images are data URLs; keep one chat under this
_ID = re.compile(r"^[A-Za-z0-9_-]{4,64}$")


def db_path() -> Path:
    return paths.data_root() / "memory" / "chats.db"


def _text_of(message: dict[str, Any]) -> str:
    c = message.get("content")
    if isinstance(c, list):
        return " ".join(str(p.get("text") or "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return str(c or "")


def _strip_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For search: text only."""
    return [{"role": m.get("role"), "text": _text_of(m)} for m in messages if isinstance(m, dict)]


def make_title(messages: list[dict[str, Any]]) -> str:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            docs, question = pasted.split(_text_of(m))
            t = re.sub(r"\s+", " ", question).strip() or (docs[0].name if docs else "")
            if t:
                return (t[:57] + "…") if len(t) > 60 else t
    return "New chat"


class ChatStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def backup_to(self, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            dest = sqlite3.connect(str(target))
            try:
                self._conn.backup(dest)
            finally:
                dest.close()

    @staticmethod
    def _summary(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "title": row["title"], "model": row["model"],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "message_count": row["message_count"], "pinned": bool(row["pinned"]),
        }

    def list(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, title, model, created_at, updated_at, message_count, pinned FROM chats "
                "ORDER BY pinned DESC, updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._summary(r) for r in rows]

    def search(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return self.list(limit)
        terms = [t for t in re.findall(r"[\wऀ-ॿ]+", q) if t]
        if not terms:
            return []
        match = " ".join(f'"{t}"*' for t in terms)
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT c.id, c.title, c.model, c.created_at, c.updated_at, c.message_count, c.pinned, "
                    "snippet(chats_fts, 2, '', '', '…', 12) AS snip "
                    "FROM chats_fts JOIN chats c ON c.id = chats_fts.id "
                    "WHERE chats_fts MATCH ? ORDER BY bm25(chats_fts) LIMIT ?", (match, limit)
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        return [{**self._summary(r), "snippet": r["snip"]} for r in rows]

    def get(self, chat_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if row is None:
            return None
        out = self._summary(row)
        out["messages"] = json.loads(row["messages"])
        return out

    def save(self, chat_id: str, messages: list[dict[str, Any]], *, model: str | None = None,
             title: str | None = None) -> dict[str, Any]:
        if not _ID.match(chat_id or ""):
            raise ValueError("bad chat id")
        messages = [m for m in messages if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
        raw = json.dumps(messages)
        while len(raw) > _MAX_MESSAGES_BYTES and messages:
            messages = messages[2:]  # drop the oldest exchange
            raw = json.dumps(messages)
        body = "\n".join(m["text"] for m in _strip_images(messages))
        now = time.time()
        with self._lock:
            old = self._conn.execute("SELECT title, created_at, pinned FROM chats WHERE id = ?",
                                     (chat_id,)).fetchone()
            created = old["created_at"] if old else now
            final_title = (title or "").strip()[:120] or (old["title"] if old else make_title(messages))
            self._conn.execute(
                "INSERT INTO chats(id, title, model, created_at, updated_at, messages, message_count, pinned) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET title = excluded.title, "
                "model = excluded.model, updated_at = excluded.updated_at, messages = excluded.messages, "
                "message_count = excluded.message_count",
                (chat_id, final_title, model, created, now, raw, len(messages), old["pinned"] if old else 0),
            )
            self._conn.execute("DELETE FROM chats_fts WHERE id = ?", (chat_id,))
            self._conn.execute("INSERT INTO chats_fts(id, title, body) VALUES(?, ?, ?)",
                               (chat_id, final_title, body))
            self._conn.commit()
        return {"id": chat_id, "title": final_title, "updated_at": now, "message_count": len(messages)}

    def rename(self, chat_id: str, title: str) -> bool:
        title = (title or "").strip()[:120]
        if not title:
            return False
        with self._lock:
            n = self._conn.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id)).rowcount
            if n:
                self._conn.execute("UPDATE chats_fts SET title = ? WHERE id = ?", (title, chat_id))
            self._conn.commit()
        return bool(n)

    def pin(self, chat_id: str, pinned: bool) -> bool:
        with self._lock:
            n = self._conn.execute("UPDATE chats SET pinned = ? WHERE id = ?",
                                   (1 if pinned else 0, chat_id)).rowcount
            self._conn.commit()
        return bool(n)

    def delete(self, chat_id: str) -> bool:
        with self._lock:
            n = self._conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,)).rowcount
            self._conn.execute("DELETE FROM chats_fts WHERE id = ?", (chat_id,))
            self._conn.commit()
        return bool(n)

    def delete_all(self) -> int:
        with self._lock:
            n = self._conn.execute("DELETE FROM chats").rowcount
            self._conn.execute("DELETE FROM chats_fts")
            self._conn.commit()
        return int(n or 0)

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM chats").fetchone()[0])


_STORE: ChatStore | None = None
_STORE_LOCK = threading.Lock()


def store() -> ChatStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = ChatStore()
        return _STORE


def set_store(new: ChatStore | None) -> None:
    global _STORE
    with _STORE_LOCK:
        _STORE = new
