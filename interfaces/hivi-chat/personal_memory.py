"""Personal memory (MEM-1 to MEM-3): what HIVI knows about the user, kept on this PC.

One SQLite file, ``<data_root>/memory/memory.db``:

* ``memories``  one short fact per row, with category, dates, the chat it came from
                and (when the embedding side service has run) a vector for recall.
* ``history``   every add / update / delete, grouped in batches, so one click undoes
                what a "remember", a "forget" or an automatic pass changed.
* ``settings``  memory on/off, auto-learn, the user's name, personality, mood, backups.
* ``feedback``  thumbs up / down and "say it like this instead" (SOUL-2).

This is separate from Hindsight (project memory for agents): saving here is instant
and never runs the chat model in the reply path.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import struct
import threading
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from hivi_paths import paths

CATEGORIES = ("about you", "preferences", "people", "projects", "other")
_CATEGORY_ORDER = {c: i for i, c in enumerate(CATEGORIES)}
MAX_TEXT = 300

DEFAULT_SETTINGS: dict[str, Any] = {
    "memory_enabled": True,      # read and write memories at all
    "auto_memory": True,         # learn from chats after replies (MEM-2)
    "user_name": "",
    "personality": True,         # HIVI's character sheet in the system prompt
    "mood_enabled": True,        # SOUL-1 mood reading (never stored)
    "backup_enabled": True,      # BAK-1 daily snapshots
    "backup_dir": "",            # empty = <data_root>/backups
    "backup_keep": 14,
}


def db_path() -> Path:
    return paths.data_root() / "memory" / "memory.db"


# --------------------------------------------------------------------------- #
# Never stored: passwords, card / bank / ID numbers, API keys
# --------------------------------------------------------------------------- #

_SECRET_PATTERNS = [
    re.compile(r"\b(?:password|passwd|pwd|passcode|pin|otp|cvv|cvc)\b\s*(?:is|=|:|was)?\s*\S+", re.IGNORECASE),
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}"),            # OpenAI / Anthropic style keys
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{10,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}", re.IGNORECASE),  # GitHub tokens
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                        # AWS access keys
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"),                    # Google API keys
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),              # Slack tokens
    re.compile(r"\bhf_[A-Za-z0-9]{20,}", re.IGNORECASE),         # Hugging Face tokens
    re.compile(r"\b(?:api|secret|access|private)[ _-]?key\b\s*(?:is|=|:)\s*\S+", re.IGNORECASE),
    re.compile(r"\b(?:token|bearer)\s*(?:is|=|:)\s*[A-Za-z0-9._-]{12,}", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                       # US SSN
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b(?=.*\b(?:aadhaar|aadhar|uid)\b)", re.IGNORECASE),
    re.compile(r"\b(?:aadhaar|aadhar)\b\D{0,12}\d{4}\s?\d{4}\s?\d{4}\b", re.IGNORECASE),
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),                      # Indian PAN
    re.compile(r"\b(?:passport|license|licence|ssn|social security)\s*(?:no\.?|number|#)?\s*(?:is|=|:)?\s*[A-Z0-9-]{6,}", re.IGNORECASE),
    re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),            # IBAN
    re.compile(r"\b(?:account|acct|a/c|routing|ifsc|sort code)\s*(?:no\.?|number|#)?\s*(?:is|=|:)?\s*[A-Z0-9]{6,}", re.IGNORECASE),
]
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def is_sensitive(text: str) -> bool:
    """True when the text holds something HIVI never keeps (secrets, card / ID numbers)."""
    if not text:
        return False
    if any(p.search(text) for p in _SECRET_PATTERNS):
        return True
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return True
    return False


# --------------------------------------------------------------------------- #
# Text similarity (works with no embedding model at all)
# --------------------------------------------------------------------------- #

_STOP_WORDS = """
a an the and or but if then so of to in on at for from by with about as is are was were be been
being am do does did have has had i me my mine myself you your yours we our us it its this that
these those there here what which who whom when where why how not no yes very really just also
too can could should would will shall may might must let lets i'm im i've ive i'd id i'll ill
user user's hivi please ki ka ke ko hai hain tha thi mai main mera meri mere mujhe hum
humara aur bhi toh ye yeh woh wo se
"""
_STOP = frozenset(_STOP_WORDS.split())
_WORD = re.compile(r"[a-z0-9\u0900-\u097f][a-z0-9\u0900-\u097f+#.'_-]*", re.IGNORECASE)


def tokens(text: str) -> list[str]:
    out = []
    for w in _WORD.findall((text or "").lower()):
        w = w.strip("._'-")
        if not w or w in _STOP:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # plural → singular, enough for matching
        out.append(w)
    return out


def lexical_similarity(a: str, b: str) -> float:
    """Overlap of content words (0..1), symmetric."""
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / math.sqrt(len(ta) * len(tb))


def coverage(query: str, text: str) -> float:
    """Share of the query's content words found in ``text`` (for "forget X")."""
    tq, tt = set(tokens(query)), set(tokens(text))
    if not tq:
        return 0.0
    return len(tq & tt) / len(tq)


def normalize(text: str) -> str:
    return " ".join(tokens(text))


def pack_vector(vec: Iterable[float]) -> bytes:
    vals = list(vec)
    return struct.pack(f"<{len(vals)}f", *vals)


def unpack_vector(blob: bytes | None) -> list[float] | None:
    if not blob:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# --------------------------------------------------------------------------- #
# Categories (for "remember" commands; automatic memories get theirs from the model)
# --------------------------------------------------------------------------- #

_PEOPLE = re.compile(
    r"\b(?:mom|mum|mother|dad|father|mummy|papa|maa|brother|sister|bhai|didi|wife|husband|"
    r"partner|girlfriend|boyfriend|friend|roommate|son|daughter|boss|manager|professor|"
    r"teacher|colleague|coworker|uncle|aunt|cousin)\b",
    re.IGNORECASE,
)
_PREF = re.compile(
    r"\b(?:prefer|prefers|like|likes|love|loves|hate|hates|dislike|always|never|don't|dont|"
    r"rather|favou?rite|want|wants|pasand|nahi chahiye|chahiye)\b",
    re.IGNORECASE,
)
_PROJECT = re.compile(
    r"\b(?:project|building|working on|startup|app|repo|thesis|assignment|course|class|"
    r"exam|deadline|job search|internship)\b",
    re.IGNORECASE,
)
_ABOUT = re.compile(
    r"\b(?:my name|call me|i am|i'm|im|i live|i work|i study|student|years old|born|from|"
    r"based in|mera naam|main\s+\w+\s+hoon)\b",
    re.IGNORECASE,
)


def guess_category(text: str) -> str:
    if _PEOPLE.search(text):
        return "people"
    if _ABOUT.search(text):
        return "about you"
    if _PREF.search(text):
        return "preferences"
    if _PROJECT.search(text):
        return "projects"
    return "other"


def clean_category(value: Any) -> str:
    v = str(value or "").strip().lower()
    aliases = {"about": "about you", "identity": "about you", "profile": "about you",
               "preference": "preferences", "person": "people", "project": "projects",
               "work": "projects", "tools": "preferences", "tool": "preferences"}
    v = aliases.get(v, v)
    return v if v in CATEGORIES else "other"


def clean_text(text: str) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip().strip("\"'“”‘’").strip()
    t = t.rstrip(" .")
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t[:MAX_TEXT]


# --------------------------------------------------------------------------- #
# "Remember …" / "Forget …" (English and Hinglish)
# --------------------------------------------------------------------------- #

_REMEMBER = [
    re.compile(r"^(?:hey\s+hivi[,!]?\s+|hivi[,!]?\s+)?(?:please\s+|pls\s+|can you\s+|could you\s+)?"
               r"(?:remember|note down|make a note|save to (?:your )?memory|add to (?:your )?memory|"
               r"keep in mind|don'?t forget)\s*(?:that|this|:|-|,)?\s+(?P<fact>.+)$", re.IGNORECASE | re.DOTALL),
    re.compile(r"^(?:hivi[,!]?\s+)?(?:yaad\s+rakh(?:o|na|ega|egi|iyo)?|yad\s+rakh(?:o|na)?)"
               r"\s*(?:ki|ke|:|-|,)?\s+(?P<fact>.+)$", re.IGNORECASE | re.DOTALL),
    re.compile(r"^(?P<fact>.+?)[,.]?\s+(?:ye(?:h)?\s+|isko\s+|ise\s+)?(?:yaad\s+rakh(?:o|na|ega|egi|iyo)?|"
               r"yad\s+rakh(?:o|na)?)[.!]*$", re.IGNORECASE | re.DOTALL),
]
_FORGET = [
    re.compile(r"^(?:hey\s+hivi[,!]?\s+|hivi[,!]?\s+)?(?:please\s+|pls\s+|can you\s+|could you\s+)?"
               r"(?:forget|remove from (?:your )?memory|delete (?:the )?memory (?:about|of|that)|"
               r"stop remembering|erase (?:the )?memory (?:about|of|that))\s*(?:that|about|:|-|,)?\s+"
               r"(?P<fact>.+)$", re.IGNORECASE | re.DOTALL),
    re.compile(r"^(?:hivi[,!]?\s+)?bhool\s+ja(?:o|na|iye)?\s*(?:ki|ke|:|-|,)?\s+(?P<fact>.+)$", re.IGNORECASE | re.DOTALL),
    re.compile(r"^(?P<fact>.+?)[,.]?\s+(?:wali baat\s+|waali baat\s+|ye(?:h)?\s+)?bhool\s+ja(?:o|na|iye)?[.!]*$",
               re.IGNORECASE | re.DOTALL),
]
# "remember when we…", "do you remember…", "forget it" are conversation, not commands.
_NOT_A_COMMAND = re.compile(
    r"^(?:when|how|what|where|why|who|the time|me\b|us\b|it\b|it\.|about it|that\.?$|this\.?$)",
    re.IGNORECASE,
)


def parse_command(text: str) -> tuple[str, str] | None:
    """("remember", fact) / ("forget", what) for an explicit memory command, else None."""
    t = (text or "").strip()
    if not t or len(t) > 600 or "\n\n" in t:
        return None
    first = t.splitlines()[0].strip()
    if first.endswith("?"):
        return None
    for kind, patterns in (("forget", _FORGET), ("remember", _REMEMBER)):
        for pat in patterns:
            m = pat.match(first)
            if not m:
                continue
            fact = clean_text(m.group("fact"))
            if len(fact) < 3 or _NOT_A_COMMAND.match(fact):
                return None
            return kind, fact
    return None


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'other',
    origin TEXT NOT NULL DEFAULT 'manual',
    source_chat TEXT,
    source_title TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    embedding BLOB,
    embed_model TEXT
);
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch TEXT NOT NULL,
    action TEXT NOT NULL,
    memory_id INTEGER NOT NULL,
    old_text TEXT,
    new_text TEXT,
    old_category TEXT,
    new_category TEXT,
    snapshot TEXT,
    at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS history_batch ON history(batch);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT,
    rating INTEGER NOT NULL,
    prompt TEXT,
    reply TEXT,
    better TEXT,
    model TEXT,
    at REAL NOT NULL
);
"""
_COLUMNS = ("id", "text", "category", "origin", "source_chat", "source_title",
            "created_at", "updated_at")
# A new fact this close to an existing one is the same fact.
DUPLICATE_AT = 0.86


class MemoryStore:
    """Thread-safe; every public method opens its own short transaction."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = self._open()

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        conn.commit()
        return conn

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    def reopen(self) -> None:
        """After a restore replaced the file on disk."""
        with self._lock:
            self.close()
            self._conn = self._open()

    def backup_to(self, target: Path) -> None:
        """Consistent copy even mid-write (SQLite online backup)."""
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            dest = sqlite3.connect(str(target))
            try:
                self._conn.backup(dest)
            finally:
                dest.close()

    # -- settings ---------------------------------------------------------- #

    def settings(self) -> dict[str, Any]:
        out = dict(DEFAULT_SETTINGS)
        with self._lock:
            for row in self._conn.execute("SELECT key, value FROM settings"):
                try:
                    out[row["key"]] = json.loads(row["value"])
                except ValueError:
                    continue
        return out

    def setting(self, key: str) -> Any:
        return self.settings().get(key, DEFAULT_SETTINGS.get(key))

    def update_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        clean: dict[str, Any] = {}
        for key, value in values.items():
            if key not in DEFAULT_SETTINGS:
                continue
            default = DEFAULT_SETTINGS[key]
            if isinstance(default, bool):
                value = bool(value)
            elif isinstance(default, int):
                try:
                    value = max(1, min(365, int(value)))
                except (TypeError, ValueError):
                    continue
            else:
                value = str(value or "").strip()[:200]
            clean[key] = value
        with self._lock:
            for key, value in clean.items():
                self._conn.execute(
                    "INSERT INTO settings(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, json.dumps(value)),
                )
            self._conn.commit()
        return self.settings()

    # -- reading ----------------------------------------------------------- #

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        return {k: row[k] for k in _COLUMNS}

    def get(self, memory_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._row(row) if row else None

    def all(self, query: str = "") -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM memories ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        items = [self._row(r) for r in rows]
        q = (query or "").strip().lower()
        if q:
            items = [m for m in items
                     if q in m["text"].lower() or q in m["category"]
                     or coverage(q, m["text"]) >= 0.5]
        return items

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def vectors(self) -> dict[int, list[float]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"
            ).fetchall()
        return {r["id"]: v for r in rows if (v := unpack_vector(r["embedding"]))}

    def missing_vectors(self, limit: int = 64) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM memories WHERE embedding IS NULL "
                "ORDER BY id LIMIT ?", (limit,)
            ).fetchall()
        return [self._row(r) for r in rows]

    def set_vector(self, memory_id: int, vec: list[float], model: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET embedding = ?, embed_model = ? WHERE id = ?",
                (pack_vector(vec), model, memory_id),
            )
            self._conn.commit()

    def find_duplicate(self, text: str, *, exclude: int | None = None) -> dict[str, Any] | None:
        norm = normalize(text)
        best, best_sim = None, 0.0
        for m in self.all():
            if m["id"] == exclude:
                continue
            if normalize(m["text"]) == norm:
                return m
            sim = lexical_similarity(text, m["text"])
            if sim > best_sim:
                best, best_sim = m, sim
        return best if best_sim >= DUPLICATE_AT else None

    def related(self, text: str, limit: int = 6) -> list[dict[str, Any]]:
        """Existing memories on the same subject (for the extraction pass)."""
        scored = [(lexical_similarity(text, m["text"]), m) for m in self.all()]
        scored = [s for s in scored if s[0] > 0.12]
        scored.sort(key=lambda s: s[0], reverse=True)
        return [m for _, m in scored[:limit]]

    # -- writing (every change lands in history for undo) ------------------- #

    @staticmethod
    def new_batch() -> str:
        return uuid.uuid4().hex[:12]

    def _log(self, batch: str, action: str, memory_id: int, *, old: dict | None = None,
             new_text: str | None = None, new_category: str | None = None) -> None:
        self._conn.execute(
            "INSERT INTO history(batch, action, memory_id, old_text, new_text, old_category, "
            "new_category, snapshot, at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (batch, action, memory_id,
             None if old is None else old["text"], new_text,
             None if old is None else old["category"], new_category,
             None if old is None else json.dumps(old), time.time()),
        )

    def add(self, text: str, *, category: str | None = None, origin: str = "manual",
            chat_id: str | None = None, chat_title: str | None = None,
            batch: str | None = None) -> tuple[str, dict[str, Any] | None]:
        """("added" | "duplicate" | "sensitive" | "empty", memory)."""
        if is_sensitive(text):
            return "sensitive", None
        text = clean_text(text)
        if len(text) < 3:
            return "empty", None
        dup = self.find_duplicate(text)
        if dup is not None:
            return "duplicate", dup
        cat = clean_category(category) if category else guess_category(text)
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO memories(text, category, origin, source_chat, source_title, "
                "created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (text, cat, origin, chat_id, (chat_title or "")[:120] or None, now, now),
            )
            mid = int(cur.lastrowid)
            self._log(batch or self.new_batch(), "add", mid, new_text=text, new_category=cat)
            self._conn.commit()
        return "added", self.get(mid)

    def update(self, memory_id: int, text: str | None = None, category: str | None = None,
               *, batch: str | None = None) -> tuple[str, dict[str, Any] | None]:
        old = self.get(memory_id)
        if old is None:
            return "missing", None
        if text is not None and is_sensitive(text):
            return "sensitive", old
        new_text = clean_text(text) if text is not None else old["text"]
        new_cat = clean_category(category) if category else old["category"]
        if len(new_text) < 3:
            return "empty", old
        if new_text == old["text"] and new_cat == old["category"]:
            return "unchanged", old
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET text = ?, category = ?, updated_at = ?, "
                "embedding = CASE WHEN ? THEN NULL ELSE embedding END WHERE id = ?",
                (new_text, new_cat, time.time(), new_text != old["text"], memory_id),
            )
            self._log(batch or self.new_batch(), "update", memory_id, old=old,
                      new_text=new_text, new_category=new_cat)
            self._conn.commit()
        return "updated", self.get(memory_id)

    def delete(self, memory_id: int, *, batch: str | None = None) -> dict[str, Any] | None:
        old = self.get(memory_id)
        if old is None:
            return None
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._log(batch or self.new_batch(), "delete", memory_id, old=old)
            self._conn.commit()
        return old

    def delete_all(self) -> int:
        """Gone for good: no undo trail is kept for a full wipe."""
        with self._lock:
            n = self._conn.execute("DELETE FROM memories").rowcount
            self._conn.execute("DELETE FROM history")
            self._conn.commit()
        return int(n or 0)

    def forget(self, query: str, *, batch: str | None = None,
               limit: int = 5) -> list[dict[str, Any]]:
        """Delete the memories a "forget …" names. Conservative: needs a clear match."""
        scored = []
        for m in self.all():
            cov = coverage(query, m["text"])
            sim = lexical_similarity(query, m["text"])
            score = max(cov, sim)
            if score >= 0.6:
                scored.append((score, m))
        scored.sort(key=lambda s: s[0], reverse=True)
        if not scored:
            return []
        top = scored[0][0]
        # Only the clear winners: a near-tie set, not everything that shares a word.
        chosen = [m for s, m in scored if s >= max(0.6, top - 0.15)][:limit]
        batch = batch or self.new_batch()
        return [d for m in chosen if (d := self.delete(m["id"], batch=batch))]

    def undo(self, batch: str) -> int:
        """Reverse one batch of changes (newest first). Returns how many were undone."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM history WHERE batch = ? ORDER BY id DESC", (batch,)
            ).fetchall()
            n = 0
            for r in rows:
                if r["action"] == "add":
                    n += self._conn.execute(
                        "DELETE FROM memories WHERE id = ?", (r["memory_id"],)
                    ).rowcount
                elif r["action"] == "update":
                    n += self._conn.execute(
                        "UPDATE memories SET text = ?, category = ?, updated_at = ?, "
                        "embedding = NULL WHERE id = ?",
                        (r["old_text"], r["old_category"], time.time(), r["memory_id"]),
                    ).rowcount
                elif r["action"] == "delete" and r["snapshot"]:
                    old = json.loads(r["snapshot"])
                    self._conn.execute(
                        "INSERT OR REPLACE INTO memories(id, text, category, origin, source_chat, "
                        "source_title, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                        tuple(old[k] for k in _COLUMNS),
                    )
                    n += 1
            self._conn.execute("DELETE FROM history WHERE batch = ?", (batch,))
            self._conn.commit()
        return n

    def prune_history(self, keep_days: float = 30) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM history WHERE at < ?",
                               (time.time() - keep_days * 86400,))
            self._conn.commit()

    # -- export / import ---------------------------------------------------- #

    def export_json(self) -> dict[str, Any]:
        return {
            "format": "hivi-memory",
            "version": 1,
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "memories": [
                {"text": m["text"], "category": m["category"], "origin": m["origin"],
                 "created_at": m["created_at"], "updated_at": m["updated_at"],
                 "source_title": m["source_title"]}
                for m in self.all()
            ],
        }

    def export_markdown(self) -> str:
        lines = ["# HIVI memory", "",
                 f"Exported {time.strftime('%Y-%m-%d %H:%M')}. One memory per line.", ""]
        items = self.all()
        for cat in CATEGORIES:
            group = [m for m in items if m["category"] == cat]
            if not group:
                continue
            lines += [f"## {cat.capitalize()}", ""]
            lines += [f"- {m['text']}" for m in group]
            lines.append("")
        return "\n".join(lines)

    def import_items(self, items: list[dict[str, Any]], *, batch: str | None = None) -> dict[str, int]:
        batch = batch or self.new_batch()
        counts = {"added": 0, "duplicate": 0, "sensitive": 0, "empty": 0}
        for item in items:
            status, _ = self.add(str(item.get("text", "")), category=item.get("category"),
                                 origin="import", chat_title=item.get("source_title"),
                                 batch=batch)
            counts[status] = counts.get(status, 0) + 1
        return counts

    # -- feedback (SOUL-2) --------------------------------------------------- #

    def add_feedback(self, *, rating: int, prompt: str, reply: str, better: str = "",
                     chat_id: str | None = None, model: str | None = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO feedback(chat_id, rating, prompt, reply, better, model, at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?)",
                (chat_id, 1 if rating > 0 else -1, prompt[:8000], reply[:16000],
                 (better or "")[:8000] or None, model, time.time()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def feedback_stats(self) -> dict[str, int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), SUM(rating > 0), SUM(rating < 0), SUM(better IS NOT NULL) "
                "FROM feedback"
            ).fetchone()
        return {"total": row[0] or 0, "up": row[1] or 0, "down": row[2] or 0,
                "rewrites": row[3] or 0}

    def feedback_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM feedback ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def delete_feedback(self) -> int:
        with self._lock:
            n = self._conn.execute("DELETE FROM feedback").rowcount
            self._conn.commit()
        return int(n or 0)


def parse_import(raw: str) -> list[dict[str, Any]]:
    """JSON export from HIVI, a JSON list, or Markdown / plain bullet lines."""
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw[0] in "[{":
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        if isinstance(data, dict):
            data = data.get("memories")
        if isinstance(data, list):
            out = []
            for item in data:
                if isinstance(item, str):
                    out.append({"text": item})
                elif isinstance(item, dict) and item.get("text"):
                    out.append(item)
            return out
    out, category = [], None
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("#"):
            heading = s.lstrip("#").strip().lower()
            category = heading if heading in CATEGORIES else None
            continue
        m = re.match(r"^(?:[-*•]|\d+[.)])\s+(.+)$", s)
        if m:
            out.append({"text": m.group(1), "category": category})
    return out


# --------------------------------------------------------------------------- #
# Prompt blocks
# --------------------------------------------------------------------------- #

PROFILE_CHARS = 1800      # ≈ 450 tokens, the "who you are" block in the system prompt
RECALL_CHARS = 900        # ≈ 225 tokens per message for recalled memories
RECALL_MAX = 6


def profile_memories(items: list[dict[str, Any]], budget: int = PROFILE_CHARS) -> list[dict[str, Any]]:
    """The most important memories that fit the budget: who they are, then preferences…"""
    ordered = sorted(items, key=lambda m: (_CATEGORY_ORDER.get(m["category"], 9),
                                           -m["updated_at"]))
    chosen, used = [], 0
    for m in ordered:
        cost = len(m["text"]) + 3
        if used + cost > budget:
            continue
        chosen.append(m)
        used += cost
    return chosen


def profile_block(items: list[dict[str, Any]], user_name: str = "") -> str:
    lines = []
    if user_name:
        lines.append(f"The user's name is {user_name}. Use it now and then, not in every reply.")
    if items:
        lines.append(
            "What you know about the user from earlier chats (saved on this PC). Use a fact "
            "only when it helps with the current message; never list these back or mention "
            "that you have memory unless the user asks:"
        )
        lines += [f"- {m['text']}" for m in items]
    return "\n".join(lines)


def recall_lexical(query: str, items: list[dict[str, Any]], *, limit: int = RECALL_MAX,
                   threshold: float = 0.2) -> list[tuple[float, dict[str, Any]]]:
    scored = []
    for m in items:
        s = max(lexical_similarity(query, m["text"]), coverage(m["text"], query) * 0.8)
        if s >= threshold:
            scored.append((s, m))
    scored.sort(key=lambda s: s[0], reverse=True)
    return scored[:limit]


def recall_vectors(qvec: list[float], items: list[dict[str, Any]], vectors: dict[int, list[float]],
                   *, limit: int = RECALL_MAX, threshold: float = 0.45) -> list[tuple[float, dict]]:
    scored = []
    for m in items:
        v = vectors.get(m["id"])
        if v is None:
            continue
        s = cosine(qvec, v)
        if s >= threshold:
            scored.append((s, m))
    scored.sort(key=lambda s: s[0], reverse=True)
    return scored[:limit]


def fit_budget(items: list[dict[str, Any]], budget: int = RECALL_CHARS) -> list[dict[str, Any]]:
    out, used = [], 0
    for m in items:
        cost = len(m["text"]) + 3
        if used + cost > budget:
            break
        out.append(m)
        used += cost
    return out


_STORE: MemoryStore | None = None
_STORE_LOCK = threading.Lock()


def store() -> MemoryStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = MemoryStore()
        return _STORE


def set_store(new: MemoryStore | None) -> None:
    """Tests and restore swap the store."""
    global _STORE
    with _STORE_LOCK:
        _STORE = new
