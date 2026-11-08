"""HIVI SQLite app database (event ledger, retention queue — not semantic search)."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    actor_role TEXT,
    agent_id TEXT,
    task_id TEXT,
    content TEXT NOT NULL,
    evidence_path TEXT,
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS retention_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (event_id) REFERENCES events(id)
);

CREATE TABLE IF NOT EXISTS memory_corrections (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    target_memory_key TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT,
    actor_role TEXT NOT NULL,
    new_content TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_events_project ON events(project_id);
CREATE INDEX IF NOT EXISTS idx_queue_status ON retention_queue(status);
CREATE INDEX IF NOT EXISTS idx_corrections_project ON memory_corrections(project_id);
"""


def default_db_path() -> Path:
    root = Path(os.environ.get("HIVI_DATA_DIR", Path.cwd() / "data" / "hivi"))
    root.mkdir(parents=True, exist_ok=True)
    return root / "hivi.sqlite"


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn
