"""BAK-1: local snapshots of what HIVI learns, and a one-step restore.

A snapshot is one folder, ``<backup_dir>/hivi-YYYYMMDD-HHMMSS[-reason]/``:

    memory.db        personal memory, feedback, settings (SQLite online backup)
    configs/         model_roles.yaml, routing.yaml, storage.yaml
    haie/            HAIE's learned data (data/haie), if present
    manifest.json    what is inside, HIVI version, the reason

Models are never backed up (re-downloadable); the memory export is also written
as ``memory.md`` beside the database so a snapshot is readable without HIVI.

When: daily (checked a minute after start and then hourly), before a restore,
and on demand ("before update"). Retention keeps the newest N (default 14),
never deleting anything outside the folders this module created.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import chat_history
import personal_memory as pm
from hivi_paths import paths

PREFIX = "hivi-"
DAILY_S = 24 * 3600
_lock = threading.Lock()


def default_dir() -> Path:
    return paths.data_root() / "backups"


def backup_dir(store: pm.MemoryStore | None = None) -> Path:
    store = store or pm.store()
    custom = str(store.setting("backup_dir") or "").strip()
    return Path(custom).expanduser() if custom else default_dir()


def _configs_dir() -> Path:
    return paths.repo_root() / "configs"


def _haie_dir() -> Path:
    return paths.data_root() / "haie"


def _snapshot_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out = [p for p in root.iterdir()
           if p.is_dir() and p.name.startswith(PREFIX) and (p / "manifest.json").is_file()]
    return sorted(out, key=lambda p: p.name, reverse=True)


def _read_manifest(p: Path) -> dict[str, Any]:
    try:
        return json.loads((p / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def list_snapshots(store: pm.MemoryStore | None = None) -> list[dict[str, Any]]:
    root = backup_dir(store)
    out = []
    for p in _snapshot_dirs(root):
        m = _read_manifest(p)
        out.append({
            "name": p.name,
            "path": str(p),
            "at": m.get("at"),
            "reason": m.get("reason", ""),
            "memories": m.get("memories"),
            "size_kb": round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024),
            "ok": bool(m.get("at")),
        })
    return out


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def create_snapshot(reason: str = "manual", store: pm.MemoryStore | None = None) -> dict[str, Any]:
    """Make one snapshot now. Raises OSError if the destination can't be written."""
    store = store or pm.store()
    root = backup_dir(store)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = "".join(c for c in reason if c.isalnum() or c in "-_")[:24]
    name = f"{PREFIX}{stamp}" + (f"-{safe}" if safe and safe != "manual" else "")
    target = root / name
    tmp = root / (name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    with _lock:
        tmp.mkdir(parents=True)
        try:
            store.backup_to(tmp / "memory.db")
            (tmp / "memory.md").write_text(store.export_markdown(), encoding="utf-8")
            chat_history.store().backup_to(tmp / "chats.db")
            cfg = _configs_dir()
            if cfg.is_dir():
                shutil.copytree(cfg, tmp / "configs", dirs_exist_ok=True)
            haie = _haie_dir()
            if haie.is_dir():
                shutil.copytree(haie, tmp / "haie", dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("*.tmp", "*.lock"))
            manifest = {
                "format": "hivi-snapshot",
                "version": 1,
                "at": time.time(),
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reason": reason,
                "memories": store.count(),
                "feedback": store.feedback_stats().get("total", 0),
                "chats": chat_history.store().count(),
                "items": ["memory.db", "memory.md", "chats.db", "configs", "haie"],
            }
            (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            tmp.rename(target)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    _retain(root, int(store.setting("backup_keep") or 14))
    _note_backup(store, target)
    return {"name": target.name, "path": str(target), "at": manifest["at"],
            "size_kb": round(_dir_size(target) / 1024)}


def _retain(root: Path, keep: int) -> None:
    for old in _snapshot_dirs(root)[max(1, keep):]:
        shutil.rmtree(old, ignore_errors=True)


_LAST: dict[str, Any] = {"at": None, "path": None, "error": None}


def _note_backup(store: pm.MemoryStore, target: Path) -> None:
    _LAST.update(at=time.time(), path=str(target), error=None)


def last_backup(store: pm.MemoryStore | None = None) -> dict[str, Any]:
    snaps = list_snapshots(store)
    newest = snaps[0] if snaps else None
    return {
        "last_at": newest["at"] if newest else None,
        "last_name": newest["name"] if newest else None,
        "count": len(snaps),
        "dir": str(backup_dir(store)),
        "error": _LAST.get("error"),
        "age_h": None if not newest or not newest["at"] else round((time.time() - newest["at"]) / 3600, 1),
    }


def restore_snapshot(name: str, store: pm.MemoryStore | None = None) -> dict[str, Any]:
    """Put a snapshot's memory and configs back. Takes a safety snapshot first."""
    store = store or pm.store()
    root = backup_dir(store)
    src = root / name
    if not name.startswith(PREFIX) or not (src / "manifest.json").is_file() or src.parent != root:
        raise FileNotFoundError(name)
    safety = create_snapshot("before-restore", store)
    with _lock:
        db = store.path
        store.close()
        try:
            if (src / "memory.db").is_file():
                for suffix in ("", "-wal", "-shm"):
                    p = Path(str(db) + suffix)
                    if p.exists():
                        p.unlink()
                shutil.copy2(src / "memory.db", db)
            if (src / "chats.db").is_file():
                chats = chat_history.store()
                chats.close()
                try:
                    for suffix in ("", "-wal", "-shm"):
                        p = Path(str(chats.path) + suffix)
                        if p.exists():
                            p.unlink()
                    shutil.copy2(src / "chats.db", chats.path)
                finally:
                    chat_history.set_store(None)
            if (src / "configs").is_dir():
                shutil.copytree(src / "configs", _configs_dir(), dirs_exist_ok=True)
            if (src / "haie").is_dir():
                shutil.copytree(src / "haie", _haie_dir(), dirs_exist_ok=True)
        finally:
            store.reopen()
    return {"restored": name, "safety_snapshot": safety["name"], "memories": store.count()}


# --------------------------------------------------------------------------- #
# Daily scheduler (a thread; never touches the GPU)
# --------------------------------------------------------------------------- #

def due(store: pm.MemoryStore | None = None) -> bool:
    store = store or pm.store()
    if not store.setting("backup_enabled"):
        return False
    info = last_backup(store)
    return info["last_at"] is None or time.time() - info["last_at"] >= DAILY_S


def run_if_due(store: pm.MemoryStore | None = None) -> dict[str, Any] | None:
    store = store or pm.store()
    if not due(store):
        return None
    try:
        return create_snapshot("daily", store)
    except OSError as e:
        _LAST["error"] = f"{type(e).__name__}: {e}"
        return None


_started = False


def start_scheduler(first_delay_s: float = 60.0, every_s: float = 3600.0) -> None:
    global _started
    if _started:
        return
    _started = True

    def loop() -> None:
        time.sleep(first_delay_s)
        while True:
            try:
                run_if_due()
            except Exception:  # noqa: BLE001, S110 - the scheduler must outlive one bad run
                pass
            time.sleep(every_s)

    threading.Thread(target=loop, name="hivi-backups", daemon=True).start()
