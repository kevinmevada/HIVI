"""Streaming SHA256 — never loads whole file into memory."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024, progress=None) -> dict:
    h = hashlib.sha256()
    total = 0
    try:
        size = path.stat().st_size
        last_pct = -1
        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
                total += len(chunk)
                if progress and size > 0:
                    pct = int(100 * total / size)
                    # report every 2%
                    if pct >= last_pct + 2 or total >= size:
                        last_pct = pct
                        progress(total, size, pct)
        return {
            "ok": True,
            "sha256": h.hexdigest().lower(),
            "bytes_read": total,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "sha256": None,
            "bytes_read": total,
            "error": f"{type(e).__name__}: {e}",
        }


def parse_sha256sums(path: Path) -> dict[str, str]:
    """Parse GNU-style SHA256SUMS (hash  filename or hash *filename)."""
    mapping: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return mapping
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if " *" in line:
            digest, name = line.split(" *", 1)
        elif "  " in line:
            digest, name = line.split("  ", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            digest, name = parts
        mapping[name.strip().replace("\\", "/")] = digest.strip().lower()
    return mapping
