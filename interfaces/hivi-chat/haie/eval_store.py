"""Append-only HAIE evaluation log."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(r"C:\Users\mevad\Desktop\HIVI")
EVAL_DIR = ROOT / "data" / "haie"
EVAL_PATH = EVAL_DIR / "eval.jsonl"


def log_event(event: dict[str, Any]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    row = {"ts": time.time(), **event}
    with EVAL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_plan(
    *,
    model_id: str | None,
    engine: str | None,
    meta: dict[str, Any],
    user_preview: str,
) -> None:
    log_event(
        {
            "type": "plan",
            "model_id": model_id,
            "engine": engine,
            "user_preview": (user_preview or "")[:240],
            "meta": meta,
        }
    )


def log_outcome(
    *,
    model_id: str | None,
    engine: str | None,
    finish_reason: str | None,
    completion_tokens: int | None,
    empty: bool,
    truncated: bool,
    path: str | None = None,
) -> None:
    log_event(
        {
            "type": "outcome",
            "model_id": model_id,
            "engine": engine,
            "finish_reason": finish_reason,
            "completion_tokens": completion_tokens,
            "empty": empty,
            "truncated": truncated,
            "path": path,
        }
    )
