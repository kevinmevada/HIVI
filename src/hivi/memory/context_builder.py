"""Budgeted, governance-filtered project context for agents."""
from __future__ import annotations

from dataclasses import dataclass, field

from .curation import CorrectionStore, filter_current_memories
from .models import MemoryStatus, RecalledMemory, RecallResult


@dataclass
class ContextBudget:
    max_chars: int = 4000
    max_items: int = 12
    drop_disputed: bool = True


@dataclass
class BuiltContext:
    project_id: str
    query: str
    text: str
    items: list[RecalledMemory] = field(default_factory=list)
    truncated: bool = False
    dropped: int = 0


def estimate_chars(m: RecalledMemory) -> int:
    return len(m.text) + len(str(m.metadata or {})) + 32


def dedupe_memories(results: list[RecalledMemory]) -> list[RecalledMemory]:
    seen: set[str] = set()
    out: list[RecalledMemory] = []
    for m in results:
        key = " ".join(m.text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def rank_for_context(results: list[RecalledMemory]) -> list[RecalledMemory]:
    """Prefer verified/supported and higher scores."""
    status_w = {
        MemoryStatus.VERIFIED.value: 3,
        MemoryStatus.SUPPORTED.value: 2,
        MemoryStatus.UNVERIFIED.value: 1,
        MemoryStatus.DISPUTED.value: 0,
    }

    def key(m: RecalledMemory) -> tuple:
        st = str((m.metadata or {}).get("status") or "").lower()
        return (status_w.get(st, 0), m.score or 0.0)

    return sorted(results, key=key, reverse=True)


def build_context(
    recall: RecallResult,
    *,
    budget: ContextBudget | None = None,
    corrections: CorrectionStore | None = None,
) -> BuiltContext:
    budget = budget or ContextBudget()
    blocked = corrections.blocked_keys(recall.bank_id.replace("hivi_project_", "")) if corrections else {}
    # bank_id is hivi_project_001 — CorrectionStore keys by project_id "001"
    if corrections:
        # try both raw project forms
        from .scopes import normalize_project_id

        pid = recall.bank_id
        if pid.startswith("hivi_project_"):
            pid = pid.removeprefix("hivi_project_")
        blocked = corrections.blocked_keys(normalize_project_id(pid))

    drop = {
        MemoryStatus.SUPERSEDED.value,
        MemoryStatus.INVALIDATED.value,
    }
    if budget.drop_disputed:
        drop.add(MemoryStatus.DISPUTED.value)

    filtered = filter_current_memories(recall.results, blocked=blocked, drop_statuses=drop)
    filtered = dedupe_memories(filtered)
    ranked = rank_for_context(filtered)

    chosen: list[RecalledMemory] = []
    used = 0
    truncated = False
    for m in ranked:
        if len(chosen) >= budget.max_items:
            truncated = True
            break
        cost = estimate_chars(m)
        if used + cost > budget.max_chars and chosen:
            truncated = True
            break
        chosen.append(m)
        used += cost

    lines = []
    for i, m in enumerate(chosen, 1):
        st = (m.metadata or {}).get("status", "?")
        lines.append(f"[{i}] ({st}) {m.text}")

    return BuiltContext(
        project_id=recall.bank_id,
        query=recall.query,
        text="\n".join(lines),
        items=chosen,
        truncated=truncated,
        dropped=len(recall.results) - len(chosen),
    )
