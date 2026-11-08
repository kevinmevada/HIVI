"""Teach → recall acceptance for the HIVI context engine (small local model).

Each case: retain (wait) then recall on a fresh query — not same-turn chat memory.
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hivi.memory.manager import MemoryManager
from hivi.memory.models import MemoryStatus, MemoryType, RetainItem
from hivi.memory.policies import ActorRole
from hivi.orchestrator.preflight import run_preflight

PROJECT = "096"  # dedicated bank for teach→recall suite
DB = ROOT / "data" / "hivi" / "context_teach_recall.sqlite"
OUT = ROOT / "data" / "hivi" / "context_teach_recall.json"


def _blob(rec) -> str:
    return " ".join((r.text or "") for r in rec.results).lower()


def _teach(mm: MemoryManager, content: str, memory_type: MemoryType = MemoryType.DECISION) -> dict:
    t0 = time.perf_counter()
    ret = mm.retain(
        project_id=PROJECT,
        items=[
            RetainItem(
                content=content,
                memory_type=memory_type,
                status=MemoryStatus.VERIFIED,
                source="teach_recall_smoke",
            )
        ],
        role=ActorRole.PROJECT_OWNER,
        wait=True,
    )
    return {
        "ok": bool(ret.success),
        "retain_s": round(time.perf_counter() - t0, 2),
        "bank_id": ret.bank_id,
        "items_count": ret.items_count,
    }


def _recall(mm: MemoryManager, query: str) -> dict:
    rec = mm.recall(project_id=PROJECT, query=query, max_tokens=512)
    return {
        "hits": len(rec.results),
        "sample": [(r.text or "")[:160] for r in rec.results[:5]],
        "blob": _blob(rec),
        "ids": [r.id for r in rec.results if r.id],
        "raw_results": rec.results,
    }


def _pass(name: str, ok: bool, **extra) -> dict:
    row = {"name": name, "ok": ok, **extra}
    print(json.dumps(row, indent=2))
    return row


def main() -> int:
    pf = run_preflight(project_id=PROJECT)
    print("preflight_ok", pf.ok, "model", pf.model_id, "project", PROJECT)
    if not pf.ok:
        print("errors", pf.errors)
        return 1

    mm = MemoryManager(db_path=DB)
    mm.ensure_project_bank(PROJECT)
    results: list[dict] = []
    marker = uuid.uuid4().hex[:8]

    try:
        # --- Preference memory ---
        teach = _teach(mm, f"User preference [{marker}]: I prefer Python for programming.")
        results.append(_pass("1_teach_python", teach["ok"], retain_s=teach["retain_s"], bank_id=teach["bank_id"]))

        r = _recall(mm, "Which programming language do I prefer?")
        hit = "python" in r["blob"]
        results.append(_pass("2_recall_python", hit, hits=r["hits"], sample=r["sample"]))

        # Update via supersede when we have a memory id; else retain new fact + note overlap risk
        mid = r["ids"][0] if r["ids"] else None
        if mid:
            t0 = time.perf_counter()
            mm.correct_memory(
                project_id=PROJECT,
                target_memory_key=mid,
                action="supersede",
                role=ActorRole.PROJECT_OWNER,
                reason="user changed preference",
                new_content=f"User preference [{marker}]: I now prefer C++ for programming.",
                sync_retain_replacement=True,
            )
            results.append(
                _pass(
                    "3_update_cpp",
                    True,
                    retain_s=round(time.perf_counter() - t0, 2),
                    superseded_id=mid,
                )
            )
        else:
            teach_u = _teach(
                mm,
                f"User preference [{marker}]: I now prefer C++ for programming. Previous Python preference is outdated.",
            )
            results.append(
                _pass("3_update_cpp", teach_u["ok"], retain_s=teach_u["retain_s"], note="no id; retained replacement only")
            )

        r2 = _recall(mm, "Which language do I currently prefer for programming?")
        has_cpp = "c++" in r2["blob"] or "cpp" in r2["blob"]
        # Prefer current: C++ present. Soft-fail if Python still co-listed (engine returned both).
        results.append(
            _pass(
                "4_recall_cpp",
                has_cpp,
                hits=r2["hits"],
                sample=r2["sample"],
                python_still_present="python" in r2["blob"],
            )
        )

        r3 = _recall(mm, "What's my favorite color?")
        # Must not invent a color from empty memory — no color preference taught.
        color_words = ("blue", "red", "green", "yellow", "purple", "orange", "pink", "black", "white")
        invented = any(c in r3["blob"] for c in color_words) and (
            "favorite color" in r3["blob"] or "colour" in r3["blob"] or "prefer" in r3["blob"]
        )
        # Empty or unrelated hits (no taught color) = pass
        unknown_ok = (r3["hits"] == 0) or (not invented and "favorite color" not in r3["blob"])
        results.append(
            _pass(
                "5_unknown_color",
                unknown_ok,
                hits=r3["hits"],
                sample=r3["sample"],
                note="pass if memory has no favorite-color fact",
            )
        )

        # --- GPU ---
        teach_g = _teach(mm, f"Hardware [{marker}]: My laptop has an RTX 5060 GPU.")
        results.append(_pass("gpu_teach", teach_g["ok"], retain_s=teach_g["retain_s"]))
        rg = _recall(mm, "Which GPU does my laptop have?")
        results.append(
            _pass(
                "gpu_recall",
                "5060" in rg["blob"] or "rtx" in rg["blob"],
                hits=rg["hits"],
                sample=rg["sample"],
            )
        )

        # --- Project context ---
        teach_p = _teach(mm, f"Project Alpha [{marker}] uses Python as its programming language.")
        results.append(_pass("alpha_teach", teach_p["ok"], retain_s=teach_p["retain_s"]))
        ra = _recall(mm, "What language does Project Alpha use?")
        results.append(
            _pass(
                "alpha_recall",
                "python" in ra["blob"] and "alpha" in ra["blob"],
                hits=ra["hits"],
                sample=ra["sample"],
            )
        )

        # --- Local-model constraint ---
        teach_c = _teach(
            mm,
            f"Constraint [{marker}]: I only want local AI models, never cloud APIs.",
            memory_type=MemoryType.CONSTRAINT,
        )
        results.append(_pass("local_teach", teach_c["ok"], retain_s=teach_c["retain_s"]))
        rc = _recall(mm, "Can I use a cloud API for my project?")
        constraint_hit = any(
            x in rc["blob"] for x in ("local", "never cloud", "cloud api", "no cloud", "not cloud")
        )
        results.append(
            _pass(
                "local_recall_constraint",
                constraint_hit,
                hits=rc["hits"],
                sample=rc["sample"],
                note="expect constraint to surface; not a freeform chat refusal",
            )
        )

        # --- Restart persistence ---
        persist_fact = f"Persist marker [{marker}]: HIVI restart-persistence fact TOKEN_{marker}."
        teach_r = _teach(mm, persist_fact)
        results.append(_pass("restart_teach", teach_r["ok"], retain_s=teach_r["retain_s"]))
        mm.close()

        mm2 = MemoryManager(db_path=DB)
        try:
            rr = _recall(mm2, f"What is the restart-persistence fact TOKEN_{marker}?")
            results.append(
                _pass(
                    "restart_recall",
                    marker.lower() in rr["blob"] or f"token_{marker}".lower() in rr["blob"],
                    hits=rr["hits"],
                    sample=rr["sample"],
                )
            )
        finally:
            mm2.close()
            mm = None

    finally:
        if mm is not None:
            mm.close()

    summary = {
        "ok": all(r["ok"] for r in results),
        "passed": sum(1 for r in results if r["ok"]),
        "total": len(results),
        "model": pf.model_id,
        "project": PROJECT,
        "marker": marker,
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("---")
    print(json.dumps({"ok": summary["ok"], "passed": summary["passed"], "total": summary["total"]}, indent=2))
    print("wrote", OUT)
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
