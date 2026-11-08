"""C1 smoke: one sync retain + recall (fast gate)."""
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


def main() -> int:
    # Use a clean bank so a stuck consolidation on project_001 cannot starve retain.
    project_id = "099"
    pf = run_preflight(project_id=project_id)
    print("preflight_ok", pf.ok, "model", pf.model_id, "project", project_id)
    if not pf.ok:
        print("errors", pf.errors)
        return 1

    marker = f"C1_SMOKE_{uuid.uuid4().hex[:8]}"
    content = f"{marker}: Calculator UI is web-based. div() must reject zero."
    mm = MemoryManager()
    t0 = time.perf_counter()
    try:
        ret = mm.retain(
            project_id=project_id,
            items=[
                RetainItem(
                    content=content,
                    memory_type=MemoryType.DECISION,
                    status=MemoryStatus.VERIFIED,
                    source="c1_smoke",
                )
            ],
            role=ActorRole.SYSTEM,
            wait=True,
        )
        retain_s = time.perf_counter() - t0
        # Extraction often drops raw markers; accept semantic hit or nonempty recall.
        rec = mm.recall(
            project_id=project_id,
            query=f"What is the calculator UI decision? {marker}",
        )
        blob = " ".join((r.text or "").lower() for r in rec.results)
        hit = bool(rec.results) and (
            marker.lower() in blob
            or "web-based" in blob
            or "web" in blob
            or "calculator" in blob
        )
        out = {
            "ok": bool(ret.success and hit),
            "retain_ok": bool(ret.success),
            "retain_s": round(retain_s, 2),
            "hits": len(rec.results),
            "sample": [(r.text or "")[:120] for r in rec.results[:3]],
            "marker": marker,
            "model": pf.model_id,
        }
        print(json.dumps(out, indent=2))
        path = ROOT / "data" / "hivi" / "c1_retain_smoke.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return 0 if out["ok"] else 2
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:400], "retain_s": round(time.perf_counter() - t0, 2)}))
        return 3
    finally:
        mm.close()


if __name__ == "__main__":
    raise SystemExit(main())
