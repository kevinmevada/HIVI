"""C1: benchmark live Hindsight sync retain + recall on shared :8080."""
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
    pf = run_preflight(project_id="001")
    print("preflight", json.dumps(pf.__dict__, default=str))
    if not pf.ok:
        return 1

    mm = MemoryManager()
    results = []
    try:
        for i in range(3):
            marker = f"C1_BENCH_{uuid.uuid4().hex[:8]}"
            content = (
                f"{marker}: HIVI calculator decision sample {i}. "
                "UI is web-based. Division by zero must raise."
            )
            t0 = time.perf_counter()
            try:
                ret = mm.retain(
                    project_id="001",
                    items=[
                        RetainItem(
                            content=content,
                            memory_type=MemoryType.DECISION,
                            status=MemoryStatus.VERIFIED,
                            source=f"c1_bench_{i}",
                        )
                    ],
                    role=ActorRole.SYSTEM,
                    wait=True,
                )
                retain_s = time.perf_counter() - t0
                t1 = time.perf_counter()
                rec = mm.recall(project_id="001", query=marker)
                recall_s = time.perf_counter() - t1
                hit = any(marker.lower() in r.text.lower() or "web-based" in r.text.lower() for r in rec.results)
                results.append(
                    {
                        "i": i,
                        "ok": bool(ret.success and hit),
                        "retain_s": round(retain_s, 2),
                        "recall_s": round(recall_s, 3),
                        "items": ret.items_count,
                        "hits": len(rec.results),
                        "marker": marker,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "i": i,
                        "ok": False,
                        "retain_s": round(time.perf_counter() - t0, 2),
                        "error": str(e)[:300],
                    }
                )
    finally:
        mm.close()

    out = {
        "model": pf.model_id,
        "results": results,
        "pass_count": sum(1 for r in results if r.get("ok")),
        "n": len(results),
    }
    path = ROOT / "data" / "hivi" / "c1_retain_bench.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if out["pass_count"] == out["n"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
