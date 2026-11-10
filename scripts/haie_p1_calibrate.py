"""HAIE V1.2 P1 — confidence calibration runner (measurement only).

- Reads immutable V1.2_P0_BASELINE.json (never writes it)
- Runs 8/8 smoke acceptance as regression gate
- Evaluates labeled_set.jsonl → calibration report under data/haie/p1/reports/

Usage:
  python scripts/haie_p1_calibrate.py              # smoke + full labeled set
  python scripts/haie_p1_calibrate.py --limit 24   # smoke + subset
  python scripts/haie_p1_calibrate.py --smoke-only
"""
from __future__ import annotations

import os

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("HIVI_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "interfaces" / "hivi-chat"))

from haie.calibration.evaluate import (  # noqa: E402
    BASELINE_PATH,
    LABELED_PATH,
    baseline_fingerprint,
    evaluate_labeled_set,
    load_labeled_set,
    write_report,
)


def status() -> dict:
    try:
        with urllib.request.urlopen("http://127.0.0.1:7860/api/status", timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"error": str(e)}


def run_smoke(*, retries: int = 1) -> dict:
    """P0 8/8 gate. Retries only inside this measurement runner — production routing untouched."""
    from haie import plan_request

    cases = [
        ("greeting", "Hey", "fast"),
        ("simple_coding", "Write Hello World in python", "fast"),
        ("simple_math", "What is 2+2?", "fast"),
        ("explain_optimize", "explain and optimize this python function for sorting", "semantic"),
        ("architecture", "Design a trading engine", "semantic"),
        ("compare_moe", "Compare MoE and Dense models for local 8GB inference", "semantic"),
        ("debug_complex", "Debug this complex concurrency problem in a thread pool", "semantic"),
        ("ambiguous_multi", "explain and also implement a cache and compare two approaches", "semantic"),
    ]
    st = status()
    engine = st.get("engine") or "llama"
    model_id = st.get("model_id")
    backend = st.get("backend") or "http://127.0.0.1:8080"
    rows = []
    for name, text, expect in cases:
        ok = False
        actual = ""
        attempts = 0
        for attempt in range(1 + max(0, retries)):
            attempts = attempt + 1
            body = {"model": "local", "messages": [{"role": "user", "content": text}], "stream": False}
            _, meta = plan_request(body, engine=engine, backend_base=backend, model_id=model_id)
            actual = meta.path
            ok = meta.path == expect
            if ok:
                break
        rows.append(
            {
                "case": name,
                "expected": expect,
                "actual": actual,
                "ok": ok,
                "attempts": attempts,
            }
        )
        mark = "PASS" if ok else "FAIL"
        retry_note = f" attempts={attempts}" if attempts > 1 else ""
        print(f"[smoke {mark}] {name:16} expect={expect:8} got={actual}{retry_note}")
    passed = sum(1 for r in rows if r["ok"])
    print(f"smoke acceptance {passed}/{len(cases)}")
    return {
        "passed": passed,
        "total": len(cases),
        "ok": passed == len(cases),
        "cases": rows,
        "status": {k: st.get(k) for k in ("model_id", "engine", "status", "healthy")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="HAIE V1.2 P1 calibration (measurement only)")
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only N labeled rows")
    ap.add_argument(
        "--no-stratify",
        action="store_true",
        help="With --limit, take file prefix instead of round-robin across splits",
    )
    ap.add_argument("--smoke-only", action="store_true", help="Only run 8/8 smoke gate")
    ap.add_argument("--skip-smoke", action="store_true", help="Skip smoke (not recommended)")
    args = ap.parse_args()

    if not BASELINE_PATH.is_file():
        print("ERROR: missing frozen baseline", BASELINE_PATH)
        return 2
    fp_before = baseline_fingerprint()
    if fp_before.get("status") != "frozen":
        print("ERROR: baseline status is not frozen:", fp_before)
        return 2
    print("baseline", json.dumps(fp_before, indent=2))

    if not LABELED_PATH.is_file():
        print("ERROR: labeled set missing — run scripts/haie_p1_build_labeled_set.py first")
        return 2

    smoke = None
    if not args.skip_smoke:
        smoke = run_smoke()
        if not smoke["ok"]:
            print("ABORT: smoke regression — calibration not run (preserve P0 gate)")
            return 1

    if args.smoke_only:
        out = {
            "baseline": fp_before,
            "smoke": smoke,
            "note": "smoke-only; labeled calibration skipped",
        }
        path = write_report(out, ROOT / "data" / "haie" / "p1" / "reports" / "smoke_gate_latest.json")
        print("wrote", path)
        fp_after = baseline_fingerprint()
        if fp_after["sha256"] != fp_before["sha256"]:
            print("ERROR: baseline file changed during run — treat as incident")
            return 3
        return 0

    st = status()
    rows = load_labeled_set()
    print(f"labeled_set n={len(rows)} limit={args.limit}")
    report = evaluate_labeled_set(
        rows=rows,
        backend_base=st.get("backend") or "http://127.0.0.1:8080",
        engine=st.get("engine") or "llama",
        model_id=st.get("model_id"),
        limit=args.limit,
        stratified=not args.no_stratify,
    )
    report["smoke"] = smoke
    path = write_report(report)

    m = report["metrics"]
    ts = report["targets_status"]
    print("\n=== P1 calibration summary ===")
    print(f"evaluated {report['dataset']['evaluated']}/{report['dataset']['total_labeled']}")
    print(f"exact_intent_acc={m['exact_intent_accuracy']:.3f}  primary_acc={m['primary_intent_accuracy']:.3f}")
    print(f"macro_f1={m['macro_f1']:.3f} (target >= {report['targets']['macro_f1']})  pass={ts['macro_f1']['pass']}")
    print(f"ECE={m['ece']:.3f} (target <= {report['targets']['ece']})  pass={ts['ece']['pass']}")
    print(f"fallback_rate={m['fallback_rate']:.3f}  classify_failure_rate={m['classify_failure_rate']:.3f}")
    print(f"param_profile_mismatch_rate={m['param_profile_mismatch_rate']:.3f}")
    print(f"mean_conf correct={m['mean_confidence_when_correct']:.3f} wrong={m['mean_confidence_when_wrong']:.3f}")
    print("wrote", path)
    print("summary", path.with_name(path.stem + "_summary.json"))

    fp_after = baseline_fingerprint()
    if fp_after["sha256"] != fp_before["sha256"]:
        print("ERROR: baseline file changed during run — treat as incident")
        return 3

    # Measurement project: non-zero exit only on smoke failure / baseline mutate.
    # Target misses are reported, not gating, until labels + model are validated.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
