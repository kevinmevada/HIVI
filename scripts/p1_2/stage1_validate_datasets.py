"""P1.2 Stage 1 — validate XSTest / OR-Bench integrity and write stage1 report."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\mevad\Desktop\HIVI")
P12 = ROOT / "data/P1_2"
XSTEST = P12 / "datasets/xstest/xstest_prompts.csv"
OR_HARD = P12 / "datasets/or_bench/or-bench-hard-1k.csv"
OR_TOXIC = P12 / "datasets/or_bench/or-bench-toxic.csv"
OUT = P12 / "reports"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "phase": "P1.2",
        "stage": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "integrity_policy": "Original benchmark prompts retained; no silent alterations",
        "datasets": {},
        "issues": [],
        "status": "PASS",
    }

    # XSTest
    xs_rows = list(csv.DictReader(XSTEST.open(encoding="utf-8")))
    xs_labels = Counter(r["label"] for r in xs_rows)
    xs_types = Counter(r["type"] for r in xs_rows)
    empty_xs = sum(1 for r in xs_rows if not (r.get("prompt") or "").strip())
    report["datasets"]["xstest"] = {
        "path": str(XSTEST),
        "source": "https://github.com/paul-rottger/xstest",
        "file": "xstest_prompts.csv",
        "rows": len(xs_rows),
        "safe": xs_labels.get("safe", 0),
        "unsafe": xs_labels.get("unsafe", 0),
        "types": dict(xs_types),
        "empty_prompts": empty_xs,
        "sha256": sha256(XSTEST),
        "expected": {"rows": 450, "safe": 250, "unsafe": 200},
        "match_expected": len(xs_rows) == 450
        and xs_labels.get("safe") == 250
        and xs_labels.get("unsafe") == 200,
    }
    if not report["datasets"]["xstest"]["match_expected"] or empty_xs:
        report["issues"].append("XSTest count/empty mismatch")
        report["status"] = "FAIL"

    # OR-Bench hard
    hard_rows = list(csv.DictReader(OR_HARD.open(encoding="utf-8")))
    hard_cats = Counter(r.get("category", "") for r in hard_rows)
    empty_h = sum(1 for r in hard_rows if not (r.get("prompt") or "").strip())
    report["datasets"]["or_bench_hard"] = {
        "path": str(OR_HARD),
        "source": "https://huggingface.co/datasets/orbench-llm/or-bench",
        "file": "or-bench-hard-1k.csv",
        "rows": len(hard_rows),
        "categories": dict(hard_cats),
        "empty_prompts": empty_h,
        "sha256": sha256(OR_HARD),
        "note": "Filename says 1k; upstream CSV currently has 1319 rows — documented, not truncated",
        "deferred_to_stage": 5,
    }
    if empty_h:
        report["issues"].append("OR-Bench hard has empty prompts")
        report["status"] = "FAIL"

    # OR-Bench toxic
    tox_rows = list(csv.DictReader(OR_TOXIC.open(encoding="utf-8")))
    tox_cats = Counter(r.get("category", "") for r in tox_rows)
    empty_t = sum(1 for r in tox_rows if not (r.get("prompt") or "").strip())
    report["datasets"]["or_bench_toxic"] = {
        "path": str(OR_TOXIC),
        "source": "https://huggingface.co/datasets/orbench-llm/or-bench",
        "file": "or-bench-toxic.csv",
        "rows": len(tox_rows),
        "categories": dict(tox_cats),
        "empty_prompts": empty_t,
        "sha256": sha256(OR_TOXIC),
        "role": "unsafe contrast subset for Stage 5",
        "deferred_to_stage": 5,
    }
    if empty_t:
        report["issues"].append("OR-Bench toxic has empty prompts")
        report["status"] = "FAIL"

    pilot = json.loads((P12 / "prompts/pilot_30.json").read_text(encoding="utf-8"))
    custom = json.loads((P12 / "prompts/hivi_custom_safe.json").read_text(encoding="utf-8"))
    report["pilot_set"] = {
        "path": str(P12 / "prompts/pilot_30.json"),
        "n": len(pilot),
        "safe": sum(1 for p in pilot if p["split"] == "safe"),
        "unsafe": sum(1 for p in pilot if p["split"] == "unsafe"),
        "source": "XSTest stratified sample; prompt text unaltered",
    }
    report["hivi_custom"] = {
        "path": str(P12 / "prompts/hivi_custom_safe.json"),
        "n": len(custom),
        "note": "Separate from upstream; reserved for category coverage beyond pilot",
    }

    (OUT / "P1_2_STAGE1_DATASET_VALIDATION.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    md = [
        "# P1.2 Stage 1 — Dataset Validation",
        "",
        f"**Status:** {report['status']}",
        f"**Generated:** {report['generated_at']}",
        "",
        "## XSTest",
        f"- Rows: {report['datasets']['xstest']['rows']} (expected 450)",
        f"- Safe/Unsafe: {report['datasets']['xstest']['safe']}/{report['datasets']['xstest']['unsafe']}",
        f"- Match expected: {report['datasets']['xstest']['match_expected']}",
        f"- SHA256: `{report['datasets']['xstest']['sha256']}`",
        "",
        "## OR-Bench",
        f"- Hard CSV rows: {report['datasets']['or_bench_hard']['rows']} (filename 1k; not truncated)",
        f"- Toxic CSV rows: {report['datasets']['or_bench_toxic']['rows']}",
        "- Full OR-Bench evaluation deferred to Stage 5 (post-pilot approval)",
        "",
        "## Pilot set",
        f"- {report['pilot_set']['n']} prompts ({report['pilot_set']['safe']} safe / {report['pilot_set']['unsafe']} unsafe)",
        "- Source: XSTest stratified; original wording preserved",
        "",
        "## HIVI custom",
        f"- {report['hivi_custom']['n']} safe category prompts (A–E) stored separately",
        "",
        "## Issues",
    ]
    md.extend([f"- {i}" for i in report["issues"]] or ["- None"])
    (OUT / "P1_2_STAGE1_DATASET_VALIDATION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "issues": report["issues"]}, indent=2))


if __name__ == "__main__":
    main()
