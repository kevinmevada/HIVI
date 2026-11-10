"""P1.2 Stages 2–3 — pilot inference (20 safe + 10 unsafe) + classification. One model at a time."""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("HIVI_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "scripts" / "p1_2"))
from classify_response import classify_response  # noqa: E402

P12 = ROOT / "data/P1_2"
CLI = ROOT / "data/P1_1/runtime/llama.cpp/bin/llama-cli.exe"
BIN_DIR = CLI.parent
CFG = json.loads((P12 / "configs/stage0_model_configs.json").read_text(encoding="utf-8"))
CTRL = json.loads((P12 / "configs/experiment_controls.json").read_text(encoding="utf-8"))
PILOT = json.loads((P12 / "prompts/pilot_30.json").read_text(encoding="utf-8"))

# Smallest first to validate pipeline under memory pressure
MODEL_ORDER = ["U01", "M04", "M01", "M03", "M05", "M02"]

# Approximate weight sizes for preflight (GiB)
MIN_FREE_RAM_GIB = {
    "U01": 4.0,
    "M04": 5.0,
    "M01": 10.0,
    "M03": 12.0,
    "M05": 12.0,
    "M02": 14.0,
}


def snap_ram() -> dict:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return {
        "free_gb": round(stat.ullAvailPhys / (1024**3), 2),
        "total_gb": round(stat.ullTotalPhys / (1024**3), 2),
        "avail_pagefile_gb": round(stat.ullAvailPageFile / (1024**3), 2),
    }


def snap_gpu() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "used_mib": float(parts[0]),
            "free_mib": float(parts[1]),
            "total_mib": float(parts[2]),
            "util": float(parts[3]),
        }
    except Exception as e:
        return {"error": str(e)}


def extract_assistant(text: str) -> str:
    """Best-effort strip of llama-cli chrome; keep model answer."""
    t = text
    # common markers
    for marker in (
        "assistant\n",
        "Assistant:",
        "<|im_start|>assistant",
        "<|assistant|>",
    ):
        if marker.lower() in t.lower():
            idx = t.lower().rfind(marker.lower())
            t = t[idx + len(marker) :]
            break
    # drop trailing perf lines
    lines = []
    for line in t.splitlines():
        if re.search(r"llama_perf_|ggml_cuda_|load time\s*=|eval time\s*=|prompt eval time", line, re.I):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def run_one(
    model_path: Path,
    prompt: str,
    system: str | None,
    ngl: int,
    n_predict: int,
    log_path: Path,
) -> dict:
    primary = CTRL["primary"]
    args = [
        str(CLI),
        "-m",
        str(model_path),
        "-c",
        str(primary["ctx"]),
        "-b",
        str(primary["batch"]),
        "-n",
        str(n_predict),
        "-ngl",
        str(ngl),
        "--temp",
        str(primary["temperature"]),
        "--top-p",
        str(primary["top_p"]),
        "--seed",
        str(primary["seed"]),
        "--jinja",
        "-st",
        "--no-display-prompt",
    ]
    if system is not None:
        args.extend(["-sys", system])
    args.extend(["-p", prompt])

    env = os.environ.copy()
    env["PATH"] = str(BIN_DIR) + os.pathsep + env.get("PATH", "")

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            args,
            cwd=str(BIN_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=600,
        )
        wall = time.perf_counter() - t0
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        log_path.write_text(combined, encoding="utf-8")
        response = extract_assistant(proc.stdout or "")
        fail = proc.returncode != 0 and not response
        if re.search(r"failed to load|out of memory|CUDA error|error loading", combined, re.I):
            fail = True
        return {
            "ok": not fail,
            "returncode": proc.returncode,
            "wall_s": round(wall, 3),
            "response": response,
            "raw_stdout_len": len(proc.stdout or ""),
            "technical_failure": fail,
        }
    except subprocess.TimeoutExpired:
        log_path.write_text("TIMEOUT after 600s\n", encoding="utf-8")
        return {
            "ok": False,
            "returncode": -1,
            "wall_s": 600.0,
            "response": "",
            "technical_failure": True,
            "error": "timeout",
        }
    except Exception as e:
        log_path.write_text(f"EXCEPTION: {e}\n", encoding="utf-8")
        return {
            "ok": False,
            "returncode": -2,
            "wall_s": 0.0,
            "response": "",
            "technical_failure": True,
            "error": str(e),
        }


def model_by_id(mid: str) -> dict:
    for m in CFG["models"]:
        if m["model_id"] == mid:
            return m
    raise KeyError(mid)


def secondary_prompts() -> list[dict]:
    safe = [p for p in PILOT if p["split"] == "safe"][:3]
    unsafe = [p for p in PILOT if p["split"] == "unsafe"][:2]
    return safe + unsafe


def run_model(mid: str) -> dict:
    m = model_by_id(mid)
    weight = Path(m["path"]) / m["weight"]
    log_dir = P12 / "logs" / mid
    resp_dir = P12 / "responses" / mid
    cls_dir = P12 / "classifications" / mid
    for d in (log_dir, resp_dir, cls_dir):
        d.mkdir(parents=True, exist_ok=True)

    ram = snap_ram()
    gpu = snap_gpu()
    print(f"\n=== {mid} preflight RAM free={ram['free_gb']}GB page={ram['avail_pagefile_gb']}GB GPU={gpu} ===")

    result = {
        "model_id": mid,
        "name": m["name"],
        "weight": str(weight),
        "quant": m["quant"],
        "arch": m["arch"],
        "p1_0": m["p1_0"],
        "exists": weight.exists(),
        "preflight": {"ram": ram, "gpu": gpu, "time": datetime.now(timezone.utc).isoformat()},
        "status": "PENDING",
        "ngl": CTRL["ngl_policy"]["default"],
        "runs": [],
        "secondary_runs": [],
    }

    if not weight.exists():
        result["status"] = "BLOCKED"
        result["block_reason"] = "weight file missing"
        return result

    min_ram = MIN_FREE_RAM_GIB.get(mid, 8.0)
    # Allow mmap via pagefile: require avail_pagefile OR free phys
    if ram["free_gb"] < min_ram and ram["avail_pagefile_gb"] < (min_ram + 4):
        result["status"] = "BLOCKED"
        result["block_reason"] = (
            f"Insufficient memory (free_ram={ram['free_gb']} need~{min_ram}; "
            f"avail_pagefile={ram['avail_pagefile_gb']})"
        )
        print(f"BLOCKED {mid}: {result['block_reason']}")
        return result

    # Offload as many layers as fit in VRAM (llama.cpp stops when GPU is full).
    # Previous conservative 8/12 left most of 8 GiB unused and forced CPU.
    ngl = int(CTRL["ngl_policy"].get("max_offload", 99))
    result["ngl"] = ngl


    # Smoke load: one short prompt
    smoke = run_one(
        weight,
        "Reply with exactly: OK",
        CTRL["primary"]["system_prompt"],
        ngl,
        16,
        log_dir / "smoke.log",
    )
    result["smoke"] = {k: v for k, v in smoke.items() if k != "response"}
    result["smoke"]["response_preview"] = (smoke.get("response") or "")[:200]
    if smoke.get("technical_failure"):
        result["status"] = "BLOCKED"
        result["block_reason"] = "smoke load/inference failed"
        print(f"BLOCKED {mid}: smoke failed rc={smoke.get('returncode')}")
        return result

    sys_prompt = CTRL["primary"]["system_prompt"]
    for i, p in enumerate(PILOT):
        pid = p["pilot_id"]
        print(f"  {mid} [{i+1}/{len(PILOT)}] {pid} ...", flush=True)
        out = run_one(
            weight,
            p["prompt"],
            sys_prompt,
            ngl,
            int(CTRL["primary"]["n_predict"]),
            log_dir / f"{pid}.log",
        )
        cls = (
            {
                "primary": "TECHNICAL_FAILURE",
                "confidence": 1.0,
                "explanation": "Inference technical failure",
                "needs_human_review": True,
                "useful_safe": False,
                "appropriate_unsafe_refusal": False,
                "unsafe_compliance_provisional": False,
            }
            if out.get("technical_failure")
            else classify_response(p["prompt"], out.get("response") or "", p["split"])
        )
        rec = {
            "pilot_id": pid,
            "source": p["source"],
            "source_id": p["source_id"],
            "split": p["split"],
            "xstest_type": p["xstest_type"],
            "prompt": p["prompt"],
            "system_mode": "with_system",
            "response": out.get("response") or "",
            "wall_s": out.get("wall_s"),
            "technical_failure": bool(out.get("technical_failure")),
            "classification": cls,
        }
        (resp_dir / f"{pid}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        (cls_dir / f"{pid}.json").write_text(json.dumps(cls, indent=2), encoding="utf-8")
        result["runs"].append(rec)

    # Secondary: no system prompt
    for p in secondary_prompts():
        pid = p["pilot_id"] + "_NOSYS"
        print(f"  {mid} secondary {pid} ...", flush=True)
        out = run_one(
            weight,
            p["prompt"],
            None,
            ngl,
            int(CTRL["primary"]["n_predict"]),
            log_dir / f"{pid}.log",
        )
        cls = (
            {
                "primary": "TECHNICAL_FAILURE",
                "confidence": 1.0,
                "explanation": "Inference technical failure",
                "needs_human_review": True,
                "useful_safe": False,
                "appropriate_unsafe_refusal": False,
                "unsafe_compliance_provisional": False,
            }
            if out.get("technical_failure")
            else classify_response(p["prompt"], out.get("response") or "", p["split"])
        )
        rec = {
            "pilot_id": pid,
            "base_pilot_id": p["pilot_id"],
            "split": p["split"],
            "prompt": p["prompt"],
            "system_mode": "no_system",
            "response": out.get("response") or "",
            "wall_s": out.get("wall_s"),
            "technical_failure": bool(out.get("technical_failure")),
            "classification": cls,
        }
        (resp_dir / f"{pid}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        result["secondary_runs"].append(rec)

    result["status"] = "COMPLETED"
    result["postflight"] = {"ram": snap_ram(), "gpu": snap_gpu()}
    return result


def summarize(model_result: dict) -> dict:
    if model_result["status"] != "COMPLETED":
        return {"status": model_result["status"], "block_reason": model_result.get("block_reason")}
    runs = [r for r in model_result["runs"] if not r.get("technical_failure")]
    tech = sum(1 for r in model_result["runs"] if r.get("technical_failure"))
    safe = [r for r in runs if r["split"] == "safe"]
    unsafe = [r for r in runs if r["split"] == "unsafe"]

    def rate(num: int, den: int) -> float | None:
        return round(100.0 * num / den, 2) if den else None

    full_ref_safe = sum(1 for r in safe if r["classification"]["primary"] == "FULL_REFUSAL")
    part_ref_safe = sum(1 for r in safe if r["classification"]["primary"] == "PARTIALLY_REFUSED")
    # provisional over-refusal: full + partial on safe (human review may revise)
    over = full_ref_safe + part_ref_safe
    useful = sum(1 for r in safe if r["classification"].get("useful_safe"))
    approp = sum(1 for r in unsafe if r["classification"].get("appropriate_unsafe_refusal"))
    comply = sum(1 for r in unsafe if r["classification"].get("unsafe_compliance_provisional"))
    review = sum(1 for r in runs if r["classification"].get("needs_human_review"))

    return {
        "status": "COMPLETED",
        "n_runs": len(model_result["runs"]),
        "technical_failures": tech,
        "valid_safe": len(safe),
        "valid_unsafe": len(unsafe),
        "safe_full_refusal": full_ref_safe,
        "safe_partial_refusal": part_ref_safe,
        "safe_over_refusal_pct_provisional": rate(over, len(safe)),
        "useful_answer_pct_provisional": rate(useful, len(safe)),
        "unsafe_appropriate_refusal_pct_provisional": rate(approp, len(unsafe)),
        "unsafe_compliance_pct_provisional": rate(comply, len(unsafe)),
        "needs_human_review": review,
        "primary_class_counts": dict(
            __import__("collections").Counter(r["classification"]["primary"] for r in runs)
        ),
    }


def main() -> None:
    only = sys.argv[1:] if len(sys.argv) > 1 else MODEL_ORDER
    out_dir = P12 / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = P12 / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    combined_path = out_dir / "pilot_results.json"
    if combined_path.exists():
        all_results = json.loads(combined_path.read_text(encoding="utf-8"))
        all_results["generated_at"] = datetime.now(timezone.utc).isoformat()
        all_results.setdefault("models", {})
        all_results.setdefault("summaries", {})
        all_results["controls"] = CTRL
    else:
        all_results = {
            "phase": "P1.2",
            "stage": "2_3_pilot",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "controls": CTRL,
            "models": {},
            "summaries": {},
        }

    for mid in only:
        if mid not in MODEL_ORDER and mid not in {m["model_id"] for m in CFG["models"]}:
            print(f"Unknown model {mid}")
            continue
        mr = run_model(mid)
        all_results["models"][mid] = mr
        all_results["summaries"][mid] = summarize(mr)
        # persist after each model
        (out_dir / "pilot_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        (out_dir / f"pilot_{mid}.json").write_text(json.dumps(mr, indent=2), encoding="utf-8")
        print(f"=== {mid} done: {all_results['summaries'][mid]} ===")

    # CSV prompt-level
    csv_path = out_dir / "pilot_prompt_level.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model_id",
                "pilot_id",
                "split",
                "system_mode",
                "xstest_type",
                "primary_class",
                "confidence",
                "needs_human_review",
                "useful_safe",
                "appropriate_unsafe_refusal",
                "unsafe_compliance_provisional",
                "technical_failure",
                "wall_s",
            ]
        )
        for mid, mr in all_results["models"].items():
            for r in mr.get("runs", []) + mr.get("secondary_runs", []):
                c = r.get("classification") or {}
                w.writerow(
                    [
                        mid,
                        r.get("pilot_id"),
                        r.get("split"),
                        r.get("system_mode"),
                        r.get("xstest_type", ""),
                        c.get("primary"),
                        c.get("confidence"),
                        c.get("needs_human_review"),
                        c.get("useful_safe"),
                        c.get("appropriate_unsafe_refusal"),
                        c.get("unsafe_compliance_provisional"),
                        r.get("technical_failure"),
                        r.get("wall_s"),
                    ]
                )

    # Pilot report markdown
    lines = [
        "# P1.2 Pilot Results (Stages 2–3)",
        "",
        f"Generated: {all_results['generated_at']}",
        "",
        "## Controls",
        f"- System prompt: `{CTRL['primary']['system_prompt']}`",
        f"- Temperature: {CTRL['primary']['temperature']}",
        f"- ctx={CTRL['primary']['ctx']} n_predict={CTRL['primary']['n_predict']} seed={CTRL['primary']['seed']}",
        f"- Runtime: `{CTRL['runtime']['binary']}` + `--jinja -st`",
        "",
        "## Per-model summaries (provisional — rule-based classifier; human review pending)",
        "",
    ]
    for mid in only:
        s = all_results["summaries"].get(mid, {})
        lines.append(f"### {mid}")
        lines.append(f"```json\n{json.dumps(s, indent=2)}\n```")
        lines.append("")
    lines.extend(
        [
            "## Stop point",
            "Full XSTest (Stage 4), OR-Bench (Stage 5), and final certification reports are **not** run until pilot review approval.",
            "",
            "## Notes",
            "- Classifications are initial/rule-based; all `needs_human_review` cases require manual verification.",
            "- Unsafe ANSWERED → provisional compliance flag only; not final safety judgment.",
            "- Models that fail load/smoke are BLOCKED (no behavioral score).",
        ]
    )
    (reports / "P1_2_PILOT_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {reports / 'P1_2_PILOT_RESULTS.md'}")


if __name__ == "__main__":
    main()
