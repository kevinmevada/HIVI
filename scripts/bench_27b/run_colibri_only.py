"""Colibrì-only Config A speed matrix for all three 27B models."""
from __future__ import annotations

import os

import csv
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

ROOT = Path(os.environ.get("HIVI_ROOT") or Path(__file__).resolve().parents[2])
OUT = ROOT / "benchmarks/27b_comparison"
LOG = OUT / "logs" / "colibri_only"
COLI = Path(
    os.environ.get("HIVI_COLI")
    or ROOT / "src" / "hivi" / "inference" / "colibri-LLM-main" / "c" / "coli"
)
PROMPT = "Reply with exactly one word: pong"
CTX = 4096
NGEN = 64
TEMP = 0.1
N_MEASURED = 5

ALL_MODELS = [
    {
        "id": "A",
        "name": "Unsloth-Qwen3.8-27B-UD-Q5_K_M",
        "quant": "Q5_K_M",
        "colibri": ROOT / "Models/General/unsloth/qwen3.8-27b-q5_k_m/Qwen3.8-27B-UD-Q5_K_M.colibri",
    },
    {
        "id": "B",
        "name": "0bserverx-RVN-Heretic-Q6_K",
        "quant": "Q6_K",
        "colibri": ROOT
        / "Models/General/0bserverx/qwen3.8-27b-heretic-abliterated-uncensored-q6_k/RVN-Q6_K.colibri",
    },
    {
        "id": "C",
        "name": "Coletti-Uncensored-Q6_K",
        "quant": "Q6_K",
        "colibri": ROOT
        / "Models/General/jonathan-coletti/qwen3.8-27b-uncensored-q6_k/Qwen3.8-27B-Uncensored-Q6_K.colibri",
    },
]


def gpu_snap() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        u, m, t, p = [x.strip() for x in out.split(",")]
        return {"util": float(u), "mem": float(m), "temp": float(t), "power": float(p)}
    except Exception as e:
        return {"error": str(e)}


def parse(text: str) -> dict:
    out = {"load_s": None, "rss_gb": None, "cuda": False, "speed_tps": None}
    m = re.search(r"resident weights loaded in\s*([0-9.]+)s", text)
    if m:
        out["load_s"] = float(m.group(1))
    m = re.search(r"RSS after load:\s*([0-9.]+)\s*GB", text)
    if m:
        out["rss_gb"] = float(m.group(1))
    m = re.search(r"Speed:\s*([0-9.]+)\s*tok/s", text)
    if m:
        out["speed_tps"] = float(m.group(1))
    if re.search(r"\[CUDA\].*device 0|dense trunk -> CUDA", text):
        out["cuda"] = True
    return out


def run_one(model_dir: Path, tag: str) -> dict:
    args = [
        "python",
        str(COLI),
        "run",
        "--model",
        str(model_dir),
        "--gpu",
        "auto",
        "--ctx",
        str(CTX),
        "--ngen",
        str(NGEN),
        "--temp",
        str(TEMP),
        "--policy",
        "balanced",
        "--auto-tier",
        PROMPT,
    ]
    pre = gpu_snap()
    t0 = time.perf_counter()
    peak = {"util": 0.0, "mem": 0.0}
    proc = subprocess.Popen(
        args,
        cwd=str(COLI.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    chunks: list[str] = []
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            chunks.append(line)
        g = gpu_snap()
        if "util" in g:
            peak["util"] = max(peak["util"], g["util"])
            peak["mem"] = max(peak["mem"], g["mem"])
    wall = time.perf_counter() - t0
    text = "".join(chunks)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / f"{tag}.log").write_text(text + f"\nWALL_S={wall:.3f}\n", encoding="utf-8")
    parsed = parse(text)
    ok = parsed["load_s"] is not None or "OpenAI-compatible" in text or "colibri v" in text.lower()
    if re.search(r"Traceback|failed to load|out of memory", text, re.I) and parsed["load_s"] is None:
        ok = False
    return {
        "tag": tag,
        "ok": ok,
        "wall_s": round(wall, 3),
        "peak_gpu_util": peak["util"],
        "peak_vram_mib": peak["mem"],
        "pre_gpu": pre,
        "post_gpu": gpu_snap(),
        **parsed,
        "returncode": proc.returncode,
    }


def stats(xs: list[float | None]) -> dict:
    vals = [x for x in xs if x is not None]
    if not vals:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {
        "median": round(median(vals), 3),
        "min": round(min(vals), 3),
        "max": round(max(vals), 3),
        "n": len(vals),
    }


def main() -> None:
    import sys

    only = [a.upper() for a in sys.argv[1:] if len(a) == 1 and a.upper() in "ABC"]
    models = [m for m in ALL_MODELS if (not only) or m["id"] in only]

    results = {
        "scope": "colibri_only_27b",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "settings": {"ctx": CTX, "ngen": NGEN, "temp": TEMP, "prompt": PROMPT},
        "models": {},
        "note": f"Running models: {[m['id'] for m in models]}",
    }
    # Keep prior A summary if present
    prior = OUT / "colibri_only_partial.json"
    if prior.exists() and "A" not in [m["id"] for m in models]:
        try:
            old = json.loads(prior.read_text(encoding="utf-8"))
            if old.get("models", {}).get("A"):
                results["models"]["A"] = old["models"]["A"]
        except Exception:
            pass
    # Also merge known A from COLIBRI results if needed
    raw_rows = []

    for model in models:
        mid = model["id"]
        print(f"\n======== COLIBRI {mid} {model['name']} ========", flush=True)
        if not model["colibri"].exists():
            results["models"][mid] = {"status": "MISSING", "path": str(model["colibri"])}
            continue

        print(" warmup...", flush=True)
        warm = run_one(model["colibri"], f"{mid}_warm")
        time.sleep(8)
        measured = []
        for i in range(1, N_MEASURED + 1):
            print(f" run {i}/{N_MEASURED}...", flush=True)
            r = run_one(model["colibri"], f"{mid}_r{i}")
            r["run"] = i
            measured.append(r)
            raw_rows.append(
                {
                    "model_id": mid,
                    "model": model["name"],
                    "quant": model["quant"],
                    "run": i,
                    "ok": r["ok"],
                    "wall_s": r["wall_s"],
                    "load_s": r["load_s"],
                    "rss_gb": r["rss_gb"],
                    "gen_tps": r["speed_tps"],
                    "peak_gpu_util": r["peak_gpu_util"],
                    "peak_vram_mib": r["peak_vram_mib"],
                    "cuda": r["cuda"],
                }
            )
            time.sleep(5)

        ok_runs = [r for r in measured if r["ok"]]
        summary = {
            "wall_s": stats([r["wall_s"] for r in ok_runs]),
            "load_s": stats([r["load_s"] for r in ok_runs]),
            "rss_gb": stats([r["rss_gb"] for r in ok_runs]),
            "gen_tps": stats([r["speed_tps"] for r in ok_runs]),
            "peak_vram_mib": stats([r["peak_vram_mib"] for r in ok_runs]),
            "peak_gpu_util": stats([r["peak_gpu_util"] for r in ok_runs]),
            "cuda_verified_runs": sum(1 for r in ok_runs if r["cuda"]),
            "n_ok": len(ok_runs),
            "n_fail": len(measured) - len(ok_runs),
        }
        results["models"][mid] = {
            "status": "COMPLETED",
            "name": model["name"],
            "quant": model["quant"],
            "path": str(model["colibri"]),
            "warmup": warm,
            "measured": measured,
            "summary": summary,
        }
        (OUT / "colibri_only_partial.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(" summary", summary, flush=True)
        time.sleep(10)

    results["ended_at"] = datetime.now(timezone.utc).isoformat()
    (OUT / "colibri_only_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    with (OUT / "colibri_only_raw.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model_id",
                "model",
                "quant",
                "run",
                "ok",
                "wall_s",
                "load_s",
                "rss_gb",
                "gen_tps",
                "peak_gpu_util",
                "peak_vram_mib",
                "cuda",
            ],
        )
        w.writeheader()
        for row in raw_rows:
            w.writerow(row)

    # Markdown report
    lines = [
        "# Colibrì-only 27B Speed Report",
        "",
        f"Generated: {results['ended_at']}",
        "",
        "## Settings",
        f"- Engine: Colibrì 2.1.0 (`coli run`, qwen36 for qwen35 GGUF)",
        f"- ctx={CTX}, ngen={NGEN}, temp={TEMP}",
        f"- Prompt: `{PROMPT}`",
        f"- Measured runs per model: {N_MEASURED} (+1 warmup)",
        "",
        "## Summary",
        "",
        "| Model | Quant | Status | Load median (s) | Wall median (s) | Gen t/s median | Peak VRAM (MiB) | CUDA runs |",
        "|-------|-------|--------|-----------------|-----------------|----------------|-----------------|-----------|",
    ]
    for mid, m in results["models"].items():
        if m.get("status") != "COMPLETED":
            lines.append(f"| {mid} | | {m.get('status')} | | | | | |")
            continue
        s = m["summary"]
        lines.append(
            f"| {m['name']} | {m['quant']} | COMPLETED | "
            f"{(s['load_s'] or {}).get('median')} | {(s['wall_s'] or {}).get('median')} | "
            f"{(s['gen_tps'] or {}).get('median')} | {(s['peak_vram_mib'] or {}).get('median')} | "
            f"{s.get('cuda_verified_runs')}/{s.get('n_ok')} |"
        )
    lines += [
        "",
        "## Notes",
        "- Each run is a fresh process (includes model load).",
        "- Dense trunk is placed on CUDA when free VRAM allows; remaining work stays on CPU/RAM.",
        "- JSON: `colibri_only_results.json` · CSV: `colibri_only_raw.csv`",
        "",
    ]
    (OUT / "COLIBRI_ONLY_SPEED_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("DONE", OUT / "COLIBRI_ONLY_SPEED_REPORT.md")


if __name__ == "__main__":
    main()
