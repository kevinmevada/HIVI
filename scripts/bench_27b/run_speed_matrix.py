"""27B speed matrix: Config A, 1 warmup + 5 measured runs per model×runtime."""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

ROOT = Path(r"C:\Users\mevad\Desktop\HIVI")
OUT = ROOT / "benchmarks/27b_comparison"
LOG = OUT / "logs"
RAW = OUT / "raw"
CLI = ROOT / "data/P1_1/runtime/llama.cpp/bin/llama-cli.exe"
BIN = CLI.parent
COLI = Path(r"C:\Users\mevad\Downloads\colibri-main\colibri-main\c\coli")
CFG = json.loads((OUT / "runtime_configurations.json").read_text(encoding="utf-8"))["config_A_conservative"]
PROMPT = (OUT / "prompts/speed_prompt.txt").read_text(encoding="utf-8").strip()

MODELS = [
    {
        "id": "A",
        "name": "Qwen3.8-27B-UD-Q5_K_M",
        "quant": "Q5_K_M",
        "gguf": ROOT / "Models/General/unsloth/qwen3.8-27b-q5_k_m/Qwen3.8-27B-UD-Q5_K_M.gguf",
        "colibri": ROOT / "Models/General/unsloth/qwen3.8-27b-q5_k_m/Qwen3.8-27B-UD-Q5_K_M.colibri",
    },
    {
        "id": "B",
        "name": "RVN-Heretic-Q6_K",
        "quant": "Q6_K",
        "gguf": ROOT / "Models/General/0bserverx/qwen3.8-27b-heretic-abliterated-uncensored-q6_k/RVN-Q6_K.gguf",
        "colibri": ROOT / "Models/General/0bserverx/qwen3.8-27b-heretic-abliterated-uncensored-q6_k/RVN-Q6_K.colibri",
    },
    {
        "id": "C",
        "name": "Coletti-Uncensored-Q6_K",
        "quant": "Q6_K",
        "gguf": ROOT / "Models/General/jonathan-coletti/qwen3.8-27b-uncensored-q6_k/Qwen3.8-27B-Uncensored-Q6_K.gguf",
        "colibri": ROOT / "Models/General/jonathan-coletti/qwen3.8-27b-uncensored-q6_k/Qwen3.8-27B-Uncensored-Q6_K.colibri",
    },
]


def gpu_snap() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        u, m, t, temp, pwr = [x.strip() for x in out.split(",")]
        return {
            "util": float(u),
            "mem_used": float(m),
            "mem_total": float(t),
            "temp": float(temp),
            "power": float(pwr),
        }
    except Exception as e:
        return {"error": str(e)}


def ram_free_gb() -> float:
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

    s = MEMORYSTATUSEX()
    s.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return round(s.ullAvailPhys / (1024**3), 2)


def parse_llama(text: str) -> dict:
    out = {"prompt_tps": None, "gen_tps": None, "load_ms": None}
    m = re.search(r"Prompt:\s*([0-9.]+)\s*t/s", text)
    if m:
        out["prompt_tps"] = float(m.group(1))
    m = re.search(r"Generation:\s*([0-9.]+)\s*t/s", text)
    if m:
        out["gen_tps"] = float(m.group(1))
    m = re.search(r"load time\s*=\s*([0-9.]+)", text, re.I)
    if m:
        out["load_ms"] = float(m.group(1))
    return out


def parse_colibri(text: str) -> dict:
    out = {"load_s": None, "speed_tps": None, "rss_gb": None, "cuda": False}
    m = re.search(r"resident weights loaded in\s*([0-9.]+)s", text)
    if m:
        out["load_s"] = float(m.group(1))
    m = re.search(r"RSS after load:\s*([0-9.]+)\s*GB", text)
    if m:
        out["rss_gb"] = float(m.group(1))
    m = re.search(r"Speed:\s*([0-9.]+)\s*tok/s", text)
    if m:
        out["speed_tps"] = float(m.group(1))
    if re.search(r"\[CUDA\]|dense trunk -> CUDA", text):
        out["cuda"] = True
    return out


def run_llama(model_path: Path, tag: str, n_predict: int) -> dict:
    args = [
        str(CLI),
        "-m",
        str(model_path),
        "-c",
        str(CFG["ctx"]),
        "-n",
        str(n_predict),
        "-ngl",
        str(CFG["llama"]["ngl"]),
        "-ctk",
        CFG["llama"]["cache_type_k"],
        "-ctv",
        CFG["llama"]["cache_type_v"],
        "-fa",
        CFG["llama"]["flash_attn"],
        "-b",
        str(CFG["llama"]["batch"]),
        "-ub",
        str(CFG["llama"]["ubatch"]),
        "--temp",
        str(CFG["temperature"]),
        "--seed",
        str(CFG["seed"]),
        "--jinja",
        "-st",
        "--no-display-prompt",
        "-sys",
        "Answer briefly and directly.",
        "-p",
        PROMPT,
    ]
    env = os.environ.copy()
    env["PATH"] = str(BIN) + os.pathsep + env.get("PATH", "")
    pre = {"gpu": gpu_snap(), "ram_free_gb": ram_free_gb()}
    t0 = time.perf_counter()
    peak = {"util": 0.0, "mem": 0.0}
    proc = subprocess.Popen(
        args,
        cwd=str(BIN),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    chunks = []
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
            peak["mem"] = max(peak["mem"], g["mem_used"])
    wall = time.perf_counter() - t0
    text = "".join(chunks)
    log_path = LOG / f"{tag}.log"
    log_path.write_text(text + f"\nWALL_S={wall:.3f}\n", encoding="utf-8")
    parsed = parse_llama(text)
    fail = proc.returncode not in (0, None) and "Generation:" not in text
    if re.search(r"out of memory|CUDA error|failed to load", text, re.I):
        fail = True
    return {
        "runtime": "llama.cpp",
        "tag": tag,
        "wall_s": round(wall, 3),
        "ok": not fail,
        "pre": pre,
        "peak_gpu_util": peak["util"],
        "peak_vram_mib": peak["mem"],
        "post_gpu": gpu_snap(),
        "post_ram_free_gb": ram_free_gb(),
        **parsed,
        "returncode": proc.returncode,
    }


def run_colibri(model_dir: Path, tag: str, n_predict: int) -> dict:
    args = [
        "python",
        str(COLI),
        "run",
        "--model",
        str(model_dir),
        "--gpu",
        "auto",
        "--ctx",
        str(CFG["ctx"]),
        "--ngen",
        str(n_predict),
        "--temp",
        str(CFG["temperature"]),
        "--policy",
        "balanced",
        "--auto-tier",
        PROMPT,
    ]
    pre = {"gpu": gpu_snap(), "ram_free_gb": ram_free_gb()}
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
    chunks = []
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
            peak["mem"] = max(peak["mem"], g["mem_used"])
    wall = time.perf_counter() - t0
    text = "".join(chunks)
    log_path = LOG / f"{tag}.log"
    log_path.write_text(text + f"\nWALL_S={wall:.3f}\n", encoding="utf-8")
    parsed = parse_colibri(text)
    fail = proc.returncode not in (0, None) and not parsed.get("cuda") and "OpenAI-compatible" not in text
    # coli run often returns 0 even with warnings
    if re.search(r"Traceback|failed to load|out of memory|CUDA error", text, re.I) and "chat/completions" not in text:
        fail = True
    # success if API completed or Speed printed or wall produced response section
    ok = (not fail) and (
        "chat/completions" in text or parsed.get("load_s") is not None or "colibri v" in text.lower()
    )
    return {
        "runtime": "colibri",
        "tag": tag,
        "wall_s": round(wall, 3),
        "ok": ok,
        "pre": pre,
        "peak_gpu_util": peak["util"],
        "peak_vram_mib": peak["mem"],
        "post_gpu": gpu_snap(),
        "post_ram_free_gb": ram_free_gb(),
        "prompt_tps": None,
        "gen_tps": parsed.get("speed_tps"),
        "load_s": parsed.get("load_s"),
        "rss_gb": parsed.get("rss_gb"),
        "cuda_verified": parsed.get("cuda"),
        "returncode": proc.returncode,
    }


def settle(seconds: int = 8) -> None:
    time.sleep(seconds)


def summarize(runs: list[dict]) -> dict:
    walls = [r["wall_s"] for r in runs if r.get("ok")]
    gens = [r["gen_tps"] for r in runs if r.get("ok") and r.get("gen_tps")]
    prompts = [r["prompt_tps"] for r in runs if r.get("ok") and r.get("prompt_tps")]
    vrams = [r["peak_vram_mib"] for r in runs if r.get("ok")]
    utils = [r["peak_gpu_util"] for r in runs if r.get("ok")]

    def stats(xs):
        if not xs:
            return {"median": None, "min": None, "max": None}
        return {"median": round(median(xs), 3), "min": round(min(xs), 3), "max": round(max(xs), 3)}

    return {
        "n_ok": sum(1 for r in runs if r.get("ok")),
        "n_fail": sum(1 for r in runs if not r.get("ok")),
        "wall_s": stats(walls),
        "gen_tps": stats(gens),
        "prompt_tps": stats(prompts),
        "peak_vram_mib": stats(vrams),
        "peak_gpu_util": stats(utils),
    }


def main() -> None:
    import sys

    LOG.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    all_rows = []
    partial_path = OUT / "results_speed_partial.json"
    if partial_path.exists():
        results = json.loads(partial_path.read_text(encoding="utf-8"))
        results.setdefault("combos", {})
    else:
        results = {
            "phase": "3_speed_config_A",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": CFG,
            "prompt": PROMPT,
            "combos": {},
        }

    skip = set(sys.argv[1:])  # optional: keys already done
    order = []
    for m in MODELS:
        order.append((m, "colibri"))
        order.append((m, "llama.cpp"))

    for model, runtime in order:
        key = f"{model['id']}_{runtime.replace('.', '')}"
        if key in skip or (
            results.get("combos", {}).get(key, {}).get("status") == "COMPLETED"
            and key not in skip
            and "--force" not in sys.argv
        ):
            # resume: skip completed unless --force
            if results.get("combos", {}).get(key, {}).get("status") == "COMPLETED" and "--force" not in sys.argv:
                print(f"\n======== {key} SKIP (already COMPLETED) ========", flush=True)
                continue
        print(f"\n======== {key} ========", flush=True)
        if ram_free_gb() < 6:
            print(f"BLOCKED low RAM {ram_free_gb()}GB", flush=True)
            results["combos"][key] = {"status": "BLOCKED", "reason": f"ram_free={ram_free_gb()}"}
            continue

        measured = []
        warmup = None
        try:
            if runtime == "llama.cpp":
                if not model["gguf"].exists():
                    results["combos"][key] = {"status": "MISSING"}
                    continue
                print(" warmup...", flush=True)
                warmup = run_llama(model["gguf"], f"{key}_warm", CFG["n_predict_speed"])
                settle(10)
                for i in range(1, 6):
                    print(f" run {i}/5...", flush=True)
                    r = run_llama(model["gguf"], f"{key}_r{i}", CFG["n_predict_speed"])
                    r["run"] = i
                    measured.append(r)
                    all_rows.append({**r, "model_id": model["id"], "model": model["name"], "quant": model["quant"]})
                    settle(6)
            else:
                if not model["colibri"].exists():
                    results["combos"][key] = {"status": "MISSING_COLIBRI"}
                    continue
                print(" warmup...", flush=True)
                warmup = run_colibri(model["colibri"], f"{key}_warm", CFG["n_predict_speed"])
                settle(10)
                for i in range(1, 6):
                    print(f" run {i}/5...", flush=True)
                    r = run_colibri(model["colibri"], f"{key}_r{i}", CFG["n_predict_speed"])
                    r["run"] = i
                    measured.append(r)
                    all_rows.append({**r, "model_id": model["id"], "model": model["name"], "quant": model["quant"]})
                    settle(6)
        except Exception as e:
            results["combos"][key] = {"status": "FAILED", "error": str(e), "measured": measured}
            (OUT / "results_speed_partial.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            settle(15)
            continue

        model_meta = {
            "id": model["id"],
            "name": model["name"],
            "quant": model["quant"],
            "gguf": str(model["gguf"]),
            "colibri": str(model["colibri"]),
        }
        results["combos"][key] = {
            "status": "COMPLETED",
            "model": model_meta,
            "runtime": runtime,
            "warmup": warmup,
            "measured": measured,
            "summary": summarize(measured),
        }
        (OUT / "results_speed_partial.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(" summary", results["combos"][key]["summary"], flush=True)
        settle(12)

    results["ended_at"] = datetime.now(timezone.utc).isoformat()
    (OUT / "results_speed_config_A.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    csv_path = OUT / "benchmark_raw.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model_id",
                "model",
                "quant",
                "runtime",
                "run",
                "ok",
                "wall_s",
                "prompt_tps",
                "gen_tps",
                "peak_gpu_util",
                "peak_vram_mib",
                "load_s",
                "rss_gb",
                "cuda_verified",
            ],
        )
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    # summary csv
    sum_path = OUT / "benchmark_summary.csv"
    with sum_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "runtime",
                "quant",
                "status",
                "wall_median_s",
                "gen_tps_median",
                "prompt_tps_median",
                "peak_vram_median",
                "peak_gpu_util_median",
                "n_ok",
                "n_fail",
            ]
        )
        for key, combo in results["combos"].items():
            s = combo.get("summary") or {}
            m = combo.get("model") or {}
            w.writerow(
                [
                    m.get("name") if isinstance(m, dict) else key,
                    combo.get("runtime"),
                    m.get("quant") if isinstance(m, dict) else "",
                    combo.get("status"),
                    (s.get("wall_s") or {}).get("median"),
                    (s.get("gen_tps") or {}).get("median"),
                    (s.get("prompt_tps") or {}).get("median"),
                    (s.get("peak_vram_mib") or {}).get("median"),
                    (s.get("peak_gpu_util") or {}).get("median"),
                    s.get("n_ok"),
                    s.get("n_fail"),
                ]
            )
    print("DONE", csv_path)


if __name__ == "__main__":
    main()
