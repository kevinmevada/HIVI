"""P1.3 — 5-prompt speed benchmark across interfaces per model."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\mevad\Desktop\HIVI")
P13 = ROOT / "data/P1_3"
MATRIX = json.loads((P13 / "configs/interface_matrix.json").read_text(encoding="utf-8"))
PROMPTS = json.loads((P13 / "prompts/speed_5.json").read_text(encoding="utf-8"))
SETTINGS = MATRIX["settings"]
CLI = Path(MATRIX["interfaces"]["llama_cli"]["binary"])
SERVER = Path(MATRIX["interfaces"]["llama_server"]["binary"])
BIN_DIR = CLI.parent
COLI = Path(MATRIX["interfaces"]["colibri"]["binary"])


def snap_gpu() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        a, b, c = [x.strip() for x in out.split(",")]
        return {"used_mib": float(a), "free_mib": float(b), "util": float(c)}
    except Exception as e:
        return {"error": str(e)}


def extract_assistant(text: str) -> str:
    t = text or ""
    for marker in ("assistant\n", "Assistant:", "<|im_start|>assistant", "<|assistant|>"):
        if marker.lower() in t.lower():
            idx = t.lower().rfind(marker.lower())
            t = t[idx + len(marker) :]
            break
    lines = []
    for line in t.splitlines():
        if re.search(r"llama_perf_|eval time\s*=|prompt eval time|load time\s*=", line, re.I):
            break
        if line.strip().startswith("Exiting") or "tokens per second" in line.lower() and "Prompt:" in line:
            break
        lines.append(line)
    # strip llama banner if present
    body = "\n".join(lines)
    if "> " in body:
        # take after last user echo
        parts = body.split("\n> ")
        if len(parts) > 1:
            body = parts[-1]
            # drop first line if it is the prompt
            bl = body.splitlines()
            body = "\n".join(bl[1:]).strip() if len(bl) > 1 else body
    return body.strip()


def parse_tps(text: str) -> float | None:
    m = re.search(r"Generation:\s*([0-9.]+)\s*t/s", text or "", re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"eval time\s*=.*?([0-9.]+)\s*tokens per second", text or "", re.I)
    if m:
        return float(m.group(1))
    return None


def run_llama_cli(model_path: Path, prompt: str, log_path: Path) -> dict:
    args = [
        str(CLI),
        "-m",
        str(model_path),
        "-c",
        str(SETTINGS["ctx"]),
        "-n",
        str(SETTINGS["n_predict"]),
        "-ngl",
        str(SETTINGS["ngl"]),
        "--temp",
        str(SETTINGS["temperature"]),
        "--seed",
        str(SETTINGS["seed"]),
        "--jinja",
        "-st",
        "--no-display-prompt",
        "-sys",
        SETTINGS["system_prompt"],
        "-p",
        prompt,
    ]
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
            timeout=900,
        )
        wall = time.perf_counter() - t0
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        log_path.write_text(combined, encoding="utf-8")
        fail = proc.returncode != 0 and not (proc.stdout or "").strip()
        if re.search(r"failed to load|out of memory|CUDA error", combined, re.I):
            fail = True
        resp = extract_assistant(proc.stdout or "")
        return {
            "ok": not fail and bool(resp),
            "wall_s": round(wall, 3),
            "gen_tps": parse_tps(combined),
            "response": resp,
            "technical_failure": fail,
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        log_path.write_text("TIMEOUT\n", encoding="utf-8")
        return {"ok": False, "wall_s": 900.0, "technical_failure": True, "error": "timeout", "response": ""}
    except Exception as e:
        log_path.write_text(f"EXC {e}\n", encoding="utf-8")
        return {"ok": False, "wall_s": 0.0, "technical_failure": True, "error": str(e), "response": ""}


def start_llama_server(model_path: Path, port: int, log_path: Path) -> subprocess.Popen | None:
    args = [
        str(SERVER),
        "-m",
        str(model_path),
        "-c",
        str(SETTINGS["ctx"]),
        "-ngl",
        str(SETTINGS["ngl"]),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--jinja",
    ]
    env = os.environ.copy()
    env["PATH"] = str(BIN_DIR) + os.pathsep + env.get("PATH", "")
    lf = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            args,
            cwd=str(BIN_DIR),
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
        )
        # wait ready
        for _ in range(120):
            time.sleep(1)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        return proc
            except Exception:
                if proc.poll() is not None:
                    return None
        return proc if proc.poll() is None else None
    except Exception:
        return None


def stop_proc(proc: subprocess.Popen | None) -> None:
    if not proc:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_llama_server_prompt(port: int, prompt: str) -> dict:
    body = {
        "messages": [
            {"role": "system", "content": SETTINGS["system_prompt"]},
            {"role": "user", "content": prompt},
        ],
        "temperature": SETTINGS["temperature"],
        "max_tokens": SETTINGS["n_predict"],
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        wall = time.perf_counter() - t0
        text = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage") or {}
        ct = usage.get("completion_tokens") or 0
        tps = round(ct / wall, 2) if wall and ct else None
        return {
            "ok": True,
            "wall_s": round(wall, 3),
            "gen_tps": tps,
            "completion_tokens": ct,
            "response": text or "",
            "technical_failure": False,
        }
    except Exception as e:
        return {
            "ok": False,
            "wall_s": round(time.perf_counter() - t0, 3),
            "technical_failure": True,
            "error": str(e),
            "response": "",
        }


def run_ollama(tag: str, prompt: str) -> dict:
    body = {
        "model": tag,
        "prompt": prompt,
        "system": SETTINGS["system_prompt"],
        "stream": False,
        "options": {
            "temperature": SETTINGS["temperature"],
            "num_predict": SETTINGS["n_predict"],
            "seed": SETTINGS["seed"],
            "num_ctx": SETTINGS["ctx"],
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        wall = time.perf_counter() - t0
        # ollama reports eval metrics in ns sometimes
        eval_count = payload.get("eval_count") or 0
        eval_ns = payload.get("eval_duration") or 0
        tps = None
        if eval_count and eval_ns:
            tps = round(eval_count / (eval_ns / 1e9), 2)
        elif eval_count and wall:
            tps = round(eval_count / wall, 2)
        return {
            "ok": True,
            "wall_s": round(wall, 3),
            "gen_tps": tps,
            "completion_tokens": eval_count,
            "response": payload.get("response") or "",
            "technical_failure": False,
            "load_duration_s": round((payload.get("load_duration") or 0) / 1e9, 3),
        }
    except Exception as e:
        return {
            "ok": False,
            "wall_s": round(time.perf_counter() - t0, 3),
            "technical_failure": True,
            "error": str(e),
            "response": "",
        }


def run_colibri(model_dir: Path, prompt: str, log_path: Path) -> dict:
    prompt_file = log_path.with_suffix(".prompt.txt")
    prompt_file.write_text(prompt, encoding="utf-8")
    env = os.environ.copy()
    env["SNAP"] = str(model_dir)
    env["TOK"] = str(model_dir / "tokenizer.json")
    env["N_NEW"] = str(SETTINGS["n_predict"])
    env["COLI_TIMERS"] = "1"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "8")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(COLI), "256", "4", str(prompt_file)],
            cwd=str(COLI.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=900,
        )
        wall = time.perf_counter() - t0
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        log_path.write_text(combined, encoding="utf-8")
        m = re.search(r"Speed:\s*([0-9.]+)\s*tok/s", combined)
        tps = float(m.group(1)) if m else None
        # rough response: after Generated
        resp = ""
        if "Generated" in combined:
            resp = combined.split("Generated", 1)[-1][:500]
        return {
            "ok": proc.returncode == 0 or tps is not None,
            "wall_s": round(wall, 3),
            "gen_tps": tps,
            "response": resp.strip(),
            "technical_failure": proc.returncode != 0 and tps is None,
            "returncode": proc.returncode,
        }
    except Exception as e:
        log_path.write_text(f"EXC {e}\n", encoding="utf-8")
        return {"ok": False, "wall_s": 0.0, "technical_failure": True, "error": str(e), "response": ""}


def summarize_iface(runs: list[dict]) -> dict:
    ok = [r for r in runs if r.get("ok") and not r.get("technical_failure")]
    walls = [r["wall_s"] for r in ok if r.get("wall_s") is not None]
    tps = [r["gen_tps"] for r in ok if r.get("gen_tps")]
    return {
        "n_ok": len(ok),
        "n_fail": len(runs) - len(ok),
        "mean_wall_s": round(sum(walls) / len(walls), 3) if walls else None,
        "mean_gen_tps": round(sum(tps) / len(tps), 3) if tps else None,
        "min_wall_s": min(walls) if walls else None,
        "max_wall_s": max(walls) if walls else None,
    }


def bench_model(model: dict, only_ifaces: list[str] | None = None) -> dict:
    mid = model["id"]
    log_root = P13 / "logs" / mid
    resp_root = P13 / "responses" / mid
    log_root.mkdir(parents=True, exist_ok=True)
    resp_root.mkdir(parents=True, exist_ok=True)

    result = {
        "id": mid,
        "name": model["name"],
        "size_gb": model["size_gb"],
        "format": model["format"],
        "recommended": model["recommended"],
        "airllm": model.get("airllm"),
        "preflight_gpu": snap_gpu(),
        "interfaces": {},
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    try_list = only_ifaces or model["try"]
    path = Path(model["path"])

    # AirLLM always recorded as N/A for this inventory
    result["interfaces"]["airllm"] = {
        "status": "N/A",
        "reason": model.get("airllm", "unsupported"),
    }

    if "llama_cli" in try_list and model["format"] == "GGUF":
        print(f"  [{mid}] llama_cli ...", flush=True)
        if not path.exists():
            result["interfaces"]["llama_cli"] = {"status": "BLOCKED", "reason": "missing weight"}
        else:
            runs = []
            for p in PROMPTS:
                print(f"    {p['id']}", flush=True)
                out = run_llama_cli(path, p["prompt"], log_root / f"cli_{p['id']}.log")
                rec = {"prompt_id": p["id"], "prompt": p["prompt"], **out}
                (resp_root / f"cli_{p['id']}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
                runs.append(rec)
            result["interfaces"]["llama_cli"] = {
                "status": "COMPLETED",
                "summary": summarize_iface(runs),
                "runs": runs,
            }

    if "llama_server" in try_list and model["format"] == "GGUF":
        print(f"  [{mid}] llama_server ...", flush=True)
        port = int(MATRIX["interfaces"]["llama_server"]["port"])
        if not path.exists():
            result["interfaces"]["llama_server"] = {"status": "BLOCKED", "reason": "missing weight"}
        else:
            proc = start_llama_server(path, port, log_root / "server_boot.log")
            if not proc:
                result["interfaces"]["llama_server"] = {
                    "status": "BLOCKED",
                    "reason": "server failed to start",
                }
            else:
                # warm
                run_llama_server_prompt(port, "ping")
                runs = []
                try:
                    for p in PROMPTS:
                        print(f"    {p['id']}", flush=True)
                        out = run_llama_server_prompt(port, p["prompt"])
                        rec = {"prompt_id": p["id"], "prompt": p["prompt"], **out}
                        (resp_root / f"server_{p['id']}.json").write_text(
                            json.dumps(rec, indent=2), encoding="utf-8"
                        )
                        runs.append(rec)
                    result["interfaces"]["llama_server"] = {
                        "status": "COMPLETED",
                        "summary": summarize_iface(runs),
                        "runs": runs,
                        "note": "persistent load; walls exclude cold boot except first after warm",
                    }
                finally:
                    stop_proc(proc)
                    time.sleep(2)

    if "ollama" in try_list:
        tag = model.get("ollama_tag")
        print(f"  [{mid}] ollama ({tag}) ...", flush=True)
        if not tag:
            result["interfaces"]["ollama"] = {"status": "N/A", "reason": "no ollama tag mapped"}
        else:
            runs = []
            for p in PROMPTS:
                print(f"    {p['id']}", flush=True)
                out = run_ollama(tag, p["prompt"])
                rec = {"prompt_id": p["id"], "prompt": p["prompt"], **out}
                (resp_root / f"ollama_{p['id']}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
                runs.append(rec)
            result["interfaces"]["ollama"] = {
                "status": "COMPLETED" if any(r.get("ok") for r in runs) else "FAILED",
                "summary": summarize_iface(runs),
                "runs": runs,
                "tag": tag,
            }

    if "colibri" in try_list and model["format"] == "COLIBRI":
        print(f"  [{mid}] colibri ...", flush=True)
        if not path.exists() or not COLI.exists():
            result["interfaces"]["colibri"] = {"status": "BLOCKED", "reason": "missing model or engine"}
        else:
            runs = []
            for p in PROMPTS:
                print(f"    {p['id']}", flush=True)
                out = run_colibri(path, p["prompt"], log_root / f"coli_{p['id']}.log")
                rec = {"prompt_id": p["id"], "prompt": p["prompt"], **out}
                (resp_root / f"coli_{p['id']}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
                runs.append(rec)
            result["interfaces"]["colibri"] = {
                "status": "COMPLETED",
                "summary": summarize_iface(runs),
                "runs": runs,
                "note": "each prompt reloads engine (cold+decode in wall)",
            }

    result["ended_at"] = datetime.now(timezone.utc).isoformat()
    result["postflight_gpu"] = snap_gpu()
    return result


def write_report(all_results: dict) -> None:
    reports = P13 / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P1.3 — Runtime Interface Speed Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Method",
        "- 5 fixed short prompts (`prompts/speed_5.json`)",
        f"- temp={SETTINGS['temperature']} n_predict={SETTINGS['n_predict']} ctx={SETTINGS['ctx']} ngl={SETTINGS['ngl']}",
        "- Interfaces: llama-cli (direct), llama-server (HTTP), Ollama, Colibrì; AirLLM N/A (no HF weights for this inventory)",
        "- `llama_cli` ≈ bare engine / no chat UI wrapper; `llama_server` / `ollama` ≈ HTTP service interfaces",
        "",
        "## Model × interface summary",
        "",
        "| Model | Size | Recommended | Interface | Status | Mean wall (s) | Mean gen t/s |",
        "|-------|------|-------------|-----------|--------|---------------|--------------|",
    ]
    for mid, mr in all_results.get("models", {}).items():
        for iface, data in (mr.get("interfaces") or {}).items():
            if iface == "airllm":
                lines.append(
                    f"| {mr['name']} | {mr['size_gb']} | {mr['recommended']} | {iface} | {data.get('status')} | — | — |"
                )
                continue
            s = data.get("summary") or {}
            lines.append(
                f"| {mr['name']} | {mr['size_gb']} | {mr['recommended']} | {iface} | {data.get('status')} | "
                f"{s.get('mean_wall_s')} | {s.get('mean_gen_tps')} |"
            )
    lines += [
        "",
        "## Notes",
        "- Large GGUFs (>~15 GB) on 8 GiB VRAM run hybrid GPU+CPU via llama.cpp and will be slow.",
        "- Ollama only where a local tag already exists.",
        "- AirLLM requires HF safetensors; current library is GGUF/Colibrì — recorded N/A, not a runtime failure.",
        "- Full JSON: `results/interface_bench_results.json`",
        "",
    ]
    (reports / "P1_3_INTERFACE_SPEED_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    only = sys.argv[1:]  # model ids optional
    out_dir = P13 / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "interface_bench_results.json"
    if combined_path.exists():
        all_results = json.loads(combined_path.read_text(encoding="utf-8"))
    else:
        all_results = {
            "phase": "P1.3",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "settings": SETTINGS,
            "models": {},
        }

    models = MATRIX["models"]
    if only:
        models = [m for m in models if m["id"] in only or m["name"] in only]

    for m in models:
        print(f"\n=== {m['id']} {m['name']} ===", flush=True)
        mr = bench_model(m)
        all_results["models"][m["id"]] = mr
        all_results["generated_at"] = datetime.now(timezone.utc).isoformat()
        (out_dir / f"bench_{m['id']}.json").write_text(json.dumps(mr, indent=2), encoding="utf-8")
        combined_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        write_report(all_results)
        print(f"=== {m['id']} saved ===", flush=True)

    write_report(all_results)
    print("DONE", combined_path)


if __name__ == "__main__":
    main()
