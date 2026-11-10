"""Measure Colibrì decode tok/s for 27B models via serve + streaming API.

coli run does not emit Speed/tok/s. This keeps one process warm and times
generation from SSE first-token → end using usage.completion_tokens.
"""
from __future__ import annotations

import os

import csv
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

ROOT = Path(os.environ.get("HIVI_ROOT") or Path(__file__).resolve().parents[2])
OUT = ROOT / "benchmarks/27b_comparison"
LOG = OUT / "logs" / "colibri_tps"
COLI = Path(
    os.environ.get("HIVI_COLI")
    or ROOT / "src" / "hivi" / "inference" / "colibri-LLM-main" / "c" / "coli"
)
HOST = "127.0.0.1"
PORT = 18765
BASE = f"http://{HOST}:{PORT}/v1"
CTX = 4096
NGEN = 64
TEMP = 0.1
N_MEASURED = 5
# Force length-limited generation so tok/s isn't skewed by early EOS.
# Serve is started with --no-think so completion tokens are visible content.
PROMPT = (
    "Write a continuous technical explanation of how transformer attention "
    "works. Use plain prose only. Do not stop early; fill the full token budget."
)

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


def wait_health(timeout_s: float = 300.0) -> str | None:
    """Return model id when /v1/models is up, else None."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"{BASE}/models", timeout=3) as r:
                if r.status != 200:
                    time.sleep(1.0)
                    continue
                body = json.loads(r.read().decode("utf-8", errors="replace"))
                data = body.get("data") or []
                if data and isinstance(data[0], dict) and data[0].get("id"):
                    return str(data[0]["id"])
        except Exception:
            time.sleep(1.0)
    return None


def delta_text(delta: dict) -> str:
    if not isinstance(delta, dict):
        return ""
    for key in ("content", "reasoning_content", "reasoning"):
        val = delta.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def sse_events(response):
    data: list[str] = []
    for raw in response:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data.append(value[1:] if value.startswith(" ") else value)


def chat_stream(model_id: str, max_tokens: int) -> dict:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": TEMP,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    t0 = time.perf_counter()
    out = {
        "ok": False,
        "ttft_s": None,
        "wall_s": None,
        "completion_tokens": None,
        "prompt_tokens": None,
        "finish_reason": None,
        "error": None,
        "decode_tps": None,
        "e2e_tps": None,
        "mode": "stream",
    }
    text_chunks: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            ctype = (resp.headers.get_content_type() or "").lower()
            if "text/event-stream" not in ctype and "json" in ctype:
                # Server fell back to non-stream JSON
                body = json.loads(resp.read().decode("utf-8", errors="replace"))
                wall = time.perf_counter() - t0
                choice = (body.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                text = msg.get("content") or msg.get("reasoning_content") or ""
                usage = body.get("usage") or {}
                out["wall_s"] = round(wall, 3)
                out["finish_reason"] = choice.get("finish_reason")
                out["completion_tokens"] = usage.get("completion_tokens")
                out["prompt_tokens"] = usage.get("prompt_tokens")
                out["mode"] = "json_fallback"
                n = out["completion_tokens"]
                if isinstance(n, int) and n > 0 and wall > 0:
                    out["e2e_tps"] = round(n / wall, 3)
                    # Without TTFT, e2e is the best available throughput figure.
                    out["decode_tps"] = out["e2e_tps"]
                out["ok"] = out["finish_reason"] is not None and isinstance(n, int)
                out["preview"] = text[:120]
                return out
            for event in sse_events(resp):
                if event == "[DONE]":
                    break
                try:
                    chunk = json.loads(event)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    if isinstance(usage.get("completion_tokens"), int):
                        out["completion_tokens"] = usage["completion_tokens"]
                    if isinstance(usage.get("prompt_tokens"), int):
                        out["prompt_tokens"] = usage["prompt_tokens"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                piece = delta_text(choice.get("delta") or {})
                if piece:
                    if out["ttft_s"] is None:
                        out["ttft_s"] = time.perf_counter() - t0
                    text_chunks.append(piece)
                fr = choice.get("finish_reason")
                if fr:
                    out["finish_reason"] = fr
        wall = time.perf_counter() - t0
        out["wall_s"] = round(wall, 3)
        n = out["completion_tokens"]
        if n is None and text_chunks:
            # Stream ended without usage — fall back to non-stream for counts.
            out["error"] = "stream_usage_missing"
            return chat_nonstream(model_id, max_tokens)
        if out["ttft_s"] is not None and isinstance(n, int) and n > 1:
            decode_s = wall - out["ttft_s"]
            if decode_s > 0:
                out["decode_tps"] = round((n - 1) / decode_s, 3)
        if wall > 0 and isinstance(n, int) and n > 0:
            out["e2e_tps"] = round(n / wall, 3)
            if out["decode_tps"] is None:
                out["decode_tps"] = out["e2e_tps"]
        out["ok"] = out["finish_reason"] is not None and isinstance(n, int)
        out["preview"] = "".join(text_chunks)[:120]
    except Exception as e:
        out["wall_s"] = round(time.perf_counter() - t0, 3)
        out["error"] = f"{type(e).__name__}: {e}"
        # Last resort: non-stream
        try:
            return chat_nonstream(model_id, max_tokens)
        except Exception as e2:
            out["error"] = f"{out['error']} | fallback {type(e2).__name__}: {e2}"
    return out


def chat_nonstream(model_id: str, max_tokens: int) -> dict:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": TEMP,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    out = {
        "ok": False,
        "ttft_s": None,
        "wall_s": None,
        "completion_tokens": None,
        "prompt_tokens": None,
        "finish_reason": None,
        "error": None,
        "decode_tps": None,
        "e2e_tps": None,
        "mode": "nonstream",
    }
    with urllib.request.urlopen(req, timeout=900) as resp:
        body = json.loads(resp.read().decode("utf-8", errors="replace"))
    wall = time.perf_counter() - t0
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content") or msg.get("reasoning_content") or ""
    usage = body.get("usage") or {}
    n = usage.get("completion_tokens")
    out["wall_s"] = round(wall, 3)
    out["finish_reason"] = choice.get("finish_reason")
    out["completion_tokens"] = n if isinstance(n, int) else None
    out["prompt_tokens"] = usage.get("prompt_tokens")
    out["preview"] = text[:120]
    if isinstance(n, int) and n > 0 and wall > 0:
        out["e2e_tps"] = round(n / wall, 3)
        out["decode_tps"] = out["e2e_tps"]
        out["ok"] = out["finish_reason"] is not None
    return out


def start_serve(model_dir: Path, tag: str) -> tuple[subprocess.Popen, float, dict]:
    LOG.mkdir(parents=True, exist_ok=True)
    log_path = LOG / f"{tag}_serve.log"
    args = [
        "python",
        str(COLI),
        "serve",
        "--model",
        str(model_dir),
        "--host",
        HOST,
        "--port",
        str(PORT),
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
        "--no-think",
    ]
    logf = log_path.open("w", encoding="utf-8")
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        args,
        cwd=str(COLI.parent),
        stdout=logf,
        stderr=subprocess.STDOUT,
        text=True,
    )
    model_id = wait_health(timeout_s=420.0)
    load_s = time.perf_counter() - t0
    meta = {
        "ready": bool(model_id),
        "model_id": model_id,
        "load_wait_s": round(load_s, 3),
        "log": str(log_path),
        "pid": proc.pid,
    }
    if not model_id:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        logf.close()
        raise RuntimeError(f"serve not ready for {tag}: {log_path}")
    # Parse load from log if present
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"resident weights loaded in\s*([0-9.]+)s", text)
    if m:
        meta["engine_load_s"] = float(m.group(1))
    m = re.search(r"RSS after load:\s*([0-9.]+)\s*GB", text)
    if m:
        meta["rss_gb"] = float(m.group(1))
    meta["cuda"] = bool(re.search(r"\[CUDA\].*device 0|dense trunk -> CUDA", text))
    meta["log_handle"] = logf
    return proc, load_s, meta


def stop_serve(proc: subprocess.Popen, meta: dict) -> None:
    try:
        # Prefer coli stop if available; always terminate process tree.
        subprocess.run(
            ["python", str(COLI), "stop"],
            cwd=str(COLI.parent),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        pass
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    handle = meta.get("log_handle")
    if handle:
        try:
            handle.close()
        except Exception:
            pass
    time.sleep(4)


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
        "scope": "colibri_tps_27b",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "ctx": CTX,
            "ngen": NGEN,
            "temp": TEMP,
            "prompt": PROMPT,
            "method": "coli serve --no-think + streaming chat; decode_tps=(n-1)/(wall-ttft) when TTFT known else e2e",
            "port": PORT,
        },
        "models": {},
    }
    raw_rows: list[dict] = []

    for model in models:
        mid = model["id"]
        print(f"\n======== TPS {mid} {model['name']} ========", flush=True)
        if not model["colibri"].exists():
            results["models"][mid] = {"status": "MISSING", "path": str(model["colibri"])}
            continue

        pre = gpu_snap()
        proc, load_wait, meta = start_serve(model["colibri"], mid)
        model_id = meta["model_id"]
        print(
            f" serve ready in {load_wait:.1f}s id={model_id} load={meta.get('engine_load_s')} cuda={meta.get('cuda')}",
            flush=True,
        )
        try:
            print(" warmup...", flush=True)
            warm = chat_stream(model_id, NGEN)
            print(
                f"  warm decode_tps={warm.get('decode_tps')} tokens={warm.get('completion_tokens')} "
                f"wall={warm.get('wall_s')} mode={warm.get('mode')} err={warm.get('error')}",
                flush=True,
            )
            time.sleep(2)
            measured = []
            peak_mem = pre.get("mem", 0.0) if isinstance(pre.get("mem"), (int, float)) else 0.0
            for i in range(1, N_MEASURED + 1):
                print(f" run {i}/{N_MEASURED}...", flush=True)
                g0 = gpu_snap()
                r = chat_stream(model_id, NGEN)
                g1 = gpu_snap()
                if isinstance(g1.get("mem"), (int, float)):
                    peak_mem = max(peak_mem, g1["mem"])
                r["run"] = i
                r["gpu_before"] = g0
                r["gpu_after"] = g1
                measured.append(r)
                raw_rows.append(
                    {
                        "model_id": mid,
                        "model": model["name"],
                        "quant": model["quant"],
                        "run": i,
                        "ok": r["ok"],
                        "wall_s": r["wall_s"],
                        "ttft_s": r["ttft_s"],
                        "completion_tokens": r["completion_tokens"],
                        "prompt_tokens": r["prompt_tokens"],
                        "decode_tps": r["decode_tps"],
                        "e2e_tps": r["e2e_tps"],
                        "finish_reason": r["finish_reason"],
                        "error": r["error"],
                        "mode": r.get("mode"),
                    }
                )
                print(
                    f"  decode_tps={r.get('decode_tps')} e2e={r.get('e2e_tps')} "
                    f"tok={r.get('completion_tokens')} ttft={r.get('ttft_s')} wall={r.get('wall_s')} "
                    f"mode={r.get('mode')}",
                    flush=True,
                )
                time.sleep(1.5)

            ok_runs = [r for r in measured if r.get("ok") and r.get("decode_tps") is not None]
            summary = {
                "decode_tps": stats([r["decode_tps"] for r in ok_runs]),
                "e2e_tps": stats([r["e2e_tps"] for r in ok_runs]),
                "ttft_s": stats([r["ttft_s"] for r in ok_runs]),
                "wall_s": stats([r["wall_s"] for r in ok_runs]),
                "completion_tokens": stats([r["completion_tokens"] for r in ok_runs]),
                "n_ok": len(ok_runs),
                "n_fail": len(measured) - len(ok_runs),
                "peak_vram_mib": peak_mem,
            }
            results["models"][mid] = {
                "status": "COMPLETED",
                "name": model["name"],
                "quant": model["quant"],
                "path": str(model["colibri"]),
                "serve": {k: v for k, v in meta.items() if k != "log_handle"},
                "warmup": {k: v for k, v in warm.items()},
                "measured": measured,
                "summary": summary,
                "pre_gpu": pre,
                "post_gpu": gpu_snap(),
            }
            print(" summary", summary, flush=True)
        finally:
            stop_serve(proc, meta)
            print(" serve stopped", flush=True)

        (OUT / "colibri_tps_partial.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        time.sleep(8)

    results["ended_at"] = datetime.now(timezone.utc).isoformat()
    (OUT / "colibri_tps_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    with (OUT / "colibri_tps_raw.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model_id",
                "model",
                "quant",
                "run",
                "ok",
                "wall_s",
                "ttft_s",
                "completion_tokens",
                "prompt_tokens",
                "decode_tps",
                "e2e_tps",
                "finish_reason",
                "error",
                "mode",
            ],
        )
        w.writeheader()
        for row in raw_rows:
            w.writerow(row)

    # Merge into speed report
    prior_path = OUT / "colibri_only_results.json"
    prior = {}
    if prior_path.exists():
        try:
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
        except Exception:
            prior = {}

    lines = [
        "# Colibrì-only 27B Speed Report",
        "",
        f"Generated: {results['ended_at']}",
        "",
        "## Settings",
        "- Engine: Colibrì 2.1.0 (`coli serve` + streaming `/v1/chat/completions`)",
        f"- ctx={CTX}, max_tokens/ngen={NGEN}, temp={TEMP}",
        f"- Prompt: length-filling technical prose (forces ~{NGEN} completion tokens)",
        f"- Measured runs per model: {N_MEASURED} (+1 warmup), process kept warm",
        "- **decode tok/s** = `(completion_tokens - 1) / (wall - TTFT)`",
        "",
        "## Summary",
        "",
        "| Model | Quant | Load (s) | Decode t/s median | E2E t/s median | TTFT median (s) | Tokens median | Peak VRAM (MiB) |",
        "|-------|-------|----------|-------------------|----------------|-----------------|---------------|-----------------|",
    ]
    for mid, m in results["models"].items():
        if m.get("status") != "COMPLETED":
            lines.append(f"| {mid} | | {m.get('status')} | | | | | |")
            continue
        s = m["summary"]
        load = (m.get("serve") or {}).get("engine_load_s")
        # Prefer prior cold-start wall/load from earlier coli-run matrix if present
        prior_m = (prior.get("models") or {}).get(mid) or {}
        prior_load = ((prior_m.get("summary") or {}).get("load_s") or {}).get("median")
        load_show = prior_load if prior_load is not None else load
        lines.append(
            f"| {m['name']} | {m['quant']} | {load_show} | "
            f"{(s['decode_tps'] or {}).get('median')} | {(s['e2e_tps'] or {}).get('median')} | "
            f"{(s['ttft_s'] or {}).get('median')} | {(s['completion_tokens'] or {}).get('median')} | "
            f"{s.get('peak_vram_mib')} |"
        )
    lines += [
        "",
        "## Notes",
        "- Earlier `coli run` matrix measured cold-start wall/load only; it did **not** print tok/s.",
        "- This pass measures generation throughput on a warm server (correct for decode t/s).",
        "- JSON: `colibri_tps_results.json` · CSV: `colibri_tps_raw.csv`",
        "",
    ]
    report = OUT / "COLIBRI_ONLY_SPEED_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("DONE", report)


if __name__ == "__main__":
    main()
