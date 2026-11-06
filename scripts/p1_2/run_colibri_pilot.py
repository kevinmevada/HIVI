"""P1.2 side-test: Colibri Qwen3.6-35B-A3B i4-gs64 via persistent OpenAI-compatible serve.

Not one of the six GGUF Abliterated/Heretic targets — runtime/format comparison only.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\mevad\Desktop\HIVI")
sys.path.insert(0, str(ROOT / "scripts" / "p1_2"))
from classify_response import classify_response  # noqa: E402

P12 = ROOT / "data/P1_2"
PILOT = json.loads((P12 / "prompts/pilot_30.json").read_text(encoding="utf-8"))
CTRL = json.loads((P12 / "configs/experiment_controls.json").read_text(encoding="utf-8"))
BASE = "http://127.0.0.1:8765/v1"
MID = "COLIBRI"
MODEL_NAME = "qwen3.6-colibri"


def chat(prompt: str, system: str | None, max_tokens: int = 128) -> dict:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
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
        return {
            "ok": True,
            "response": text or "",
            "wall_s": round(wall, 3),
            "usage": usage,
            "raw": payload,
        }
    except Exception as e:
        return {
            "ok": False,
            "response": "",
            "wall_s": round(time.perf_counter() - t0, 3),
            "error": str(e),
            "technical_failure": True,
        }


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(PILOT)
    prompts = PILOT[:limit]
    resp_dir = P12 / "responses" / MID
    cls_dir = P12 / "classifications" / MID
    log_dir = P12 / "logs" / MID
    for d in (resp_dir, cls_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # health
    try:
        with urllib.request.urlopen(f"{BASE}/models", timeout=10) as r:
            print("models:", r.read()[:200])
    except Exception as e:
        print("serve not ready:", e)
        sys.exit(1)

    system = CTRL["primary"]["system_prompt"]
    results = {
        "model_id": MID,
        "name": MODEL_NAME,
        "runtime": "colibri/qwen36.exe serve :8765",
        "note": "Side-test vs GGUF suite; different model family (not Abliterated/Heretic inventory)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": [],
    }
    walls = []
    for i, p in enumerate(prompts):
        pid = p["pilot_id"]
        print(f"[{i+1}/{len(prompts)}] {pid} ...", flush=True)
        out = chat(p["prompt"], system, max_tokens=128)
        if out.get("ok"):
            cls = classify_response(p["prompt"], out["response"], p["split"])
            tech = False
        else:
            cls = {
                "primary": "TECHNICAL_FAILURE",
                "confidence": 1.0,
                "explanation": out.get("error", "fail"),
                "needs_human_review": True,
                "useful_safe": False,
                "appropriate_unsafe_refusal": False,
                "unsafe_compliance_provisional": False,
            }
            tech = True
        usage = out.get("usage") or {}
        completion_toks = usage.get("completion_tokens") or 0
        tps = (
            round(completion_toks / out["wall_s"], 2)
            if out.get("wall_s") and completion_toks
            else None
        )
        rec = {
            "pilot_id": pid,
            "split": p["split"],
            "xstest_type": p["xstest_type"],
            "prompt": p["prompt"],
            "system_mode": "with_system",
            "response": out.get("response") or "",
            "wall_s": out.get("wall_s"),
            "completion_tokens": completion_toks,
            "tok_per_s": tps,
            "technical_failure": tech,
            "classification": cls,
            "error": out.get("error"),
        }
        walls.append(out.get("wall_s") or 0)
        (resp_dir / f"{pid}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        (cls_dir / f"{pid}.json").write_text(json.dumps(cls, indent=2), encoding="utf-8")
        results["runs"].append(rec)
        print(f"  wall={rec['wall_s']}s tps={tps} class={cls['primary']}", flush=True)

    valid = [r for r in results["runs"] if not r["technical_failure"]]
    tps_vals = [r["tok_per_s"] for r in valid if r.get("tok_per_s")]
    results["speed_summary"] = {
        "n": len(results["runs"]),
        "mean_wall_s": round(sum(walls) / len(walls), 2) if walls else None,
        "mean_tok_per_s": round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else None,
        "min_tok_per_s": min(tps_vals) if tps_vals else None,
        "max_tok_per_s": max(tps_vals) if tps_vals else None,
    }
    out_path = P12 / "results" / "pilot_COLIBRI.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(results["speed_summary"], indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
