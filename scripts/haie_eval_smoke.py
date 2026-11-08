"""HAIE V1.1 acceptance: path expectations + optional live classify on loaded model."""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(r"C:\Users\mevad\Desktop\HIVI")
sys.path.insert(0, str(ROOT / "interfaces" / "hivi-chat"))

from haie import plan_request  # noqa: E402
from haie.eval_store import log_outcome  # noqa: E402
from haie.harmony import extract_final_text  # noqa: E402

# expected_path: fast | semantic (fallback counts as fail for semantic cases)
ACCEPTANCE = [
    ("greeting", "Hey", "fast"),
    ("simple_coding", "Write Hello World in python", "fast"),
    ("simple_math", "What is 2+2?", "fast"),
    ("explain_optimize", "explain and optimize this python function for sorting", "semantic"),
    ("architecture", "Design a trading engine", "semantic"),
    ("compare_moe", "Compare MoE and Dense models for local 8GB inference", "semantic"),
    ("debug_complex", "Debug this complex concurrency problem in a thread pool", "semantic"),
    ("ambiguous_multi", "explain and also implement a cache and compare two approaches", "semantic"),
]


def status() -> dict:
    try:
        with urllib.request.urlopen("http://127.0.0.1:7860/api/status", timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"error": str(e)}


def main() -> int:
    st = status()
    engine = st.get("engine") or "llama"
    model_id = st.get("model_id")
    backend = st.get("backend") or "http://127.0.0.1:8080"
    print("status", json.dumps({k: st.get(k) for k in ("model_id", "engine", "status", "healthy")}, indent=2))

    rows = []
    passed = 0
    for name, text, expect in ACCEPTANCE:
        body = {"model": "local", "messages": [{"role": "user", "content": text}], "stream": False}
        adapted, meta = plan_request(
            body,
            engine=engine,
            backend_base=backend,
            model_id=model_id,
        )
        ok = meta.path == expect
        # semantic expected: also accept only exact semantic (not fallback)
        if ok:
            passed += 1
        mark = "PASS" if ok else "FAIL"
        rows.append(
            {
                "case": name,
                "prompt": text,
                "expected": expect,
                "actual": meta.path,
                "ok": ok,
                "intents": meta.intents,
                "confidence": meta.confidence,
                "classify_ms": meta.classify_ms,
                "temp": adapted.get("temperature"),
                "max_tokens": adapted.get("max_tokens"),
            }
        )
        print(
            f"[{mark}] {name:16} expect={expect:8} got={meta.path:8} "
            f"intents={meta.intents} conf={meta.confidence:.2f} "
            f"classify_ms={None if meta.classify_ms is None else round(meta.classify_ms)} "
            f"prompt_ms={None if meta.classify_prompt_ms is None else round(meta.classify_prompt_ms)} "
            f"gen_ms={None if meta.classify_predicted_ms is None else round(meta.classify_predicted_ms)} "
            f"temp={adapted.get('temperature')} max={adapted.get('max_tokens')}"
        )

    print(f"\nacceptance {passed}/{len(ACCEPTANCE)}")

    if st.get("healthy") or st.get("status") == "ready":
        print("\n-- live completion (simple coding) --")
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "Write Hello World in python as one line"}],
            "stream": False,
            "max_tokens": 128,
        }
        req = urllib.request.Request(
            "http://127.0.0.1:7860/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                headers = {
                    k: r.headers.get(k)
                    for k in (
                        "X-HIVI-HAIE-Path",
                        "X-HIVI-HAIE-Intents",
                        "X-HIVI-HAIE-Temp",
                        "X-HIVI-HAIE-Max-Tokens",
                        "X-HIVI-HAIE-Classify-Ms",
                    )
                }
                obj = json.loads(r.read().decode())
            choice = (obj.get("choices") or [{}])[0]
            raw = (choice.get("message") or {}).get("content") or ""
            final = extract_final_text(raw)
            leaked = "<|channel|>" in raw and not final
            print("headers", headers)
            print("raw_preview", repr(raw[:160]))
            print("final_preview", repr(final[:160]))
            print("harmony_leak_without_final", leaked, "final_empty", not bool(final))
            log_outcome(
                model_id=model_id,
                engine=engine,
                finish_reason=choice.get("finish_reason"),
                completion_tokens=(obj.get("usage") or {}).get("completion_tokens"),
                empty=not bool(final.strip()),
                truncated=choice.get("finish_reason") == "length",
                path=headers.get("X-HIVI-HAIE-Path"),
            )
        except urllib.error.HTTPError as e:
            print("live HTTP", e.code, e.read()[:300])
        except Exception as e:
            print("live error", e)
    else:
        print("\nskip live — engine not ready")

    out = ROOT / "data" / "haie" / "smoke_last.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"status": st, "passed": passed, "total": len(ACCEPTANCE), "cases": rows}, indent=2),
        encoding="utf-8",
    )
    print("wrote", out)
    return 0 if passed == len(ACCEPTANCE) else 1


if __name__ == "__main__":
    raise SystemExit(main())
