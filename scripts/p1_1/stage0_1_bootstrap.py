"""P1.1 Stage 0–1: reconciliation + environment audit (no model loads, no installs)."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\mevad\Desktop\HIVI")
P10 = ROOT / "data" / "P1_0"
P11 = ROOT / "data" / "P1_1"
INV = P10 / "P1_0_MASTER_INVENTORY.json"
CERT = P10 / "P1_0_FINAL_CERTIFICATION.md"


def run(cmd: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace"
        )
        return p.returncode, ((p.stdout or "") + ("\n" + p.stderr if p.stderr else "")).strip()
    except FileNotFoundError:
        return 127, "NOT_FOUND"
    except Exception as e:  # noqa: BLE001
        return 1, f"{type(e).__name__}: {e}"


def ensure_dirs() -> None:
    for rel in [
        "",
        "environment",
        "configs/llama_cpp",
        "configs/transformers",
        "configs/onnx",
        "tests",
        "results",
        "logs",
        "diagnostics/memory",
        "diagnostics/gpu",
        "diagnostics/runtime",
        "reports",
    ]:
        (P11 / rel).mkdir(parents=True, exist_ok=True)
    for mid in [
        "M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "U01", "U02", "U03"
    ]:
        (P11 / "logs" / mid).mkdir(parents=True, exist_ok=True)


def reconcile() -> dict:
    data = json.loads(INV.read_text(encoding="utf-8"))
    models = data["models"]
    cert_bytes = CERT.read_bytes()
    cert_sha = hashlib.sha256(cert_bytes).hexdigest()
    rows = []
    for m in models:
        path = m.get("abs_path")
        exists = path not in (None, "MISSING") and Path(path).exists()
        weights = []
        missing_weights = []
        for wf in m.get("weight_files") or []:
            fn = wf["filename"]
            fp = Path(path) / fn if exists else None
            ok = bool(fp and fp.exists())
            weights.append({"filename": fn, "exists": ok, "size": wf.get("size")})
            if not ok:
                missing_weights.append(fn)
        arch = m.get("architecture") or {}
        rows.append(
            {
                "model_id": m.get("model_id"),
                "model_name": m.get("folder_name"),
                "package_path": path,
                "package_exists": exists,
                "weight_files": weights,
                "missing_weight_files": missing_weights,
                "format": m.get("format_class"),
                "quantization": arch.get("quantization"),
                "architecture": arch.get("family"),
                "architecture_class": arch.get("class"),
                "tokenizer_present": m.get("tokenizer_files") != "MISSING",
                "config_present": m.get("config_files") != "MISSING",
                "source_present": m.get("SOURCE.txt") == "PRESENT",
                "sha256sums_present": m.get("SHA256SUMS.txt") == "PRESENT",
                "chat_template_gguf": arch.get("chat_template"),
                "p1_0_status": (m.get("certification") or {}).get("overall"),
                "p1_0_integrity": (m.get("integrity") or {}).get("status"),
                "p1_0_completeness": (m.get("completeness") or {}).get("status"),
                "p1_0_warnings": (m.get("completeness") or {}).get("reasons") or [],
                "size_GiB": (m.get("total_weight_size") or {}).get("GiB_binary"),
                "reconciliation_status": (
                    "OK"
                    if exists and not missing_weights
                    else ("BLOCKED" if not exists else "WEIGHT_FILE_MISSING")
                ),
            }
        )
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "p1_0_final_cert_path": str(CERT),
        "p1_0_final_cert_sha256": cert_sha,
        "p1_0_final_cert_bytes": len(cert_bytes),
        "note": "P1.0 reports were read only; not modified.",
        "models": rows,
    }
    (P11 / "environment" / "p1_0_baseline_reconciliation.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    return out


def environment_audit() -> dict:
    hw: dict = {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python": sys.version,
        "cpu": {},
        "ram": {},
        "gpu": {},
        "disk": {},
    }
    code, out = run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfLogicalProcessors | ConvertTo-Json -Compress",
        ]
    )
    if code == 0 and out not in ("", "NOT_FOUND"):
        try:
            hw["cpu"] = json.loads(out)
        except json.JSONDecodeError:
            hw["cpu"] = {"raw": out}
    code, out = run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "$o=Get-CimInstance Win32_OperatingSystem; "
                "@{TotalGB=[math]::Round($o.TotalVisibleMemorySize/1MB,2); "
                "FreeGB=[math]::Round($o.FreePhysicalMemory/1MB,2)} | ConvertTo-Json -Compress"
            ),
        ]
    )
    if code == 0:
        try:
            hw["ram"] = json.loads(out)
        except json.JSONDecodeError:
            hw["ram"] = {"raw": out}

    code, out = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu,utilization.memory",
            "--format=csv,noheader",
        ]
    )
    hw["gpu"]["nvidia_smi"] = out if code == 0 else "UNAVAILABLE"
    code, out = run(["nvidia-smi"])
    hw["gpu"]["nvidia_smi_full_available"] = code == 0

    usage = shutil.disk_usage(str(ROOT))
    hw["disk"]["project"] = {
        "path": str(ROOT),
        "total_GB": round(usage.total / 1e9, 2),
        "free_GB": round(usage.free / 1e9, 2),
    }
    if Path("D:/").exists():
        u = shutil.disk_usage("D:/")
        hw["disk"]["usb"] = {
            "path": "D:/",
            "total_GB": round(u.total / 1e9, 2),
            "free_GB": round(u.free / 1e9, 2),
        }
    else:
        hw["disk"]["usb"] = {"path": "D:/", "status": "NOT_CONNECTED"}

    (P11 / "environment" / "hardware_report.json").write_text(json.dumps(hw, indent=2), encoding="utf-8")

    sw = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "packages": {},
        "ollama": {},
    }
    for pkg in ("torch", "transformers", "onnxruntime", "safetensors", "gguf", "llama_cpp", "yaml", "pydantic"):
        try:
            mod = __import__(pkg if pkg != "yaml" else "yaml")
            sw["packages"][pkg] = getattr(mod, "__version__", "present")
        except ImportError:
            sw["packages"][pkg] = "UNAVAILABLE"
    code, out = run(["ollama", "--version"])
    sw["ollama"] = {"available": code == 0, "version": out if code == 0 else "UNAVAILABLE"}
    code, out = run(["ollama", "list"])
    sw["ollama"]["list"] = out if code == 0 else "UNAVAILABLE"

    # torch cuda detail if present
    if sw["packages"]["torch"] != "UNAVAILABLE":
        try:
            import torch

            sw["torch_cuda"] = {
                "cuda_available": torch.cuda.is_available(),
                "version": torch.__version__,
                "cuda_version": getattr(torch.version, "cuda", None),
            }
        except Exception as e:  # noqa: BLE001
            sw["torch_cuda"] = {"error": str(e)}
    else:
        sw["torch_cuda"] = {"cuda_available": False, "note": "torch not installed"}

    (P11 / "environment" / "software_report.json").write_text(json.dumps(sw, indent=2), encoding="utf-8")

    llama_src = ROOT / "src" / "hivi" / "inference" / "llama.cpp-master"
    colibri_src = ROOT / "src" / "hivi" / "inference" / "colibri-LLM-main"
    airllm_src = ROOT / "src" / "hivi" / "inference" / "airllm-main"
    binaries = list(llama_src.rglob("llama-cli*.exe")) if llama_src.exists() else []
    servers = list(llama_src.rglob("llama-server*.exe")) if llama_src.exists() else []
    # also check PATH
    code_cli, which_cli = run(["where", "llama-cli"])
    code_server, which_server = run(["where", "llama-server"])

    runtime = {
        "llama_cpp": {
            "source_present": llama_src.exists(),
            "source_path": str(llama_src),
            "cli_binaries_found": [str(p) for p in binaries[:20]],
            "server_binaries_found": [str(p) for p in servers[:20]],
            "path_llama_cli": which_cli if code_cli == 0 else "NOT_ON_PATH",
            "path_llama_server": which_server if code_server == 0 else "NOT_ON_PATH",
            "working_executable": bool(binaries) or code_cli == 0,
            "cuda_backend": "UNKNOWN_UNTIL_BINARY_RUN",
            "version": "UNKNOWN_UNTIL_BINARY_RUN",
            "git_commit": None,
        },
        "colibri": {"source_present": colibri_src.exists(), "source_path": str(colibri_src)},
        "airllm": {"source_present": airllm_src.exists(), "source_path": str(airllm_src)},
        "ollama": sw["ollama"],
        "transformers_ready": sw["packages"]["transformers"] != "UNAVAILABLE" and sw["packages"]["torch"] != "UNAVAILABLE",
        "onnx_ready": sw["packages"]["onnxruntime"] != "UNAVAILABLE",
    }
    # try read llama.cpp git commit if .git exists — usually zip has no .git
    commit_file = llama_src / "COMMIT"
    if (llama_src / ".git").exists():
        c, o = run(["git", "-C", str(llama_src), "rev-parse", "HEAD"])
        runtime["llama_cpp"]["git_commit"] = o if c == 0 else None

    (P11 / "environment" / "runtime_inventory.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    warnings = []
    if not runtime["llama_cpp"]["working_executable"]:
        warnings.append(
            "BLOCKER: No llama-cli/llama-server executable found. GGUF load tests cannot run until llama.cpp is built or a binary is provided."
        )
    if not runtime["transformers_ready"]:
        warnings.append(
            "BLOCKER for Safetensors/Transformers path: torch and/or transformers UNAVAILABLE (needed for some non-GGUF paths; Colibri may need its own stack)."
        )
    if not runtime["onnx_ready"]:
        warnings.append("BLOCKER for U03: onnxruntime UNAVAILABLE.")
    if hw["gpu"].get("nvidia_smi") == "UNAVAILABLE":
        warnings.append("WARNING: nvidia-smi unavailable — GPU metrics/offload verification limited.")

    (P11 / "environment" / "environment_warnings.md").write_text(
        "# P1.1 Environment Warnings\n\n"
        + "\n".join(f"- {w}" for w in warnings)
        + "\n\nGenerated: "
        + datetime.now(timezone.utc).isoformat()
        + "\n",
        encoding="utf-8",
    )

    # test registry stubs
    prompts = {
        "A": "Explain what a CPU does in three concise bullet points.",
        "B": "Return exactly three numbered items describing the benefits of local AI inference.",
        "C": "Write a Python function that returns the factorial of a non-negative integer. Include input validation and a short example.",
        "D1": "My project name is HIVI. Remember that within this conversation.",
        "D2": "What is the project name I just gave you?",
        "E": [
            "Name one advantage of NVMe storage.",
            "What is 17 + 25?",
            "Say the word ready.",
        ],
    }
    (P11 / "tests" / "test_prompts.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    criteria = {
        "conversational": ["load", "coherent_output", "structured_prompt", "coding_smoke", "multi_turn", "stability_5"],
        "bert": ["load_or_init", "one_appropriate_pass"],
        "onnx": ["graph_open", "io_inspect", "optional_inference"],
        "settings": {
            "ctx": 2048,
            "n_predict": 128,
            "temperature": 0.7,
            "top_p": 0.9,
            "batch": 128,
        },
    }
    (P11 / "tests" / "acceptance_criteria.json").write_text(json.dumps(criteria, indent=2), encoding="utf-8")

    return {"hardware": hw, "software": sw, "runtime": runtime, "warnings": warnings}


def write_readme(recon: dict, env: dict) -> None:
    ok = sum(1 for m in recon["models"] if m["reconciliation_status"] == "OK")
    blocked = [m["model_id"] for m in recon["models"] if m["reconciliation_status"] != "OK"]
    text = f"""# HIVI P1.1 — Runtime Compatibility & Load Testing

**Status:** Stages 0–1 complete. **No model load tests started** (runtimes not ready).

## Safety
- Model packages are never modified.
- All P1.1 artifacts live under `data/P1_1/` only.
- P1.0 reports are read-only (fingerprint recorded).

## Stage 0 — Baseline reconciliation
- Models reconciled: {len(recon['models'])}
- OK mappings: {ok}
- Blocked/unmapped: {blocked or 'none'}
- P1.0 final cert SHA256: `{recon['p1_0_final_cert_sha256']}`

## Stage 1 — Environment
See `environment/` reports.

## Hard blockers before Group A (M04, M07, U01)
{chr(10).join('- ' + w for w in env['warnings']) or '- None'}

## Approval required
To proceed with GGUF tests, approve **one** of:
1. Build llama.cpp from `src/hivi/inference/llama.cpp-master` with CUDA (recommended for RTX 5060), or
2. Provide an existing `llama-cli.exe` path, or
3. Approve use of an alternate GGUF runtime (document which).

Optional later approvals:
- Install `onnxruntime` for U03
- Install Colibri/Transformers/PyTorch stack for M06 (do not assume stock Transformers supports I4-GS64)

## Test order (from master prompt; do not change without documenting)
M04 → M07 → U01 → M01 → M09 → M03 → M05 → M06 → M02 → M08 → U02 → U03
"""
    (P11 / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    recon = reconcile()
    env = environment_audit()
    write_readme(recon, env)
    readiness = {
        "stage": "1_COMPLETE_AWAITING_RUNTIME_APPROVAL",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "models_reconciled": len(recon["models"]),
        "models_ok": sum(1 for m in recon["models"] if m["reconciliation_status"] == "OK"),
        "llama_cpp_ready": env["runtime"]["llama_cpp"]["working_executable"],
        "transformers_ready": env["runtime"]["transformers_ready"],
        "onnx_ready": env["runtime"]["onnx_ready"],
        "warnings": env["warnings"],
        "next_safe_step": "Obtain approval to build/install llama.cpp (or provide binary), then begin Stage 3 with M04 only.",
        "models_completed": [],
        "models_remaining": [
            "M04", "M07", "U01", "M01", "M09", "M03", "M05", "M06", "M02", "M08", "U02", "U03"
        ],
    }
    (P11 / "environment" / "readiness_report.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    print(json.dumps(readiness, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
