#!/usr/bin/env python3
"""
HIVI P1.0 Master Audit — READ-ONLY.

Does NOT modify model packages, configs, tokenizers, checksum files, or Ollama store.
Writes reports only under data/P1_0/.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running as: uv run python scripts/p1_0/run.py
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p1_0 import __version__ as AUDIT_VERSION  # noqa: E402
from p1_0.discovery import (  # noqa: E402
    classify_format,
    find_weight_packages,
    guess_quant_from_name,
    human_bytes,
    match_expected,
)
from p1_0.expected import OLLAMA_EXPECTED, PC_EXPECTED, USB_EXPECTED  # noqa: E402
from p1_0.gguf_meta import inspect_gguf  # noqa: E402
from p1_0.hashutil import parse_sha256sums, sha256_file  # noqa: E402
from p1_0.safetensors_meta import inspect_json_config, inspect_safetensors  # noqa: E402

PC_ROOT = ROOT / "Models"
USB_ROOT = Path("D:/HIVI/Models")
LEGACY_ROOT = Path("D:/HIVI/HIVI_MODELS")
OUT = ROOT / "data" / "P1_0"
LOG_LINES: list[str] = []


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


def run_cmd(args: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
        return p.returncode, out.strip()
    except FileNotFoundError:
        return 127, "NOT_FOUND"
    except Exception as e:  # noqa: BLE001
        return 1, f"{type(e).__name__}: {e}"


def module1_environment() -> dict[str, Any]:
    log("MODULE 1 — Storage and Environment Audit")
    usb_connected = Path("D:/").exists()
    vol = {
        "connected": usb_connected,
        "label": "UNKNOWN",
        "filesystem": "UNKNOWN",
        "health": "UNKNOWN",
        "size_bytes": None,
        "free_bytes": None,
    }
    if usb_connected:
        code, out = run_cmd(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Get-Volume -DriveLetter D | "
                    "Select-Object FileSystemLabel,FileSystem,HealthStatus,Size,SizeRemaining | "
                    "ConvertTo-Json -Compress"
                ),
            ]
        )
        if code == 0 and out and out != "NOT_FOUND":
            try:
                data = json.loads(out)
                if isinstance(data, list):
                    data = data[0] if data else {}
                vol["label"] = data.get("FileSystemLabel") or "(unlabeled)"
                vol["filesystem"] = data.get("FileSystem") or "UNKNOWN"
                vol["health"] = data.get("HealthStatus") or "UNKNOWN"
                vol["size_bytes"] = data.get("Size")
                vol["free_bytes"] = data.get("SizeRemaining")
            except json.JSONDecodeError:
                vol["label"] = "PARSE_ERROR"

    pc_usage = shutil.disk_usage(str(PC_ROOT if PC_ROOT.exists() else ROOT))
    env = {
        "audit_version": AUDIT_VERSION,
        "audit_utc": datetime.now(timezone.utc).isoformat(),
        "computer_name": socket.gethostname(),
        "project_root": str(ROOT),
        "os": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "pc_library_exists": PC_ROOT.exists(),
        "pc_library": str(PC_ROOT),
        "pc_disk": human_bytes(pc_usage.total) | {"free": human_bytes(pc_usage.free)},
        "usb": vol,
        "usb_library_exists": USB_ROOT.exists(),
        "legacy_exists": LEGACY_ROOT.exists(),
        "ollama": {},
        "gpu": {},
        "packages": {},
    }

    code, out = run_cmd(["ollama", "--version"])
    env["ollama"]["available"] = code == 0 and out != "NOT_FOUND"
    env["ollama"]["version"] = out if code == 0 else "UNAVAILABLE"

    code, out = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,cuda_version",
            "--format=csv,noheader",
        ]
    )
    env["gpu"]["nvidia_smi"] = out if code == 0 else "UNAVAILABLE"

    for pkg in ("yaml", "pydantic", "huggingface_hub", "safetensors", "gguf", "torch", "transformers"):
        try:
            mod = __import__(pkg if pkg != "yaml" else "yaml")
            env["packages"][pkg] = getattr(mod, "__version__", "present")
        except ImportError:
            env["packages"][pkg] = "UNAVAILABLE"

    return env


def inventory_location(location: str, root: Path, expected: list[dict]) -> list[dict[str, Any]]:
    log(f"Discovering packages under {root}")
    records: list[dict[str, Any]] = []
    if not root.exists():
        log(f"  MISSING root: {root}")
        return records

    packages = find_weight_packages(root)
    log(f"  Found {len(packages)} weight package(s)")
    used_ids: set[str] = set()

    for pkg in packages:
        weights: list[Path] = sorted(pkg["weight_files"], key=lambda p: p.name.lower())
        others: list[Path] = sorted(pkg.get("other_files", []), key=lambda p: p.name.lower())
        total = sum(f.stat().st_size for f in weights)
        fmt = classify_format(weights)
        exp_id = match_expected(pkg["rel"], pkg["name"], expected)
        if exp_id and exp_id in used_ids:
            # allow multiple only if different paths; keep first match priority by score already
            exp_id = match_expected(pkg["rel"], pkg["name"], [e for e in expected if e["id"] not in used_ids]) or exp_id
        if exp_id:
            used_ids.add(exp_id)

        source_path = pkg["path"] / "SOURCE.txt"
        sha_path = pkg["path"] / "SHA256SUMS.txt"
        configs = [f for f in others if f.name.lower() in {"config.json", "config.hf.json"}]
        tok_files = [
            f
            for f in others
            if any(x in f.name.lower() for x in ("tokenizer", "vocab", "merges", "special_tokens"))
        ]
        chat_files = [f for f in others if "chat" in f.name.lower() or f.suffix == ".jinja"]

        rec: dict[str, Any] = {
            "model_id": exp_id or f"DISC-{location}-{pkg['name']}",
            "expected_match": exp_id or "UNMATCHED",
            "location": location,
            "folder_name": pkg["name"],
            "rel_path": pkg["rel"],
            "abs_path": str(pkg["path"]),
            "format_class": fmt,
            "weight_files": [
                {
                    "filename": f.name,
                    "extension": f.suffix.lower(),
                    "size": human_bytes(f.stat().st_size),
                    "quant_guess_from_name": guess_quant_from_name(f.name),
                }
                for f in weights
            ],
            "weight_shard_count": len(weights),
            "total_weight_size": human_bytes(total),
            "other_files": [f.name for f in others],
            "SOURCE.txt": "PRESENT" if source_path.exists() else "MISSING",
            "SHA256SUMS.txt": "PRESENT" if sha_path.exists() else "MISSING",
            "config_files": [f.name for f in configs] or "MISSING",
            "tokenizer_files": [f.name for f in tok_files] or "MISSING",
            "chat_template_files": [f.name for f in chat_files] or "MISSING",
            "source_text": None,
            "gguf": None,
            "safetensors": None,
            "json_configs": [],
            "architecture": {},
            "integrity": {},
            "completeness": {},
            "runtime_precheck": {},
            "provenance": {},
            "certification": {},
        }

        if source_path.exists():
            try:
                rec["source_text"] = source_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:  # noqa: BLE001
                rec["source_text"] = f"READ_ERROR: {e}"

        # Metadata inspection (no weight load)
        if fmt == "GGUF":
            # inspect largest / first gguf
            ggufs = [f for f in weights if f.suffix.lower() == ".gguf"]
            target = max(ggufs, key=lambda p: p.stat().st_size)
            log(f"  GGUF header: {target.name}")
            rec["gguf"] = inspect_gguf(target)
            ext = rec["gguf"].get("extracted") or {}
            rec["architecture"] = {
                "family": ext.get("architecture", "UNKNOWN"),
                "class": _arch_class(ext.get("architecture")),
                "parameter_count": ext.get("parameter_count", "UNKNOWN"),
                "quantization": ext.get("quantization", "UNKNOWN"),
                "context_length": ext.get("context_length", "UNKNOWN"),
                "embedding_length": ext.get("embedding_length", "UNKNOWN"),
                "block_count": ext.get("block_count", "UNKNOWN"),
                "attention_heads": ext.get("attention_heads", "UNKNOWN"),
                "attention_kv_heads": ext.get("attention_kv_heads", "UNKNOWN"),
                "chat_template": ext.get("chat_template", "UNKNOWN"),
                "tokenizer_type": ext.get("tokenizer_type", "UNKNOWN"),
                "tags_hint": ext.get("tags_hint", []),
                "moe": _moe_hint(ext),
            }
        elif fmt == "Safetensors":
            st_info = []
            total_tensors = 0
            for f in weights:
                if f.suffix.lower() != ".safetensors":
                    continue
                info = inspect_safetensors(f)
                st_info.append({"file": f.name, **{k: v for k, v in info.items() if k != "tensors"}, "sample_tensors": info.get("tensors", [])[:8]})
                if info.get("ok"):
                    total_tensors += int(info.get("tensor_count") or 0)
            rec["safetensors"] = {"shards": st_info, "total_tensor_entries": total_tensors}
            for cfg_name in ("config.json", "config.hf.json"):
                cfg = pkg["path"] / cfg_name
                if cfg.exists():
                    rec["json_configs"].append(inspect_json_config(cfg))
            arch = "UNKNOWN"
            quant = "I4-GS64 (claimed by package / path)" if "colibri" in pkg["rel"].lower() else "UNKNOWN"
            moe = {"is_moe": "UNKNOWN", "notes": []}
            for jc in rec["json_configs"]:
                if jc.get("ok"):
                    ex = jc.get("extracted") or {}
                    if ex.get("architectures") not in (None, "UNKNOWN"):
                        arch = ex.get("architectures")
                    if ex.get("model_type") not in (None, "UNKNOWN"):
                        arch = f"{ex.get('model_type')} / {arch}"
                    qc = ex.get("quantization_config")
                    if qc not in (None, "UNKNOWN"):
                        quant = qc
                    # MoE hints
                    raw_keys = jc.get("raw_keys") or []
                    for k in raw_keys:
                        if "expert" in k.lower():
                            moe["notes"].append(k)
                            moe["is_moe"] = "LIKELY"
            rec["architecture"] = {
                "family": arch,
                "class": "Mixture of Experts (possible)" if moe["is_moe"] == "LIKELY" else "Decoder-only / unknown dense",
                "parameter_count": "UNKNOWN",
                "quantization": quant,
                "chat_template": "PRESENT"
                if (pkg["path"] / "chat_template.jinja").exists()
                or any("chat_template" in f.name.lower() for f in others)
                else "MISSING_OR_EMBEDDED_UNKNOWN",
                "tokenizer_type": "PRESENT" if tok_files else "MISSING",
                "moe": moe,
                "safetensors_shard_count": len([f for f in weights if f.suffix.lower() == ".safetensors"]),
            }
        else:
            rec["architecture"] = {
                "family": fmt,
                "class": "Encoder-only" if "bert" in pkg["name"].lower() else "UNKNOWN",
                "parameter_count": "UNKNOWN",
                "quantization": guess_quant_from_name(weights[0].name) if weights else "UNKNOWN",
                "chat_template": "N/A",
                "tokenizer_type": "PRESENT" if tok_files else "MISSING",
            }

        records.append(rec)

    # Expected missing
    found_ids = {r["expected_match"] for r in records if r["expected_match"] != "UNMATCHED"}
    for exp in expected:
        if exp["id"] not in found_ids:
            records.append(
                {
                    "model_id": exp["id"],
                    "expected_match": exp["id"],
                    "location": location,
                    "folder_name": "MISSING",
                    "rel_path": "MISSING",
                    "abs_path": "MISSING",
                    "format_class": exp["format"],
                    "weight_files": [],
                    "weight_shard_count": 0,
                    "total_weight_size": human_bytes(0),
                    "SOURCE.txt": "MISSING",
                    "SHA256SUMS.txt": "MISSING",
                    "config_files": "MISSING",
                    "tokenizer_files": "MISSING",
                    "chat_template_files": "MISSING",
                    "architecture": {"family": "UNKNOWN", "class": "UNKNOWN"},
                    "integrity": {"status": "MISSING", "files": []},
                    "completeness": {"status": "INCOMPLETE", "reason": "Expected package not found"},
                    "runtime_precheck": {},
                    "provenance": {"status": "MISSING"},
                    "certification": {
                        "overall": "FAIL",
                        "file_integrity": "MISSING",
                        "package_completeness": "INCOMPLETE",
                        "architecture_identification": "UNKNOWN",
                        "source_provenance": "MISSING",
                        "runtime_precheck": "UNKNOWN",
                    },
                    "expected_name": exp["name"],
                }
            )
            log(f"  EXPECTED MISSING: {exp['id']} {exp['name']}")
    return records


def _arch_class(family: Any) -> str:
    if family in (None, "UNKNOWN"):
        return "UNKNOWN"
    s = str(family).lower()
    if "bert" in s:
        return "Encoder-only"
    if any(x in s for x in ("llama", "qwen", "mistral", "glm", "phi", "gemma", "ornith")):
        return "Decoder-only text LLM (likely dense unless MoE fields present)"
    return f"UNKNOWN ({family})"


def _moe_hint(ext: dict) -> dict:
    tags = ext.get("tags_hint") or []
    name = str(ext.get("model_name", "")).lower()
    if "moe" in tags or "a3b" in name or "expert" in name:
        return {"is_moe": "POSSIBLE", "note": "Name/tag suggests MoE; confirm from architecture fields"}
    return {"is_moe": "UNLIKELY", "note": "No MoE indicators in extracted GGUF tags"}


def module6_sha256(records: list[dict]) -> None:
    log("MODULE 6 — SHA256 Integrity Audit (streaming)")
    for rec in records:
        if rec.get("abs_path") in (None, "MISSING"):
            continue
        pkg = Path(rec["abs_path"])
        if not pkg.exists():
            continue
        local_sums = parse_sha256sums(pkg / "SHA256SUMS.txt") if (pkg / "SHA256SUMS.txt").exists() else {}
        file_results = []
        for wf in rec.get("weight_files") or []:
            path = pkg / wf["filename"]
            log(f"  Hashing {rec['model_id']}: {wf['filename']} ({wf['size']['GiB_binary']} GiB)")

            def _prog(done: int, size: int, pct: int, _fn=wf["filename"]) -> None:
                print(
                    f"\r    -> {_fn}: {pct}% ({done / (1024**3):.2f}/{size / (1024**3):.2f} GiB)",
                    end="",
                    flush=True,
                )

            result = sha256_file(path, progress=_prog)
            print("", flush=True)
            status = "ERROR"
            ref = None
            # match checksum entry by filename or relative path
            for key, digest in local_sums.items():
                if key.endswith(wf["filename"]) or Path(key).name == wf["filename"]:
                    ref = digest
                    break
            if not result["ok"]:
                status = "ERROR"
            elif ref is None:
                status = "LOCAL HASH GENERATED"
            elif ref == result["sha256"]:
                status = "VERIFIED"  # verified against local SHA256SUMS.txt only — not publisher unless noted
                # Clarify: local file verification, not publisher proof
                status = "VERIFIED_AGAINST_LOCAL_SHA256SUMS"
            else:
                status = "MISMATCH"
            file_results.append(
                {
                    "filename": wf["filename"],
                    "sha256": result.get("sha256"),
                    "bytes_read": result.get("bytes_read"),
                    "reference": ref,
                    "reference_source": "SHA256SUMS.txt" if ref else None,
                    "status": status,
                    "error": result.get("error"),
                    "note": (
                        "VERIFIED means match to local SHA256SUMS.txt only; "
                        "not independent publisher proof unless SOURCE documents a trusted digest."
                        if status.startswith("VERIFIED")
                        else None
                    ),
                }
            )
        # Aggregate
        statuses = [f["status"] for f in file_results]
        if not file_results:
            agg = "MISSING"
        elif any(s == "ERROR" for s in statuses):
            agg = "ERROR"
        elif any(s == "MISMATCH" for s in statuses):
            agg = "MISMATCH"
        elif all(s.startswith("VERIFIED") for s in statuses):
            agg = "VERIFIED_AGAINST_LOCAL_SHA256SUMS"
        else:
            agg = "LOCAL HASH GENERATED"
        rec["integrity"] = {"status": agg, "files": file_results}


def assess_completeness(rec: dict) -> None:
    if rec.get("abs_path") == "MISSING":
        rec["completeness"] = {"status": "INCOMPLETE", "reason": "Package missing"}
        return
    fmt = rec.get("format_class")
    reasons = []
    if rec.get("weight_shard_count", 0) < 1:
        reasons.append("No weight files")
    if fmt == "GGUF":
        gg = rec.get("gguf") or {}
        if not gg.get("ok"):
            reasons.append(f"GGUF header unreadable: {gg.get('error')}")
        # standalone GGUF OK without external tokenizer/config
        if rec.get("SOURCE.txt") == "MISSING":
            reasons.append("SOURCE.txt missing (warning)")
        if rec.get("SHA256SUMS.txt") == "MISSING":
            reasons.append("SHA256SUMS.txt missing (warning)")
        if reasons and any("unreadable" in r.lower() or r.startswith("No weight") for r in reasons):
            status = "CORRUPTED" if any("unreadable" in r.lower() for r in reasons) else "INCOMPLETE"
        elif reasons:
            status = "COMPLETE WITH WARNINGS"
        else:
            status = "COMPLETE"
    elif fmt == "Safetensors":
        if rec.get("config_files") == "MISSING":
            reasons.append("config.json missing (warning for many backends)")
        if rec.get("tokenizer_files") == "MISSING":
            reasons.append("tokenizer files missing (warning)")
        st = rec.get("safetensors") or {}
        if not st.get("shards"):
            reasons.append("No safetensors shards inspected")
        bad = [s for s in st.get("shards", []) if not s.get("ok")]
        if bad:
            reasons.append(f"{len(bad)} shard header read errors")
        # Colibri often ships many shards — check naming continuity if model-##### 
        names = sorted(wf["filename"] for wf in rec.get("weight_files", []) if wf["filename"].endswith(".safetensors"))
        if names and names[0].startswith("model-"):
            idxs = []
            for n in names:
                try:
                    idxs.append(int(n.split("-")[1].split(".")[0]))
                except Exception:
                    pass
            if idxs:
                expect = list(range(min(idxs), max(idxs) + 1))
                missing = sorted(set(expect) - set(idxs))
                if missing:
                    reasons.append(f"Missing shard indices: {missing[:20]}")
        hard = [r for r in reasons if "missing shard" in r.lower() or "read error" in r.lower() or r.startswith("No ")]
        if hard:
            status = "INCOMPLETE"
        elif reasons:
            status = "COMPLETE WITH WARNINGS"
        else:
            status = "COMPLETE"
    else:
        if rec.get("SOURCE.txt") == "MISSING":
            reasons.append("SOURCE.txt missing (warning)")
        status = "COMPLETE WITH WARNINGS" if reasons else "COMPLETE"
        if rec.get("weight_shard_count", 0) < 1:
            status = "INCOMPLETE"
    rec["completeness"] = {"status": status, "reasons": reasons}


def assess_provenance(rec: dict) -> None:
    text = rec.get("source_text") or ""
    url = "MISSING"
    for line in text.splitlines():
        if "http://" in line or "https://" in line:
            for token in line.split():
                if token.startswith("http"):
                    url = token.strip().rstrip(").,]")
                    break
        if url != "MISSING":
            break
    rec["provenance"] = {
        "status": "PRESENT" if rec.get("SOURCE.txt") == "PRESENT" else "MISSING",
        "source_url": url,
        "source_excerpt": (text[:500] + "...") if len(text) > 500 else text or None,
        "note": "Publisher trust not independently verified in P1.0 unless HF digest compared.",
    }


def runtime_precheck(env: dict, records: list[dict]) -> dict[str, Any]:
    log("MODULE 8 — Runtime Compatibility Precheck (no inference)")
    code, out = run_cmd(["ollama", "list"])
    ollama_models = []
    if code == 0 and out not in ("NOT_FOUND", ""):
        for line in out.splitlines()[1:]:
            parts = line.split()
            if parts:
                ollama_models.append({"name": parts[0], "raw": line})

    # llama.cpp vendored tree
    llama_dir = ROOT / "src" / "hivi" / "inference" / "llama.cpp-master"
    colibri_dir = ROOT / "src" / "hivi" / "inference" / "colibri-LLM-main"
    airllm_dir = ROOT / "src" / "hivi" / "inference" / "airllm-main"

    tools = {
        "ollama": env["ollama"],
        "ollama_models": ollama_models,
        "llama_cpp_source": "PRESENT" if llama_dir.exists() else "MISSING",
        "llama_cpp_built_binary": "UNKNOWN (build not verified in P1.0)",
        "colibri_source": "PRESENT" if colibri_dir.exists() else "MISSING",
        "airllm_source": "PRESENT" if airllm_dir.exists() else "MISSING",
        "lm_studio": "UNKNOWN",
        "koboldcpp": "UNKNOWN",
        "open_webui": "UNKNOWN",
        "torch": env["packages"].get("torch", "UNAVAILABLE"),
        "transformers": env["packages"].get("transformers", "UNAVAILABLE"),
        "safetensors_pkg": env["packages"].get("safetensors", "UNAVAILABLE"),
        "cuda_nvidia_smi": env["gpu"].get("nvidia_smi", "UNAVAILABLE"),
        "vram_gb_approx": 8,
        "ram_gb_claimed": 32,
    }

    # quick presence checks for GUI apps (paths common on Windows)
    for name, paths in {
        "lm_studio": [
            Path(os.environ.get("LOCALAPPDATA", "")) / "LM-Studio",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "LM Studio",
        ],
        "koboldcpp": [Path(os.environ.get("USERPROFILE", "")) / "koboldcpp"],
    }.items():
        tools[name] = "POSSIBLE_INSTALL" if any(p.exists() for p in paths) else "NOT_FOUND_IN_COMMON_PATHS"

    for rec in records:
        if rec.get("abs_path") == "MISSING":
            rec["runtime_precheck"] = {
                "llama.cpp": "UNKNOWN",
                "ollama": "UNKNOWN",
                "lm_studio": "UNKNOWN",
                "koboldcpp": "UNKNOWN",
                "transformers": "UNKNOWN",
            }
            continue
        fmt = rec.get("format_class")
        size_gib = (rec.get("total_weight_size") or {}).get("GiB_binary") or 0
        matrix = {}
        if fmt == "GGUF":
            matrix["llama.cpp"] = "SUPPORTED" if tools["llama_cpp_source"] == "PRESENT" else "POSSIBLE"
            matrix["ollama"] = "POSSIBLE"  # may need Modelfile / conversion
            matrix["lm_studio"] = "POSSIBLE"
            matrix["koboldcpp"] = "POSSIBLE"
            matrix["transformers"] = "UNSUPPORTED"
            # VRAM note — 8GB cannot fully hold 20GB+ weights; offload expected
            matrix["hardware_note"] = (
                "8GB VRAM: large GGUFs will need CPU/RAM offload; fit-in-VRAM unlikely for >~8-10GB quants"
                if size_gib > 9
                else "May fit partially in 8GB VRAM depending on context and offload"
            )
        elif fmt == "Safetensors":
            matrix["llama.cpp"] = "UNSUPPORTED"
            matrix["ollama"] = "UNSUPPORTED"
            matrix["lm_studio"] = "UNKNOWN"
            matrix["koboldcpp"] = "UNSUPPORTED"
            matrix["transformers"] = "POSSIBLE" if tools["transformers"] != "UNAVAILABLE" else "UNKNOWN"
            if "colibri" in (rec.get("rel_path") or "").lower():
                matrix["colibri"] = "SUPPORTED" if tools["colibri_source"] == "PRESENT" else "POSSIBLE"
                matrix["transformers"] = "POSSIBLE"
            matrix["hardware_note"] = "Colibri/I4 path intended; do not treat as GGUF Q4"
        elif fmt == "ONNX":
            matrix = {
                "llama.cpp": "UNSUPPORTED",
                "ollama": "UNSUPPORTED",
                "lm_studio": "UNSUPPORTED",
                "koboldcpp": "UNSUPPORTED",
                "transformers": "POSSIBLE",
                "onnxruntime": "POSSIBLE",
            }
        else:
            matrix = {
                "llama.cpp": "UNSUPPORTED",
                "ollama": "UNSUPPORTED",
                "lm_studio": "UNKNOWN",
                "koboldcpp": "UNKNOWN",
                "transformers": "POSSIBLE",
            }
        # Architecture-specific unknown if GGUF arch unknown
        arch = (rec.get("architecture") or {}).get("family")
        if fmt == "GGUF" and arch in (None, "UNKNOWN"):
            for k in list(matrix):
                if matrix[k] == "SUPPORTED":
                    matrix[k] = "POSSIBLE"
            matrix["note"] = "Architecture UNKNOWN — do not assume generic GGUF support"
        rec["runtime_precheck"] = matrix
    return tools


def certify(rec: dict) -> None:
    integrity = (rec.get("integrity") or {}).get("status", "UNKNOWN")
    complete = (rec.get("completeness") or {}).get("status", "UNKNOWN")
    arch = (rec.get("architecture") or {}).get("family", "UNKNOWN")
    prov = (rec.get("provenance") or {}).get("status", "MISSING")
    rt = rec.get("runtime_precheck") or {}

    if rec.get("abs_path") == "MISSING":
        overall = "FAIL"
    elif complete in ("INCOMPLETE", "CORRUPTED") or integrity in ("MISMATCH", "ERROR", "MISSING"):
        overall = "FAIL"
    elif complete == "COMPLETE WITH WARNINGS" or prov == "MISSING" or arch == "UNKNOWN":
        overall = "PASS WITH WARNINGS"
    elif integrity == "LOCAL HASH GENERATED":
        overall = "PASS WITH WARNINGS"
    else:
        overall = "PASS"

    # Blocked if critical tools missing for the only path
    if complete.startswith("COMPLETE") and fmt_blocked(rec, rt):
        # only block if we literally cannot assess
        pass

    rec["certification"] = {
        "overall": overall,
        "file_integrity": integrity,
        "package_completeness": complete,
        "architecture_identification": "IDENTIFIED" if arch not in (None, "UNKNOWN") else "UNKNOWN",
        "source_provenance": prov,
        "runtime_precheck": "ASSESSED" if rt else "UNKNOWN",
    }


def fmt_blocked(rec: dict, rt: dict) -> bool:
    return False


def legacy_scan() -> dict[str, Any]:
    log("Scanning legacy D:\\HIVI\\HIVI_MODELS (separate from v3 library)")
    info: dict[str, Any] = {"exists": LEGACY_ROOT.exists(), "top_level": [], "weight_files": 0, "total_bytes": 0}
    if not LEGACY_ROOT.exists():
        return info
    try:
        info["top_level"] = [p.name for p in LEGACY_ROOT.iterdir()]
        for f in LEGACY_ROOT.rglob("*"):
            if f.is_file() and f.suffix.lower() in {".gguf", ".safetensors", ".bin", ".onnx"}:
                if ".cache" in f.parts:
                    continue
                info["weight_files"] += 1
                try:
                    info["total_bytes"] += f.stat().st_size
                except OSError:
                    pass
        info["total_size"] = human_bytes(info["total_bytes"])
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def write_reports(env: dict, records: list[dict], tools: dict, legacy: dict) -> None:
    log("Writing reports to data/P1_0/")
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = env["audit_utc"]

    # JSON
    payload = {
        "meta": env,
        "tools": tools,
        "legacy": legacy,
        "models": records,
        "ollama_expected": OLLAMA_EXPECTED,
    }
    (OUT / "P1_0_MASTER_INVENTORY.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # CSV
    csv_path = OUT / "P1_0_MASTER_INVENTORY.csv"
    fields = [
        "model_id",
        "location",
        "folder_name",
        "rel_path",
        "format_class",
        "weight_shard_count",
        "size_GiB",
        "quantization",
        "architecture",
        "SOURCE.txt",
        "SHA256SUMS.txt",
        "integrity",
        "completeness",
        "provenance",
        "certification",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow(
                {
                    "model_id": r.get("model_id"),
                    "location": r.get("location"),
                    "folder_name": r.get("folder_name"),
                    "rel_path": r.get("rel_path"),
                    "format_class": r.get("format_class"),
                    "weight_shard_count": r.get("weight_shard_count"),
                    "size_GiB": (r.get("total_weight_size") or {}).get("GiB_binary"),
                    "quantization": (r.get("architecture") or {}).get("quantization"),
                    "architecture": (r.get("architecture") or {}).get("family"),
                    "SOURCE.txt": r.get("SOURCE.txt"),
                    "SHA256SUMS.txt": r.get("SHA256SUMS.txt"),
                    "integrity": (r.get("integrity") or {}).get("status"),
                    "completeness": (r.get("completeness") or {}).get("status"),
                    "provenance": (r.get("provenance") or {}).get("status"),
                    "certification": (r.get("certification") or {}).get("overall"),
                }
            )

    # SHA256 manifest
    lines = [
        f"# HIVI P1.0 SHA256 Manifest — {stamp}",
        "# Status legend: VERIFIED_AGAINST_LOCAL_SHA256SUMS | LOCAL HASH GENERATED | MISMATCH | ERROR | MISSING",
        "# Generated hashes are fingerprints only — not publisher proof by themselves.",
        "",
    ]
    for r in records:
        for fr in (r.get("integrity") or {}).get("files") or []:
            digest = fr.get("sha256") or "ERROR"
            lines.append(f"{digest}  {r.get('rel_path')}/{fr.get('filename')}  # {fr.get('status')} [{r.get('location')}]")
    (OUT / "P1_0_SHA256_MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def md_table(headers: list[str], rows: list[list[Any]]) -> str:
        out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        for row in rows:
            out.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(out)

    # Integrity report
    (OUT / "P1_0_INTEGRITY_REPORT.md").write_text(
        "\n".join(
            [
                f"# P1.0 Integrity Report",
                f"",
                f"Audit UTC: {stamp}",
                f"",
                md_table(
                    ["Model ID", "Location", "Integrity", "Completeness", "Notes"],
                    [
                        [
                            r.get("model_id"),
                            r.get("location"),
                            (r.get("integrity") or {}).get("status"),
                            (r.get("completeness") or {}).get("status"),
                            "; ".join((r.get("completeness") or {}).get("reasons") or []) or "-",
                        ]
                        for r in records
                    ],
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    (OUT / "P1_0_ARCHITECTURE_REPORT.md").write_text(
        "\n".join(
            [
                "# P1.0 Architecture Report",
                "",
                f"Audit UTC: {stamp}",
                "",
                md_table(
                    ["Model ID", "Format", "Architecture", "Class", "Quant", "Params", "Ctx", "Chat tpl"],
                    [
                        [
                            r.get("model_id"),
                            r.get("format_class"),
                            (r.get("architecture") or {}).get("family"),
                            (r.get("architecture") or {}).get("class"),
                            (r.get("architecture") or {}).get("quantization"),
                            (r.get("architecture") or {}).get("parameter_count"),
                            (r.get("architecture") or {}).get("context_length", "-"),
                            (r.get("architecture") or {}).get("chat_template", "-"),
                        ]
                        for r in records
                    ],
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    (OUT / "P1_0_RUNTIME_COMPATIBILITY.md").write_text(
        "\n".join(
            [
                "# P1.0 Runtime Compatibility (precheck only — no inference)",
                "",
                f"Audit UTC: {stamp}",
                "",
                "## Installed / detected tools",
                "```json",
                json.dumps(tools, indent=2, default=str),
                "```",
                "",
                md_table(
                    ["Model ID", "llama.cpp", "Ollama", "LM Studio", "KoboldCpp", "Transformers", "Notes"],
                    [
                        [
                            r.get("model_id"),
                            (r.get("runtime_precheck") or {}).get("llama.cpp", "-"),
                            (r.get("runtime_precheck") or {}).get("ollama", "-"),
                            (r.get("runtime_precheck") or {}).get("lm_studio", "-"),
                            (r.get("runtime_precheck") or {}).get("koboldcpp", "-"),
                            (r.get("runtime_precheck") or {}).get("transformers", "-"),
                            (r.get("runtime_precheck") or {}).get("hardware_note")
                            or (r.get("runtime_precheck") or {}).get("colibri")
                            or "-",
                        ]
                        for r in records
                    ],
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    (OUT / "P1_0_SOURCE_PROVENANCE.md").write_text(
        "\n".join(
            [
                "# P1.0 Source & Provenance",
                "",
                f"Audit UTC: {stamp}",
                "",
                md_table(
                    ["Model ID", "SOURCE.txt", "URL", "Status"],
                    [
                        [
                            r.get("model_id"),
                            r.get("SOURCE.txt"),
                            (r.get("provenance") or {}).get("source_url"),
                            (r.get("provenance") or {}).get("status"),
                        ]
                        for r in records
                    ],
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Duplicates by hash
    hash_map: dict[str, list[str]] = {}
    for r in records:
        for fr in (r.get("integrity") or {}).get("files") or []:
            h = fr.get("sha256")
            if not h:
                continue
            hash_map.setdefault(h, []).append(f"{r.get('location')}:{r.get('rel_path')}/{fr.get('filename')}")
    dup_hashes = {k: v for k, v in hash_map.items() if len(v) > 1}

    # Same base name different quants
    name_groups: dict[str, list[str]] = {}
    for r in records:
        base = (r.get("folder_name") or "").lower()
        key = base.split("-q")[0].split("_q")[0]
        name_groups.setdefault(key, []).append(r.get("model_id"))

    (OUT / "P1_0_DUPLICATE_AUDIT.md").write_text(
        "\n".join(
            [
                "# P1.0 Duplicate Audit",
                "",
                f"Audit UTC: {stamp}",
                "",
                "## Exact SHA256 duplicates",
                (
                    md_table(
                        ["SHA256", "Paths"],
                        [[h[:16] + "...", "<br>".join(paths)] for h, paths in dup_hashes.items()],
                    )
                    if dup_hashes
                    else "_None found among hashed weight files._"
                ),
                "",
                "## Legacy library",
                "```json",
                json.dumps(legacy, indent=2, default=str),
                "```",
                "",
                "No consolidation performed (read-only).",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (OUT / "P1_0_ENVIRONMENT_REPORT.md").write_text(
        "\n".join(
            [
                "# P1.0 Environment Report",
                "",
                f"Audit UTC: {stamp}",
                "",
                "```json",
                json.dumps(env, indent=2, default=str),
                "```",
                "",
                "## Hardware baseline (declared)",
                "- CPU: Intel Core Ultra 9 275HX",
                "- GPU: NVIDIA RTX 5060 Laptop (8 GB)",
                "- RAM: 32 GB",
                "- OS: Windows 11 + WSL",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Summary stats
    pc = [r for r in records if r.get("location") == "PC"]
    usb = [r for r in records if r.get("location") == "USB"]
    present = [r for r in records if r.get("abs_path") != "MISSING"]
    total_bytes = sum((r.get("total_weight_size") or {}).get("bytes") or 0 for r in present)
    weight_files = sum(r.get("weight_shard_count") or 0 for r in present)

    cert_rows = [
        [
            r.get("model_id"),
            r.get("location"),
            (r.get("certification") or {}).get("overall"),
            (r.get("certification") or {}).get("file_integrity"),
            (r.get("certification") or {}).get("package_completeness"),
            (r.get("certification") or {}).get("architecture_identification"),
            (r.get("certification") or {}).get("source_provenance"),
        ]
        for r in records
    ]

    # P1.1 order: small → medium → large; GGUF before Colibri; skip FAIL
    order = sorted(
        [r for r in present if (r.get("certification") or {}).get("overall") != "FAIL"],
        key=lambda r: ((r.get("total_weight_size") or {}).get("bytes") or 0, r.get("format_class") != "GGUF"),
    )

    final = [
        "# P1.0 Final Certification",
        "",
        f"Audit UTC: {stamp}",
        f"Audit script version: {AUDIT_VERSION}",
        f"Computer: {env.get('computer_name')}",
        f"Project root: {env.get('project_root')}",
        "",
        "## Confirmation",
        "",
        "This audit was **READ-ONLY** with respect to model packages. No model files were deleted, renamed, moved, re-quantized, or rewritten. Reports were written only under `data/P1_0/`.",
        "",
        "## Master summary",
        "",
        f"1. Total inventory records: **{len(records)}**",
        f"2. Expected PC+USB IDs: **{len(PC_EXPECTED) + len(USB_EXPECTED)}**",
        f"3. PC records: **{len(pc)}**",
        f"4. USB records: **{len(usb)}**",
        f"5. Legacy weight files (approx): **{legacy.get('weight_files', 0)}** / {legacy.get('total_size', {})}",
        f"6. Ollama models listed: **{len(tools.get('ollama_models') or [])}**",
        f"7. Total weight storage (discovered packages): **{human_bytes(total_bytes)['GiB_binary']} GiB** / **{human_bytes(total_bytes)['GB_decimal']} GB**",
        f"8. Total weight files: **{weight_files}**",
        f"9. Verified vs local SHA256SUMS: **{sum(1 for r in records for fr in (r.get('integrity') or {}).get('files') or [] if str(fr.get('status','')).startswith('VERIFIED'))}**",
        f"10. Locally generated hashes: **{sum(1 for r in records for fr in (r.get('integrity') or {}).get('files') or [] if fr.get('status')=='LOCAL HASH GENERATED')}**",
        f"11. Checksum mismatches: **{sum(1 for r in records for fr in (r.get('integrity') or {}).get('files') or [] if fr.get('status')=='MISMATCH')}**",
        f"12. COMPLETE packages: **{sum(1 for r in records if (r.get('completeness') or {}).get('status')=='COMPLETE')}**",
        f"13. COMPLETE WITH WARNINGS: **{sum(1 for r in records if (r.get('completeness') or {}).get('status')=='COMPLETE WITH WARNINGS')}**",
        f"14. INCOMPLETE packages: **{sum(1 for r in records if (r.get('completeness') or {}).get('status')=='INCOMPLETE')}**",
        f"15. Unknown architectures: **{sum(1 for r in records if (r.get('architecture') or {}).get('family') in (None,'UNKNOWN'))}**",
        f"16. Known source URLs: **{sum(1 for r in records if (r.get('provenance') or {}).get('source_url') not in (None,'MISSING'))}**",
        f"17. Missing SOURCE: **{sum(1 for r in records if r.get('SOURCE.txt')=='MISSING')}**",
        f"18. Runtime: Ollama {tools.get('ollama',{}).get('version')}; llama.cpp source {tools.get('llama_cpp_source')}; Colibri source {tools.get('colibri_source')}; torch {tools.get('torch')}",
        f"19. Exact hash duplicates: **{len(dup_hashes)}** groups",
        "20. Recommended P1.1 order (smallest present non-FAIL first):",
        "",
    ]
    for i, r in enumerate(order, 1):
        final.append(
            f"    {i}. {r.get('model_id')} — {r.get('rel_path')} ({(r.get('total_weight_size') or {}).get('GiB_binary')} GiB, {r.get('format_class')})"
        )
    final += [
        "",
        "## Certification table",
        "",
        md_table(
            ["Model ID", "Loc", "Overall", "Integrity", "Completeness", "Architecture", "Provenance"],
            cert_rows,
        ),
        "",
        "## Blockers / investigate before inference",
        "",
    ]
    for r in records:
        issues = []
        if r.get("abs_path") == "MISSING":
            issues.append("Package missing on disk")
        issues.extend((r.get("completeness") or {}).get("reasons") or [])
        if (r.get("integrity") or {}).get("status") in ("MISMATCH", "ERROR"):
            issues.append(f"Integrity: {(r.get('integrity') or {}).get('status')}")
        if not issues and (r.get("certification") or {}).get("overall") == "PASS":
            continue
        if not issues and (r.get("certification") or {}).get("overall") == "PASS WITH WARNINGS":
            issues.append("Warnings only — see completeness/provenance")
        final.append(f"- **{r.get('model_id')}**: " + "; ".join(issues))
    final += [
        "",
        "## P1.1 gate",
        "",
        "Do **not** begin P1.1 until this report is reviewed.",
        "",
    ]
    (OUT / "P1_0_FINAL_CERTIFICATION.md").write_text("\n".join(final) + "\n", encoding="utf-8")

    (OUT / "P1_0_EXECUTION_LOG.txt").write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
    log("Reports written.")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"HIVI P1.0 audit v{AUDIT_VERSION} starting (READ-ONLY)")
    log(f"Project root: {ROOT}")
    log("Will NOT modify model files, checksum files, or Ollama store.")
    try:
        env = module1_environment()
        pc_records = inventory_location("PC", PC_ROOT, PC_EXPECTED)
        usb_records = inventory_location("USB", USB_ROOT, USB_EXPECTED)
        records = pc_records + usb_records

        # Modules 3-5 done during inventory; 6 hash; 7 completeness; provenance; 8 runtime
        module6_sha256(records)
        for r in records:
            assess_completeness(r)
            assess_provenance(r)
        tools = runtime_precheck(env, records)
        for r in records:
            certify(r)
        legacy = legacy_scan()
        write_reports(env, records, tools, legacy)
        log("P1.0 COMPLETE")
        print("\nOpen: data/P1_0/P1_0_FINAL_CERTIFICATION.md\n")
        return 0
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
        (OUT / "P1_0_EXECUTION_LOG.txt").write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
