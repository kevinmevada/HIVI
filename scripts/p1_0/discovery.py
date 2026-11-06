"""Read-only model package discovery."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .expected import META_EXTS, SKIP_DIR_NAMES, WEIGHT_EXTS


def _should_skip_dir(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES or part.startswith(".") for part in path.parts)


def iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if _should_skip_dir(p.relative_to(root).parent if p.parent != root else Path(".")):
            # skip anything under .cache etc.
            if any(part in SKIP_DIR_NAMES or (part.startswith(".") and part not in {".", ".."}) for part in p.parts):
                # allow SOURCE.txt etc at package root; skip .cache trees
                if ".cache" in p.parts or ".git" in p.parts or "__pycache__" in p.parts:
                    continue
        if ".cache" in p.parts or ".git" in p.parts or "__pycache__" in p.parts:
            continue
        files.append(p)
    return files


def find_weight_packages(root: Path) -> list[dict[str, Any]]:
    """Find directories that contain weight files (package leaves)."""
    if not root.exists():
        return []
    packages: dict[str, dict[str, Any]] = {}
    for f in iter_files(root):
        ext = f.suffix.lower()
        if ext not in WEIGHT_EXTS:
            continue
        pkg_dir = f.parent
        key = str(pkg_dir.resolve())
        if key not in packages:
            packages[key] = {
                "path": pkg_dir,
                "rel": str(pkg_dir.relative_to(root)).replace("\\", "/"),
                "name": pkg_dir.name,
                "weight_files": [],
                "other_files": [],
            }
        packages[key]["weight_files"].append(f)

    # attach sibling non-weight files in same directory only
    for pkg in packages.values():
        for f in pkg["path"].iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() in WEIGHT_EXTS:
                continue
            if f.name.startswith(".") and f.name not in {".gitattributes"}:
                continue
            if f.suffix.lower() in META_EXTS or f.name.upper() in {
                "SOURCE.TXT",
                "SHA256SUMS.TXT",
                "LICENSE",
                "README.MD",
                "README",
            }:
                pkg["other_files"].append(f)
            elif f.suffix.lower() in {".json", ".txt", ".jinja", ".model", ".py", ".md"}:
                pkg["other_files"].append(f)
    return sorted(packages.values(), key=lambda x: x["rel"].lower())


def match_expected(rel: str, name: str, expected_list: list[dict]) -> str | None:
    blob = (rel + "/" + name).lower().replace("\\", "/")
    best = None
    best_score = 0
    for exp in expected_list:
        score = 0
        for token in exp["match"]:
            if token.lower() in blob:
                score += 1
        exclude = exp.get("exclude_sub") or []
        if any(x.lower() in blob for x in exclude):
            continue
        # U03 needs onnx in path
        if exp["id"] == "U03" and "onnx" not in blob:
            continue
        if exp["id"] == "U02" and "onnx" in blob:
            continue
        if score > best_score:
            best_score = score
            best = exp["id"]
    return best if best_score > 0 else None


def classify_format(weight_files: list[Path]) -> str:
    exts = {f.suffix.lower() for f in weight_files}
    if ".gguf" in exts:
        return "GGUF"
    if ".safetensors" in exts:
        return "Safetensors"
    if ".onnx" in exts:
        return "ONNX"
    if ".bin" in exts or ".pt" in exts or ".pth" in exts:
        return "PyTorch/bin"
    return "UNKNOWN"


def human_bytes(n: int) -> dict[str, float]:
    return {
        "bytes": n,
        "GB_decimal": round(n / 1e9, 3),
        "GiB_binary": round(n / (1024**3), 3),
    }


def guess_quant_from_name(filename: str) -> str:
    patterns = [
        r"Q8_0",
        r"Q6_K_M",
        r"Q6_K_L",
        r"Q6_K",
        r"Q5_K_M",
        r"Q5_K_S",
        r"Q5_K",
        r"Q4_K_M",
        r"Q4_K_S",
        r"Q4_0",
        r"Q3_K_M",
        r"IQ4_XS",
        r"I4[_-]?GS64",
        r"int4",
    ]
    upper = filename.upper().replace("-", "_")
    for p in patterns:
        m = re.search(p, upper, re.I)
        if m:
            return m.group(0)
    return "UNKNOWN"
