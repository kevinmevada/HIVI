"""Download unsloth Qwen3-Coder-30B-A3B Instruct Q4_K_M into Models/Coding."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("HIVI_ROOT") or Path(__file__).resolve().parents[1])
OUT = ROOT / "Models/Coding/unsloth/qwen3-coder-30b-a3b-instruct-q4_k_m"
CACHE = ROOT / "Models/.hf_cache"

OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE / "hub")
os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
FILE = "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"

print("HF_HOME", os.environ["HF_HOME"], flush=True)
print("Downloading", REPO, FILE, flush=True)
path = hf_hub_download(
    repo_id=REPO,
    filename=FILE,
    local_dir=str(OUT),
)
p = Path(path)
print("DOWNLOADED", p, flush=True)
print("SIZE_GB", round(p.stat().st_size / (1024**3), 2), flush=True)

src = OUT / "SOURCE.txt"
src.write_text(
    "\n".join(
        [
            f"Model repository: https://huggingface.co/{REPO}",
            "Base model: Qwen/Qwen3-Coder-30B-A3B-Instruct",
            f"File: {FILE}",
            "Quant: Q4_K_M (~18.6 GB)",
            "Arch: MoE 30B total / ~3B active",
            "Runtime: llama.cpp with --fit on and --cpu-moe on 8GB VRAM",
            "",
        ]
    ),
    encoding="utf-8",
)
print("SOURCE", src, flush=True)
