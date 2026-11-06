"""Download ggml-org/gpt-oss-20b-GGUF main MXFP4 weights into Models/."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(r"C:\Users\mevad\Desktop\HIVI")
OUT = ROOT / "Models/General/ggml-org/gpt-oss-20b-gguf"
CACHE = ROOT / "Models/.hf_cache"

OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE / "hub")
os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import hf_hub_download  # noqa: E402

print("HF_HOME", os.environ["HF_HOME"], flush=True)
path = hf_hub_download(
    repo_id="ggml-org/gpt-oss-20b-GGUF",
    filename="gpt-oss-20b-MXFP4.gguf",
    local_dir=str(OUT),
)
p = Path(path)
print("DOWNLOADED", p, flush=True)
print("SIZE_GB", round(p.stat().st_size / (1024**3), 2), flush=True)

src = OUT / "SOURCE.txt"
src.write_text(
    "\n".join(
        [
            "Model repository: https://huggingface.co/ggml-org/gpt-oss-20b-GGUF",
            "Base model: openai/gpt-oss-20b",
            "File: gpt-oss-20b-MXFP4.gguf",
            "Quant: MXFP4 (~11.3 GB)",
            "Note: Eagle3 draft GGUFs not downloaded.",
            "",
        ]
    ),
    encoding="utf-8",
)
print("SOURCE", src, flush=True)
