"""Scan HIVI Models/ and decide Colibrì vs llama.cpp for each GGUF."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(r"C:\Users\mevad\Desktop\HIVI")
MODELS = ROOT / "Models"


@dataclass
class ModelEntry:
    id: str
    name: str
    path: str
    quant: str
    params: str
    size_gb: float
    engine: str  # "colibri" | "llama"
    engine_reason: str
    colibri_path: str | None = None


def _quant(name: str) -> str:
    m = re.search(r"(?i)(UD-)?(MXFP4|Q\d+_K(?:_[A-Z]+)?|Q\d+_0|IQ\d+\w*)", name)
    return m.group(0).upper() if m else "?"


def _params(name: str, rel: str) -> str:
    if re.search(r"(?i)deepseek-coder-v2-lite", name):
        return "16B MoE (~2.4B active)"
    if re.search(r"(?i)gpt-oss-(\d+)", name):
        return f"{re.search(r'(?i)gpt-oss-(\d+)', name).group(1)}B"
    m = re.search(r"(?i)(\d+(?:\.\d+)?)[Bb](?:-|_|\.|$)", name)
    if m:
        return f"{m.group(1)}B"
    if "27b" in rel.lower():
        return "27B"
    return "?"


def _pick_engine(name: str, rel: str) -> tuple[str, str]:
    n = name.lower()
    r = rel.lower().replace("\\", "/")
    # Explicit llama-only families (Colibrì unsupported)
    if "gpt-oss" in n or "gpt-oss" in r:
        return "llama", "architecture gpt-oss not supported by Colibrì"
    if "qwen2.5" in n or "qwen2.5" in r or "qwen2_" in n:
        return "llama", "architecture qwen2 not supported by Colibrì"
    if "deepseek-coder-v2" in n or "deepseek-coder-v2" in r:
        return "llama", "DeepSeek-Coder-V2 MoE → llama.cpp"
    # Colibrì-friendly (qwen3 / qwen35 / ornith)
    if any(k in n or k in r for k in ("qwen3.8", "qwen3.5", "ornith", "qwen3-")):
        return "colibri", "qwen3/qwen35 family supported by Colibrì"
    if "qwen3" in n or "qwen3" in r:
        return "colibri", "qwen3 family supported by Colibrì"
    return "llama", "default to llama.cpp"


def _display_name(stem: str, rel: str) -> str:
    low = rel.lower().replace("\\", "/")
    if "rvn" in stem.lower() or "heretic-abliterated-uncensored" in low:
        return "Qwen3.8-27B RVN Heretic"
    if "jonathan-coletti" in low:
        return "Qwen3.8-27B Coletti Uncensored"
    if "unsloth" in low and "27b" in low:
        return "Qwen3.8-27B Unsloth"
    if "hivemind" in low:
        return "Qwen3-8B Hivemind Heretic"
    if "gpt-oss" in low:
        return "gpt-oss-20b"
    s = stem
    s = re.sub(r"(?i)-(UD-)?(MXFP4|Q\d+_K(?:_[A-Z]+)?|Q\d+_0|IQ\d+\w*|imat)$", "", s)
    s = s.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


def scan_models() -> list[ModelEntry]:
    out: list[ModelEntry] = []
    if not MODELS.exists():
        return out
    for gguf in sorted(MODELS.rglob("*.gguf")):
        rel = str(gguf.relative_to(MODELS))
        stem = gguf.stem
        mid = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").lower()
        engine, reason = _pick_engine(stem, rel)
        colibri = gguf.with_suffix(".colibri")
        # also accept NAME.colibri next to file with same stem
        if not colibri.exists():
            alt = gguf.parent / f"{stem}.colibri"
            colibri = alt if alt.exists() else colibri
        out.append(
            ModelEntry(
                id=mid,
                name=_display_name(stem, rel),
                path=str(gguf),
                quant=_quant(stem),
                params=_params(stem, rel),
                size_gb=round(gguf.stat().st_size / (1024**3), 2),
                engine=engine,
                engine_reason=reason,
                colibri_path=str(colibri) if colibri.exists() else None,
            )
        )
    return out


def models_as_dicts() -> list[dict]:
    # Light → heavy (file size), then name for ties
    models = sorted(scan_models(), key=lambda m: (m.size_gb, m.name.lower()))
    return [asdict(m) for m in models]


def get_model(model_id: str) -> ModelEntry | None:
    for m in scan_models():
        if m.id == model_id:
            return m
    return None
