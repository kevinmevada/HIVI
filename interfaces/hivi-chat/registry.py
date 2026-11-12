"""Scan HIVI Models/ and decide Colibrì vs llama.cpp for each GGUF.

Every file also gets a ``kind``: ``chat`` models are loaded on the main backend
and listed in the UI; ``embedding`` and ``reranker`` models run as separate
llama-server side services (see aux_services.py) and never take the chat slot.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from hivi_paths import paths

CHAT = "chat"
EMBEDDING = "embedding"
RERANKER = "reranker"
FIM = "fim"  # base code-completion models: editor autocomplete only, no chat template


def models_root() -> Path:
    """Resolved at call time so HIVI_MODELS_ROOT / storage.yaml changes apply."""
    return paths.models_root()


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
    vision_projector_path: str | None = None
    kind: str = CHAT  # "chat" | "embedding" | "reranker" | "fim"


def _kind(name: str, rel: str) -> str:
    """Embedding / reranker GGUFs are retrieval models, not chat models."""
    text = f"{name} {rel}".lower()
    if "rerank" in text:
        return RERANKER
    if "embed" in text:
        return EMBEDDING
    # Base coder models only continue code; in a chat they produce noise.
    if "fim" in text or ("qwen2.5-coder" in text and "instruct" not in text):
        return FIM
    return CHAT


def _quant(name: str) -> str:
    m = re.search(r"(?i)(UD-)?(MXFP4|Q\d+_K(?:_[A-Z]+)?|Q\d+_0|IQ\d+\w*)", name)
    return m.group(0).upper() if m else "?"


def _params(name: str, rel: str) -> str:
    if re.search(r"(?i)qwen3\.6-35b-a3b|35b-a3b", name) or re.search(
        r"(?i)qwen3\.6-35b-a3b|35b-a3b", rel
    ):
        return "35B MoE (~3B active)"
    if re.search(r"(?i)gemma-4-26b-a4b|26b-a4b", name) or re.search(
        r"(?i)gemma-4-26b-a4b|26b-a4b", rel
    ):
        return "26B MoE (~4B active)"
    if re.search(r"(?i)deepseek-coder-v2-lite", name):
        return "16B MoE (~2.4B active)"
    if re.search(r"(?i)qwen3-coder-30b-a3b|30b-a3b", name) or re.search(
        r"(?i)qwen3-coder-30b-a3b|30b-a3b", rel
    ):
        return "30B MoE (~3B active)"
    # Generic "<total>B-A<active>B" MoE naming (e.g. Nemotron Bolt 74B-A9B)
    moe = re.search(r"(?i)(\d+(?:\.\d+)?)b[-_]a(\d+(?:\.\d+)?)b", f"{name} {rel}")
    if moe:
        return f"{moe.group(1)}B MoE (~{moe.group(2)}B active)"
    oss = re.search(r"(?i)gpt-oss-(\d+)", name)
    if oss:
        return f"{oss.group(1)}B MoE"
    m = re.search(r"(?i)(\d+(?:\.\d+)?)[Bb](?:-|_|\.|$)", name)
    if m:
        return f"{m.group(1)}B"
    if "27b" in rel.lower():
        return "27B"
    return "?"


_UNCENSORED_MARKERS = ("uncensored", "abliterat", "heretic", "hauhau", "decensor")


def _pick_engine(
    name: str, rel: str, *, has_projector: bool = False, kind: str = CHAT
) -> tuple[str, str]:
    n = name.lower()
    r = rel.lower().replace("\\", "/")
    # `coli serve` is a chat server; embeddings/reranking need llama-server flags.
    if kind == EMBEDDING:
        return "llama", "embedding model → llama-server --embedding (side service)"
    if kind == RERANKER:
        return "llama", "reranker model → llama-server --reranking (side service)"
    # Vision needs llama-server --mmproj; Colibrì has no GGUF projector path.
    if has_projector:
        return "llama", "vision projector found → llama.cpp with --mmproj"
    # Uncensored builds (abliterated / heretic / HauhauCS) → llama.cpp.
    # Their refusal removal is a tiny weight edit (KL ~0.001–0.01). The Heretic Q6_K
    # still refused under Colibrì (experimental-fast + auto-tier), while the HauhauCS
    # Q5_K_P answered fully under plain llama.cpp (2026-11-11). Keep them on the
    # reference engine so the edit is reproduced exactly.
    if any(k in n or k in r for k in _UNCENSORED_MARKERS):
        return "llama", "uncensored build → llama.cpp (Colibrì tiering weakens abliteration)"
    # Explicit llama-only families (Colibrì unsupported)
    if "gpt-oss" in n or "gpt-oss" in r:
        return "llama", "architecture gpt-oss not supported by Colibrì"
    if "qwen2.5" in n or "qwen2.5" in r or "qwen2_" in n:
        return "llama", "architecture qwen2 not supported by Colibrì"
    if "deepseek-coder-v2" in n or "deepseek-coder-v2" in r:
        return "llama", "DeepSeek-Coder-V2 MoE → llama.cpp"
    if "qwen3-coder" in n or "qwen3-coder" in r or "30b-a3b" in n or "30b-a3b" in r:
        return "llama", "Qwen3-Coder MoE → llama.cpp (--cpu-moe)"
    # Colibrì-friendly (qwen3 / qwen35 / ornith)
    if any(k in n or k in r for k in ("qwen3.8", "qwen3.5", "ornith", "qwen3-")):
        return "colibri", "qwen3/qwen35 family supported by Colibrì"
    if "qwen3" in n or "qwen3" in r:
        return "colibri", "qwen3 family supported by Colibrì"
    return "llama", "default to llama.cpp"


def _display_name(stem: str, rel: str) -> str:
    low = rel.lower().replace("\\", "/")
    if "mmproj-" in stem.lower():
        return "Qwen3.6 Vision Projector (mmproj F16)"
    if "qwen3.6-35b-a3b-uncensored-heretic" in low:
        return "Qwen3.6-35B-A3B Heretic"
    if "qwen3.6-35b-a3b" in low:
        return "Qwen3.6-35B-A3B"
    if "gemma-4-26b-a4b" in low:
        return "Gemma 4 26B-A4B IT"
    if "qwen3.5-9b" in low:
        return "Qwen3.5-9B"
    if "gpt-oss" in low:
        return "GPT-OSS-20B"
    if "rvn" in stem.lower() or "heretic-abliterated-uncensored" in low:
        return "Qwen3.8-27B RVN Heretic"
    if "jonathan-coletti" in low:
        return "Qwen3.8-27B Coletti Uncensored"
    if "unsloth" in low and "27b" in low:
        return "Qwen3.8-27B Unsloth"
    if "hivemind" in low:
        return "Qwen3-8B Hivemind Heretic"
    s = stem
    s = re.sub(r"(?i)[-.](UD-)?(MXFP4|Q\d+_K(?:_[A-Z]+)?|Q\d+_0|IQ\d+\w*|imat)$", "", s)
    s = s.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


def _find_projector(gguf: Path) -> Path | None:
    """A vision projector (mmproj-*.gguf) living in the model's own folder."""
    for candidate in sorted(gguf.parent.glob("mmproj*.gguf")):
        if candidate.is_file():
            return candidate
    return None


def scan_models(root: Path | None = None) -> list[ModelEntry]:
    out: list[ModelEntry] = []
    models_dir = Path(root) if root is not None else models_root()
    if not models_dir.exists():
        return out
    for gguf in sorted(models_dir.rglob("*.gguf")):
        rel = str(gguf.relative_to(models_dir))
        stem = gguf.stem
        if stem.lower().startswith("mmproj"):
            continue
        mid = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").lower()
        kind = _kind(stem, rel)
        # Retrieval models never take a vision projector.
        projector = _find_projector(gguf) if kind == CHAT else None
        engine, reason = _pick_engine(stem, rel, has_projector=projector is not None, kind=kind)
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
                vision_projector_path=str(projector) if projector else None,
                kind=kind,
            )
        )
    return out


def chat_models(root: Path | None = None) -> list[ModelEntry]:
    """Models that can be loaded on the main chat backend."""
    return [m for m in scan_models(root) if m.kind == CHAT]


def models_as_dicts(kind: str | None = CHAT) -> list[dict]:
    """Light → heavy (file size), then name for ties. ``kind=None`` returns everything."""
    models = [m for m in scan_models() if kind is None or m.kind == kind]
    models.sort(key=lambda m: (m.size_gb, m.name.lower()))
    return [asdict(m) for m in models]


def get_model(model_id: str) -> ModelEntry | None:
    for m in scan_models():
        if m.id == model_id:
            return m
    return None