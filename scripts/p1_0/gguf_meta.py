"""Minimal GGUF metadata reader — header/KV only, never loads tensors."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

GGUF_MAGIC = b"GGUF"

# GGUF value types
_T = {
    0: ("B", 1),  # UINT8
    1: ("b", 1),  # INT8
    2: ("H", 2),  # UINT16
    3: ("h", 2),  # INT16
    4: ("I", 4),  # UINT32
    5: ("i", 4),  # INT32
    6: ("f", 4),  # FLOAT32
    7: ("?", 1),  # BOOL
    10: ("Q", 8),  # UINT64
    11: ("q", 8),  # INT64
    12: ("d", 8),  # FLOAT64
}


def _read_str(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    raw = f.read(n)
    return raw.decode("utf-8", errors="replace")


def _read_value(f, vtype: int) -> Any:
    if vtype == 8:  # STRING
        return _read_str(f)
    if vtype == 9:  # ARRAY
        (atype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        # Cap array reads for huge arrays (e.g. token lists) — keep length only for huge string arrays
        if atype == 8 and count > 64:
            # skip strings
            for _ in range(count):
                (n,) = struct.unpack("<Q", f.read(8))
                f.seek(n, 1)
            return f"<string_array len={count}>"
        items = [_read_value(f, atype) for _ in range(count)]
        if len(items) > 32:
            return items[:32] + [f"... (+{len(items) - 32} more)"]
        return items
    if vtype not in _T:
        raise ValueError(f"Unknown GGUF type {vtype}")
    fmt, size = _T[vtype]
    data = f.read(size)
    if len(data) < size:
        raise EOFError("truncated GGUF metadata value")
    return struct.unpack("<" + fmt, data)[0]


def inspect_gguf(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ok": False,
        "error": None,
        "version": None,
        "tensor_count": None,
        "metadata": {},
        "extracted": {},
    }
    try:
        with path.open("rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                out["error"] = f"Not GGUF magic: {magic!r}"
                return out
            (version,) = struct.unpack("<I", f.read(4))
            (tensor_count,) = struct.unpack("<Q", f.read(8))
            (kv_count,) = struct.unpack("<Q", f.read(8))
            out["version"] = version
            out["tensor_count"] = tensor_count
            meta: dict[str, Any] = {}
            for _ in range(kv_count):
                key = _read_str(f)
                (vtype,) = struct.unpack("<I", f.read(4))
                meta[key] = _read_value(f, vtype)
            out["metadata"] = meta
            out["ok"] = True
    except Exception as e:  # noqa: BLE001 — audit must continue
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    m = out["metadata"]

    def g(*keys, default="UNKNOWN"):
        for k in keys:
            if k in m and m[k] not in (None, ""):
                return m[k]
        return default

    quant = g("general.file_type", "general.quantization_version")
    # Map numeric file_type if present
    file_type = m.get("general.file_type")
    quant_name = _file_type_name(file_type) if isinstance(file_type, int) else str(quant)

    out["extracted"] = {
        "architecture": g("general.architecture"),
        "model_name": g("general.name", "general.basename"),
        "base_model": g("general.base_model.name", "general.base_model"),
        "parameter_count": g("general.parameter_count"),
        "quantization": quant_name,
        "file_type_raw": file_type if file_type is not None else "UNKNOWN",
        "context_length": g(
            f"{g('general.architecture', default='unknown')}.context_length",
            "llama.context_length",
            "qwen2.context_length",
            "qwen3.context_length",
            "general.context_length",
        ),
        "embedding_length": g(
            f"{g('general.architecture', default='unknown')}.embedding_length",
            "llama.embedding_length",
            "qwen2.embedding_length",
            "qwen3.embedding_length",
        ),
        "block_count": g(
            f"{g('general.architecture', default='unknown')}.block_count",
            "llama.block_count",
            "qwen2.block_count",
            "qwen3.block_count",
        ),
        "attention_heads": g(
            f"{g('general.architecture', default='unknown')}.attention.head_count",
            "llama.attention.head_count",
        ),
        "attention_kv_heads": g(
            f"{g('general.architecture', default='unknown')}.attention.head_count_kv",
            "llama.attention.head_count_kv",
        ),
        "vocab_size": _vocab_size(m),
        "tokenizer_type": g("tokenizer.ggml.model", "tokenizer.ggml.pre"),
        "chat_template": "PRESENT" if "tokenizer.chat_template" in m else "MISSING",
        "chat_template_preview": _preview(m.get("tokenizer.chat_template")),
        "bos_token_id": g("tokenizer.ggml.bos_token_id"),
        "eos_token_id": g("tokenizer.ggml.eos_token_id"),
        "tensor_count": tensor_count,
        "gguf_version": version,
        "license": g("general.license", "general.license_name"),
        "finetune": g("general.finetune"),
        "tags_hint": _tags_from_meta(m),
    }
    return out


def _preview(val: Any, n: int = 120) -> str:
    if val is None or val == "UNKNOWN":
        return "UNKNOWN"
    s = str(val).replace("\n", "\\n")
    return s[:n] + ("..." if len(s) > n else "")


def _vocab_size(m: dict) -> Any:
    tokens = m.get("tokenizer.ggml.tokens")
    if isinstance(tokens, list):
        # may be truncated representation
        return "PRESENT_IN_META"
    if isinstance(tokens, str) and "len=" in tokens:
        return tokens
    return m.get("tokenizer.ggml.scores", "UNKNOWN")


def _tags_from_meta(m: dict) -> list[str]:
    tags = []
    blob = " ".join(str(v).lower() for v in m.values() if isinstance(v, str))
    name = str(m.get("general.name", "")).lower()
    blob = blob + " " + name
    for tip in ("imatrix", "abliterated", "uncensored", "heretic", "instruct", "moe", "expert"):
        if tip in blob or tip in name:
            tags.append(tip)
    return tags


def _file_type_name(ft: int) -> str:
    # llama.cpp ggml_type / gguf file type common mapping (subset)
    mapping = {
        0: "F32",
        1: "F16",
        2: "Q4_0",
        3: "Q4_1",
        7: "Q8_0",
        8: "Q5_0",
        9: "Q5_1",
        10: "Q2_K",
        11: "Q3_K_S",
        12: "Q3_K_M",
        13: "Q3_K_L",
        14: "Q4_K_S",
        15: "Q4_K_M",
        16: "Q5_K_S",
        17: "Q5_K_M",
        18: "Q6_K",
        19: "IQ2_XXS",
        20: "IQ2_XS",
        21: "Q2_K",
        22: "IQ3_XXS",
        23: "IQ3_S",
        24: "IQ3_M",
        25: "IQ4_XS",
        26: "IQ4_NL",
        27: "IQ4_XS",
        28: "IQ4_NL",
        29: "IQ1_S",
        30: "IQ4_NL",
        31: "BF16",
        36: "TQ1_0",
        37: "TQ2_0",
    }
    return mapping.get(ft, f"FILE_TYPE_{ft}")
