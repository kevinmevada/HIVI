"""Safetensors header inspection — does not load tensor data."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any


def inspect_safetensors(path: Path, max_tensors_listed: int = 40) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ok": False,
        "error": None,
        "header_size": None,
        "tensor_count": 0,
        "tensors": [],
        "dtypes": {},
        "total_nbytes_declared": 0,
        "metadata": {},
    }
    try:
        with path.open("rb") as f:
            hdr_len_raw = f.read(8)
            if len(hdr_len_raw) < 8:
                out["error"] = "File too small for safetensors header"
                return out
            (hdr_len,) = struct.unpack("<Q", hdr_len_raw)
            if hdr_len > 100_000_000:
                out["error"] = f"Implausible header size: {hdr_len}"
                return out
            raw = f.read(hdr_len)
            header = json.loads(raw.decode("utf-8"))
        out["header_size"] = hdr_len
        meta = header.pop("__metadata__", {}) or {}
        out["metadata"] = meta if isinstance(meta, dict) else {"_raw": meta}
        dtypes: dict[str, int] = {}
        total = 0
        tensors = []
        for name, info in header.items():
            if not isinstance(info, dict):
                continue
            dtype = info.get("dtype", "?")
            shape = info.get("shape", [])
            data_offsets = info.get("data_offsets", [0, 0])
            nbytes = 0
            if isinstance(data_offsets, list) and len(data_offsets) == 2:
                nbytes = int(data_offsets[1]) - int(data_offsets[0])
            total += max(nbytes, 0)
            dtypes[dtype] = dtypes.get(dtype, 0) + 1
            if len(tensors) < max_tensors_listed:
                tensors.append({"name": name, "dtype": dtype, "shape": shape, "nbytes": nbytes})
        out["tensor_count"] = len(header)
        out["tensors"] = tensors
        out["dtypes"] = dtypes
        out["total_nbytes_declared"] = total
        out["ok"] = True
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def inspect_json_config(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        keys = [
            "model_type",
            "architectures",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "torch_dtype",
            "quantization_config",
            "tie_word_embeddings",
        ]
        extracted = {k: data.get(k, "UNKNOWN") for k in keys if k in data or True}
        # keep only present or interesting
        extracted = {k: data.get(k, "UNKNOWN") for k in keys}
        return {"ok": True, "path": str(path), "extracted": extracted, "raw_keys": sorted(data.keys())[:80]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "path": str(path), "error": f"{type(e).__name__}: {e}"}
