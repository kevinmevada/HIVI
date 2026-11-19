"""How much context a GGUF model was trained for (``<arch>.context_length``).

HIVI runs each model with a big window, but never past what the model was
trained on: beyond that, answers fall apart. Only the start of the metadata is
read (the key sits near the top, before the vocabulary), so this takes
milliseconds even for 30 GB files, and results are cached per path/size/mtime.
"""
from __future__ import annotations

import struct
import threading
from pathlib import Path

_MAGIC = b"GGUF"
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64 = range(13)
_SIZE = {_U8: 1, _I8: 1, _U16: 2, _I16: 2, _U32: 4, _I32: 4, _F32: 4, _BOOL: 1,
         _U64: 8, _I64: 8, _F64: 8}
_INT = {_U8: "<B", _I8: "<b", _U16: "<H", _I16: "<h", _U32: "<I", _I32: "<i",
        _U64: "<Q", _I64: "<q"}
_LOCK = threading.Lock()
_CACHE: dict[str, tuple[str, int | None]] = {}


def _read_u32(fh) -> int:
    return struct.unpack("<I", fh.read(4))[0]


def _read_u64(fh) -> int:
    return struct.unpack("<Q", fh.read(8))[0]


def _read_str(fh) -> str:
    n = _read_u64(fh)
    if n > 1 << 20:
        raise ValueError("implausible string length")
    return fh.read(n).decode("utf-8", errors="replace")


def _skip(fh, vtype: int) -> None:
    if vtype in _SIZE:
        fh.seek(_SIZE[vtype], 1)
    elif vtype == _STR:
        fh.seek(_read_u64(fh), 1)
    elif vtype == _ARR:
        etype = _read_u32(fh)
        n = _read_u64(fh)
        if etype in _SIZE:
            fh.seek(n * _SIZE[etype], 1)
        else:
            for _ in range(n):
                _skip(fh, etype)
    else:
        raise ValueError(f"bad gguf value type {vtype}")


def read_context_length(path: str | Path) -> int | None:
    """Trained context length from the file's metadata, or None if unreadable."""
    with open(path, "rb") as fh:
        if fh.read(4) != _MAGIC:
            return None
        if _read_u32(fh) < 2:
            return None
        _read_u64(fh)              # tensor count
        n_kv = _read_u64(fh)
        arch = ""
        for _ in range(min(n_kv, 100000)):
            key = _read_str(fh)
            vtype = _read_u32(fh)
            if key == "general.architecture" and vtype == _STR:
                arch = _read_str(fh)
                continue
            if arch and key == f"{arch}.context_length" and vtype in _INT:
                fmt = _INT[vtype]
                return int(struct.unpack(fmt, fh.read(struct.calcsize(fmt)))[0])
            _skip(fh, vtype)
    return None


def trained_context(path: str | Path | None) -> int | None:
    """Cached ``read_context_length``; None for missing/non-GGUF files."""
    if not path:
        return None
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return None
    stamp = f"{st.st_size}:{int(st.st_mtime)}"
    with _LOCK:
        hit = _CACHE.get(str(p))
        if hit and hit[0] == stamp:
            return hit[1]
    try:
        value = read_context_length(p)
    except (OSError, ValueError, struct.error):
        value = None
    with _LOCK:
        _CACHE[str(p)] = (stamp, value)
    return value
