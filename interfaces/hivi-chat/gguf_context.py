"""How much context a GGUF model was trained for (``<arch>.context_length``).

HIVI runs each model with a big window, but never past what the model was
trained on: beyond that, answers fall apart. Only the start of the metadata is
read (the key sits near the top, before the vocabulary), so this takes
milliseconds even for 30 GB files, and results are cached per path/size/mtime.
"""
from __future__ import annotations

import json
import re
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


# --------------------------------------------------------------------------- #
# Model facts for the picker: real parameter counts (total and, for MoE, active
# per token), trained context, and what kind of thinking the chat template has.
# Reading every tensor header takes a fraction of a second per file, so results
# are kept in data/run/gguf_info.json and only computed once per file version.
# --------------------------------------------------------------------------- #


_INFO_LOCK = threading.Lock()
_INFO: dict[str, dict] = {}
_INFO_LOADED = False
_INFO_VERSION = 1
_SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def _info_file() -> Path:
    from hivi_paths import paths
    d = paths.data_root() / "run"
    d.mkdir(parents=True, exist_ok=True)
    return d / "gguf_info.json"


def _load_info_cache() -> None:
    global _INFO_LOADED
    if _INFO_LOADED:
        return
    _INFO_LOADED = True
    try:
        data = json.loads(_info_file().read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("version") == _INFO_VERSION:
            _INFO.update(data.get("files") or {})
    except (OSError, ValueError):
        pass


def _save_info_cache() -> None:
    try:
        f = _info_file()
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": _INFO_VERSION, "files": _INFO}), encoding="utf-8")
        tmp.replace(f)
    except OSError:
        pass


def _read_value(fh, vtype: int):
    if vtype in _INT:
        fmt = _INT[vtype]
        return struct.unpack(fmt, fh.read(struct.calcsize(fmt)))[0]
    if vtype == _STR:
        n = _read_u64(fh)
        return fh.read(n).decode("utf-8", errors="replace")
    _skip(fh, vtype)
    return None


_WANTED = ("general.architecture", "general.size_label", "general.name", "tokenizer.chat_template")


def read_file_facts(path: str | Path) -> dict:
    """Metadata and tensor sizes of one .gguf file (one part of a split model)."""
    meta: dict = {}
    total = experts = 0
    with open(path, "rb") as fh:
        if fh.read(4) != _MAGIC:
            raise ValueError("not a GGUF file")
        if _read_u32(fh) < 2:
            raise ValueError("GGUF v1 is not supported")
        n_tensors = _read_u64(fh)
        n_kv = _read_u64(fh)
        arch = ""
        for _ in range(n_kv):
            key = _read_str(fh)
            vtype = _read_u32(fh)
            wanted = key in _WANTED or (arch and key in (
                f"{arch}.context_length", f"{arch}.expert_count", f"{arch}.expert_used_count"))
            if wanted:
                value = _read_value(fh, vtype)
                if key == "general.architecture" and isinstance(value, str):
                    arch = value
                meta[key.replace(f"{arch}.", "arch.") if arch else key] = value
            else:
                _skip(fh, vtype)
        for _ in range(n_tensors):
            name = _read_str(fh)
            dims = _read_u32(fh)
            count = 1
            for _ in range(dims):
                count *= _read_u64(fh)
            fh.seek(4 + 8, 1)          # type, offset
            total += count
            if "_exps" in name or ".experts." in name:
                experts += count
    meta["total_params"] = total
    meta["expert_params"] = experts
    return meta


def _parts(path: Path) -> list[Path]:
    m = _SPLIT.search(path.name)
    if not m:
        return [path]
    n = int(m.group(2))
    stem = path.name[:m.start()]
    return [path.with_name(f"{stem}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]


def thinking_kind(template: str | None) -> str:
    """What the model's chat template allows:
    "toggle" (thinking on/off, Qwen3-style enable_thinking), "effort" (gpt-oss
    reasoning effort, can't be fully off), "always" (always reasons), "none"."""
    t = template or ""
    if "enable_thinking" in t:
        return "toggle"
    if "reasoning_effort" in t or "<|channel|>" in t:
        return "effort"
    if "<think>" in t:
        return "always"
    return "none"


def model_facts(path: str | Path) -> dict | None:
    """Total/active parameters, experts, trained context and thinking kind for a
    model file (all parts of a split model). Cached; None if unreadable."""
    p = Path(path)
    try:
        parts = _parts(p)
        stamp = ";".join(f"{q.stat().st_size}:{int(q.stat().st_mtime)}" for q in parts)
    except OSError:
        return None
    with _INFO_LOCK:
        _load_info_cache()
        hit = _INFO.get(str(p))
        if hit and hit.get("stamp") == stamp:
            return hit.get("facts")
    try:
        metas = [read_file_facts(q) for q in parts]
    except (OSError, ValueError, struct.error):
        facts = None
    else:
        head = metas[0]
        total = sum(m["total_params"] for m in metas)
        exp = sum(m["expert_params"] for m in metas)
        n_exp = int(head.get("arch.expert_count") or 0)
        used = int(head.get("arch.expert_used_count") or 0)
        active = total
        if exp and n_exp > 0 and 0 < used <= n_exp:
            active = total - exp + exp * used // n_exp
        facts = {
            "arch": head.get("general.architecture") or "",
            "total_params": total,
            "active_params": active,
            "experts": n_exp,
            "experts_used": used,
            "trained_ctx": int(head.get("arch.context_length") or 0) or None,
            "size_label": head.get("general.size_label") or "",
            "thinking": thinking_kind(head.get("tokenizer.chat_template")),
        }
    with _INFO_LOCK:
        _INFO[str(p)] = {"stamp": stamp, "facts": facts}
        _save_info_cache()
    return facts


def cached_facts(path: str | Path) -> tuple[bool, dict | None]:
    """(known, facts) without reading the file: known is False until model_facts ran."""
    p = Path(path)
    try:
        stamp = ";".join(f"{q.stat().st_size}:{int(q.stat().st_mtime)}" for q in _parts(p))
    except OSError:
        return True, None
    with _INFO_LOCK:
        _load_info_cache()
        hit = _INFO.get(str(p))
    if hit and hit.get("stamp") == stamp:
        return True, hit.get("facts")
    return False, None


_WARMING: set[str] = set()


def warm(paths_: list[str], on_done=None) -> bool:
    """Compute missing facts on a side thread. True if any were missing."""
    todo = [p for p in paths_ if not cached_facts(p)[0] and p not in _WARMING]
    if not todo:
        return any(p in _WARMING for p in paths_)
    _WARMING.update(todo)

    def run() -> None:
        try:
            for p in todo:
                model_facts(p)
                _WARMING.discard(p)
        finally:
            _WARMING.difference_update(todo)
            if on_done:
                on_done()

    threading.Thread(target=run, name="hivi-gguf-facts", daemon=True).start()
    return True


def human_params(n: int) -> str:
    """12_345_678_901 -> '12B', 3_600_000_000 -> '3.6B', 600_000_000 -> '600M'."""
    if n >= 10e9:
        return f"{n / 1e9:.0f}B"
    if n >= 1e9:
        return f"{n / 1e9:.1f}B".replace(".0B", "B")
    return f"{n / 1e6:.0f}M"


def params_label(facts: dict | None) -> str | None:
    """'35B · 3B active' (MoE) or '8.2B' (dense), from real tensor counts."""
    if not facts or not facts.get("total_params"):
        return None
    total, active = facts["total_params"], facts.get("active_params") or facts["total_params"]
    if facts.get("experts") and active < total * 0.9:
        return f"{human_params(total)} · {human_params(active)} active"
    return human_params(total)
