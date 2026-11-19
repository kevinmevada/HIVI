"""Real model facts from GGUF files (sizes, MoE active share, context, thinking),
the engine pin and the thinking toggle's server side."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import gguf_context
import httpx
import pytest
import registry
import server

# GGUF value types
U32, STR, ARR = 4, 8, 9


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path: Path, *, arch: str = "qwen3", ctx: int = 40960, template: str = "",
               tensors: list[tuple[str, tuple[int, ...]]] = (), experts: tuple[int, int] | None = None,
               vocab: int = 300) -> Path:
    """A GGUF header with metadata and a tensor table (no tensor data: only headers are read)."""
    kv = [_s("general.architecture") + struct.pack("<I", STR) + _s(arch),
          _s(f"{arch}.context_length") + struct.pack("<I", U32) + struct.pack("<I", ctx),
          _s("tokenizer.ggml.tokens") + struct.pack("<I", ARR) + struct.pack("<I", STR)
          + struct.pack("<Q", vocab) + b"".join(_s(f"t{i}") for i in range(vocab))]
    if template:
        kv.append(_s("tokenizer.chat_template") + struct.pack("<I", STR) + _s(template))
    if experts:
        kv.append(_s(f"{arch}.expert_count") + struct.pack("<I", U32) + struct.pack("<I", experts[0]))
        kv.append(_s(f"{arch}.expert_used_count") + struct.pack("<I", U32) + struct.pack("<I", experts[1]))
    tens = b""
    for name, dims in tensors:
        tens += _s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        tens += struct.pack("<I", 0) + struct.pack("<Q", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors))
                     + struct.pack("<Q", len(kv)) + b"".join(kv) + tens)
    return path


QWEN3_TEMPLATE = "{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>{%- endif %}"
OSS_TEMPLATE = "<|start|>system<|message|>Reasoning: {{ reasoning_effort }}<|end|><|channel|>analysis"
R1_TEMPLATE = "<｜Assistant｜><think>\n"


def dense_8b():
    # 36 layers x 200M + embeddings 600M + output 600M = 8.4B
    return ([(f"blk.{i}.attn_q.weight", (10_000, 20_000)) for i in range(36)]
            + [("token_embd.weight", (4096, 146_484)), ("output.weight", (4096, 146_484))])


def moe_35b_a3b():
    # shared 2B + experts 32B over 128 experts with 8 used -> active = 2B + 32B*8/128 = 4B
    shared = [("token_embd.weight", (100_000, 10_000)), ("output.weight", (100_000, 10_000))]
    exps = [(f"blk.{i}.ffn_up_exps.weight", (128, 1000, 62_500)) for i in range(4)]   # 4 x 8B
    return shared + exps


def test_dense_model_facts(tmp_path):
    f = write_gguf(tmp_path / "Qwen3-8B-Q4_K_M.gguf", template=QWEN3_TEMPLATE, tensors=dense_8b())
    facts = gguf_context.model_facts(f)
    assert facts["total_params"] == 36 * 200_000_000 + 2 * 4096 * 146_484
    assert facts["active_params"] == facts["total_params"] and facts["experts"] == 0
    assert facts["trained_ctx"] == 40960 and facts["thinking"] == "toggle"
    assert gguf_context.params_label(facts) == "8.4B"


def test_moe_model_shows_active_share(tmp_path):
    f = write_gguf(tmp_path / "Qwen3.6-35B-A3B.gguf", arch="qwen35moe", ctx=262144,
                   template=QWEN3_TEMPLATE, tensors=moe_35b_a3b(), experts=(128, 8))
    facts = gguf_context.model_facts(f)
    exp = 4 * 128 * 1000 * 62_500
    assert facts["total_params"] == 2_000_000_000 + exp
    assert facts["active_params"] == 2_000_000_000 + exp * 8 // 128
    assert (facts["experts"], facts["experts_used"]) == (128, 8)
    assert gguf_context.params_label(facts) == "34B · 4B active"


def test_split_models_add_up_all_parts(tmp_path):
    t = dense_8b()
    a = write_gguf(tmp_path / "big-00001-of-00002.gguf", template=QWEN3_TEMPLATE, tensors=t[:20])
    write_gguf(tmp_path / "big-00002-of-00002.gguf", tensors=t[20:])
    facts = gguf_context.model_facts(a)
    assert facts["total_params"] == 36 * 200_000_000 + 2 * 4096 * 146_484


@pytest.mark.parametrize("template,kind", [
    (QWEN3_TEMPLATE, "toggle"), (OSS_TEMPLATE, "effort"), (R1_TEMPLATE, "always"),
    ("{{ messages }}", "none"), ("", "none"),
])
def test_thinking_kind_from_the_chat_template(template, kind):
    assert gguf_context.thinking_kind(template) == kind


def test_unreadable_files_fall_back_quietly(tmp_path):
    bad = tmp_path / "broken.gguf"
    bad.write_bytes(b"not a gguf at all")
    assert gguf_context.model_facts(bad) is None
    assert gguf_context.cached_facts(bad) == (True, None)
    assert gguf_context.params_label(None) is None


def test_facts_are_cached_on_disk_per_file_version(tmp_path, monkeypatch):
    f = write_gguf(tmp_path / "m.gguf", tensors=dense_8b())
    gguf_context.model_facts(f)
    calls = []
    monkeypatch.setattr(gguf_context, "read_file_facts", lambda p: calls.append(p) or {})
    monkeypatch.setattr(gguf_context, "_INFO", {})
    monkeypatch.setattr(gguf_context, "_INFO_LOADED", False)        # as after a restart
    assert gguf_context.model_facts(f)["total_params"] > 0 and calls == []
    write_gguf(f, tensors=dense_8b()[:3])                            # file replaced
    import os
    os.utime(f, (1, 1))
    gguf_context.cached_facts(f)
    assert gguf_context.cached_facts(f)[0] is False                  # new version: re-read


# ------------------------------------------------------------------ registry -- #

@pytest.fixture()
def models_root(tmp_path, monkeypatch):
    root = tmp_path / "Models"
    monkeypatch.setenv("HIVI_MODELS_ROOT", str(root))
    monkeypatch.setenv("HIVI_DATA_ROOT", str(tmp_path / "data"))
    registry.clear_scan_cache()
    yield root
    registry.clear_scan_cache()


def test_picker_shows_real_sizes_once_read(models_root):
    write_gguf(models_root / "General/gpt/hauhau-aggressive-Q5_K_M.gguf", arch="gptoss",
               template=OSS_TEMPLATE, tensors=moe_35b_a3b(), experts=(128, 8))
    write_gguf(models_root / "Coding/q/Qwen3-8B-Q4_K_M.gguf", template=QWEN3_TEMPLATE, tensors=dense_8b())
    first = {m.name: m for m in registry.scan_models()}
    hau = next(m for n, m in first.items() if "hauhau" in n.lower())
    assert hau.params == "?" or hau.facts_pending          # the name says nothing yet
    registry.clear_scan_cache()
    now = {m.id: m for m in registry.scan_models()}
    hau = next(m for m in now.values() if "hauhau" in m.id)
    assert hau.params == "34B · 4B active" and not hau.facts_pending
    assert hau.thinking == "effort" and hau.experts == 128
    q = next(m for m in now.values() if "qwen3-8b" in m.id)
    assert q.params == "8.4B" and q.thinking == "toggle" and q.trained_ctx == 40960


def test_split_parts_are_one_model(models_root):
    t = dense_8b()
    write_gguf(models_root / "General/big/Llama-70B-Q4_K_M-00001-of-00002.gguf", tensors=t[:20])
    write_gguf(models_root / "General/big/Llama-70B-Q4_K_M-00002-of-00002.gguf", tensors=t[20:])
    ids = [m.id for m in registry.scan_models()]
    assert len(ids) == 1 and ids[0].endswith("00001-of-00002")


def test_engine_pin_round_trip(models_root):
    write_gguf(models_root / "Coding/q/Qwen3.6-35B-A3B-Q5_K_M.gguf", template=QWEN3_TEMPLATE,
               tensors=moe_35b_a3b(), experts=(128, 8))
    (m,) = registry.scan_models()
    assert m.auto_engine == "colibri" and m.pinned is None and m.engine == "colibri"
    registry.set_engine_pin(m.id, "llama")
    (m,) = registry.scan_models()
    assert m.engine == "llama" and m.pinned == "llama" and m.auto_engine == "colibri"
    assert "auto would pick colibri" in m.engine_reason
    registry.set_engine_pin(m.id, "auto")
    (m,) = registry.scan_models()
    assert m.pinned is None and m.engine == "colibri"


# ------------------------------------------------------------------- server -- #

def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://hivi")


async def test_api_models_and_engine_pin(models_root):
    write_gguf(models_root / "Coding/q/Qwen3-8B-Q4_K_M.gguf", ctx=16384, template=QWEN3_TEMPLATE,
               tensors=dense_8b())
    registry.scan_models()
    registry.clear_scan_cache()
    async with _client() as c:
        (m,) = (await c.get("/api/models")).json()["models"]
        assert m["params"] == "8.4B" and m["thinking"] == "toggle"
        assert m["run_ctx"] == 16384           # never past what it was trained for
        assert m["trained_ctx"] == 16384 and m["pinned"] is None
        r = await c.post("/api/engine", json={"id": m["id"], "engine": "colibri"})
        assert r.status_code == 200 and r.json()["engine"] == "colibri"
        (m,) = (await c.get("/api/models")).json()["models"]
        assert m["pinned"] == "colibri"
        r = await c.post("/api/engine", json={"id": m["id"], "engine": "turbo"})
        assert r.status_code == 400


async def test_thinking_forced_on_gets_the_whole_window(monkeypatch):
    sent = []

    async def stream(request, path, body, extra=None, on_close=None, **kw):
        sent.append(json.loads(body))
        if on_close:
            on_close()
        return server.Response(content=b"data: [DONE]\n\n", media_type="text/event-stream")

    monkeypatch.setattr(server, "_proxy_stream", stream)
    def haie(raw):   # HAIE plans a short chat answer
        return json.dumps({**json.loads(raw), "max_tokens": 2048}).encode(), {"X-HIVI-HAIE-Intents": "chat"}

    monkeypatch.setattr(server, "_adapt_chat_body", haie)
    monkeypatch.setattr(server.model_router, "resolve", lambda name: None)
    monkeypatch.setattr(server, "backend_model_name", lambda: None)
    monkeypatch.setattr(server.STATE, "context_size", 16384)
    monkeypatch.setattr(server.STATE, "engine", "llama")
    body = {"model": "m", "stream": True, "max_tokens": None,
            "chat_template_kwargs": {"enable_thinking": True},
            "messages": [{"role": "user", "content": "prove it"}]}
    async with _client() as c:
        await c.post("/v1/chat/completions", json=body, headers={"X-HIVI-Think": "on"})
        await c.post("/v1/chat/completions", json={**body, "chat_template_kwargs": {}})
    on, auto = sent[0], sent[1]
    assert on["chat_template_kwargs"]["enable_thinking"] is True
    assert on["max_tokens"] > 14000            # reasoning plus answer: all the room left
    assert auto["max_tokens"] == 2048          # Auto keeps HAIE's budget
