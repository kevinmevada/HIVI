# Inference backends

HIVI drives third-party engines as separate processes. Their source is **not**
committed (it was ~135 MB of vendored code, most of it unused). Fetch it with
`scripts/fetch_backends.ps1`, which clones into the same folders the runtime
already expects.

| Engine | Upstream | Pinned ref | Folder | Used by |
|--------|----------|-----------|--------|---------|
| Colibrì | github.com/JustVugg/colibri | `v2.1.0` — the build all benchmarks used. The previously vendored copy was 1.10.2 and is rejected by the runtime | `src/hivi/inference/colibri-LLM-main` | chat runtime (`coli serve`) |
| llama.cpp | github.com/ggml-org/llama.cpp | set to the build you run; ≥ `b11454` for K2 Horizon | `src/hivi/inference/llama.cpp-master` | chat runtime (`llama-server`), ProjectRoom, Hindsight |
| AirLLM | github.com/lyogavin/airllm | optional (`-WithAirLLM`) | `src/hivi/inference/airllm-main` | nothing today (needs safetensors; library is GGUF) |
| Ollama | installed app | n/a | external, `http://127.0.0.1:11434` | nothing today |

If the Colibrì tag name differs upstream, pass the commit you tested:
`.\scripts\fetch_backends.ps1 -ColibriRef <sha>`, then update this table.

## Binary locations (override per machine)

| Variable | Default |
|----------|---------|
| `HIVI_ROOT` | the repo containing `src/hivi/paths.py` |
| `HIVI_MODELS_ROOT` | `storage.models_root` in `configs/storage.yaml` (default `Models/`) |
| `HIVI_DATA_ROOT` | `storage.data_root` (default `data/`) |
| `HIVI_LLAMA_SERVER` | `data/P1_1/runtime/llama.cpp/bin/llama-server(.exe)`, else `llama-server` on PATH |
| `HIVI_COLI` | `src/hivi/inference/colibri-LLM-main/c/coli` |
| `HIVI_BACKEND_PORT` | `8080` |
| `HIVI_EMBED_PORT` | `8081` (embedding side service, see `MODEL_ROUTER.md`) |
| `HIVI_RERANK_PORT` | `8082` (reranker side service) |
| `HIVI_LOG_DIR` | `data/logs` |
| `HIVI_LLAMA_ROUTER` | `1` — keep one llama-server running and load models inside it; `0` restarts per model |
| `HIVI_MODELS_MAX` | `3` — llama models the router may hold at once |
| `HIVI_MODELS_MEMORY_MARGIN` | `512` MiB of VRAM kept free; the router unloads idle models (oldest first) to load a new one. `off` lets models share the GPU |
| `HIVI_RAM_RESERVE_GB` | `8` — HIVI unloads idle router models so loaded weights + this reserve fit in RAM (the router itself assumes RAM is unlimited) |

## Which engine runs a model

`interfaces/hivi-chat/registry.py` picks per file:

- **Colibrì**: Qwen3 / 3.5 / 3.6 / 3.8 text models, uncensored builds included.
  Its GPU/RAM tiering is built for models larger than VRAM (27B on the 8 GB
  RTX 5060: ~38 s vs ~125 s for llama.cpp, `benchmarks/27b_comparison`).
- **llama.cpp**: GPT-OSS, Gemma, Qwen2.5, Qwen3-Coder-30B, anything with a
  vision projector (`mmproj-*.gguf` in its folder), embeddings and rerankers.
- If Colibrì cannot import or start a file, HIVI falls back to llama.cpp for
  the rest of the session (the status line says so).
- Pin a model either way; it applies on the next load:
  `POST /api/engine {"id": "<model id>", "engine": "llama" | "colibri" | "auto"}`
  (stored in `data/run/hivi_engines.json`).

llama.cpp models run in **router mode** when the llama-server build has it
(`--models-preset`): switching between llama models keeps the process, and a
model that is still loaded switches instantly. Switching to or from Colibrì
still restarts the engine. The generated preset is
`data/run/hivi_llama_models.ini`; the router log is `data/logs/hivi_llama_router.log`.

## One-time migration for an existing clone

The folders are now in `.gitignore`. Stop tracking them without deleting your
local copies:

```powershell
git rm -r --cached src/hivi/inference/llama.cpp-master src/hivi/inference/colibri-LLM-main src/hivi/inference/airllm-main
git commit -m "Stop vendoring inference engines; fetch with scripts/fetch_backends.ps1"
```

This shrinks future clones. Old commits still contain the vendored trees; only
history rewriting (e.g. `git filter-repo`) removes them from the repo size.
