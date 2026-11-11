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

## One-time migration for an existing clone

The folders are now in `.gitignore`. Stop tracking them without deleting your
local copies:

```powershell
git rm -r --cached src/hivi/inference/llama.cpp-master src/hivi/inference/colibri-LLM-main src/hivi/inference/airllm-main
git commit -m "Stop vendoring inference engines; fetch with scripts/fetch_backends.ps1"
```

This shrinks future clones. Old commits still contain the vendored trees; only
history rewriting (e.g. `git filter-repo`) removes them from the repo size.
