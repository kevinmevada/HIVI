# HIVI Memory — M1 Acceptance

**Status:** Complete  
**Date:** 2026-11-06  

## Checklist

| # | Criterion | Result |
|---|-----------|--------|
| 1 | `hindsight-api` installs on Windows | Pass — `.venv-hindsight`, `hindsight-api==1.2.0` |
| 2 | Starts with embedded pg0 | Pass |
| 3 | Health on `:8888` | Pass — `GET /health` → 200 |
| 4 | Bank `hivi_project_001` created | Pass — `PUT /v1/default/banks/hivi_project_001` |
| 5 | Connects to existing llama.cpp | Pass — log: `base_url=http://127.0.0.1:8080/v1`, connection verified |
| 6 | Retain + recall smoke | Pass — PySide6 decision retained; recall returned it (~115 ms) |
| 7 | No second model process | Pass — `llama-server` count remained **1** |
| 8 | Data survives Hindsight restart | Pass — after stop/start, bank still listed (`fact_count=1`) and recall succeeded |

## Ops notes

- Isolated venv: `.venv-hindsight` (does not modify main HIVI `.venv`)
- Data: `data/hindsight` + pg0 embedded DB
- Embeddings/reranker: local CPU (`CUDA_VISIBLE_DEVICES=` empty); HF cache forced to `Models\.hf_cache` (avoids broken `D:\` `HF_HOME`)
- Start script: `scripts/start_hindsight.ps1`
- LLM concurrency: `HINDSIGHT_API_LLM_MAX_CONCURRENT=1`

## Not in M1 (by design)

- No `src/hivi/memory` Python adapter (M2)
- No LangGraph wiring (M4)
- No Graphiti
- No MCP as primary API

## Next

**M2 — HIVI memory adapter** when authorized.
