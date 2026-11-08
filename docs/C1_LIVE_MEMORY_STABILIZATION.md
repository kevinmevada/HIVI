# C1 — Live Hindsight Retain Stabilization

**Status:** Pass (smoke)  
**Date:** 2026-11-06  

## Problem

On Colibrì Qwen3-8B (`ctx=4096`):

- Default Hindsight `response_format` grammars → 400 (fixed earlier: JSON mode off).
- Default `RETAIN_MAX_COMPLETION_TOKENS=64000` → runaway replies / timeouts.
- Auto-consolidation prompts → `context_length_exceeded` (~4.5k tokens into 4k window).
- A stuck consolidation on `hivi_project_001` starved `LLM_MAX_CONCURRENT=1` and blocked retains.

## Mitigations in `scripts/start_hindsight.ps1`

- `HINDSIGHT_API_LLM_OPENAI_COMPATIBLE_JSON_MODE=false`
- `HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION=false`
- `HINDSIGHT_API_RETAIN_CHUNK_SIZE=800`
- `HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=1024`
- `HINDSIGHT_API_LLM_MAX_CONCURRENT=1` / retain LLM concurrency 1
- Model id auto-resolved from `:8080/v1/models`

## Smoke result

Command: `python scripts/smoke_hindsight_retain.py`  
Bank: `hivi_project_099` (clean; avoids stuck `001` consolidation)

| Metric | Value |
|--------|-------|
| Retain | success |
| Retain latency | **~109 s** |
| Recall hits | ≥1 semantic facts (UI web-based; div rejects zero) |
| Result file | `data/hivi/c1_retain_smoke.json` |

## Colibrì limits (benchmark notes)

| Limit | Observation |
|-------|-------------|
| Context | 4096 — consolidation prompts overflow; keep consolidation off |
| Sync retain | ~90–110 s per short document |
| Extraction | Rewrites text (markers may disappear; semantic recall still works) |
| Grammar / JSON mode | Not supported — must use soft schema-in-prompt |

## Remaining risk

- Bank `hivi_project_001` may still have a pending consolidation op in Hindsight’s DB; prefer clean banks or purge that op before heavy E2E on `001`.
- Live ProjectRoom (C2) should use these start-script settings and avoid stacking many `wait=True` retains while any consolidation is active.

## Exit

C1 smoke gate green. Ready for **C2 — live ProjectRoom**.
