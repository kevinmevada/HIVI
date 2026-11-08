# HIVI Memory — M2 Acceptance

**Status:** Complete  
**Date:** 2026-11-06  

## Checklist

| # | Criterion | Result |
|---|-----------|--------|
| 1 | Provider-agnostic DTOs + `MemoryProvider` protocol | Pass — `models.py`, `interfaces.py` |
| 2 | Scope → bank mapping (`hivi_project_001`) | Pass — `scopes.py` + unit tests |
| 3 | Hindsight HTTP adapter (httpx only, no SDK) | Pass — `providers/hindsight.py` |
| 4 | `MemoryManager` gateway for agents | Pass — `manager.py`, `get_memory_manager()` |
| 5 | Package exports (`from hivi.memory import …`) | Pass — `__init__.py` |
| 6 | Unit tests (fake provider) | Pass — `tests/memory/test_scopes.py`, `test_hindsight_unit.py` |
| 7 | Live retain + recall via manager | Pass — `tests/memory/test_hindsight_live.py` (sync retain ~90s on Colibrì 8B; recall hit PySide6) |
| 8 | No second inference process | Pass — still shares `:8080/v1` |

## Layout

```
src/hivi/memory/
  models.py
  interfaces.py
  scopes.py
  manager.py
  providers/hindsight.py
  M0_COMPATIBILITY_AUDIT.md
  M1_ACCEPTANCE.md
  M2_ACCEPTANCE.md
tests/memory/
  test_scopes.py
  test_hindsight_unit.py
  test_hindsight_live.py
```

## Ops notes (Colibrì / local OpenAI-compat)

- `scripts/start_hindsight.ps1` auto-resolves `HINDSIGHT_API_LLM_MODEL` from `GET /v1/models` when unset or `"local"`.
- Colibrì rejects `response_format` grammars → script sets:
  - `HINDSIGHT_API_LLM_OPENAI_COMPATIBLE_JSON_MODE=false`
  - `HINDSIGHT_API_LLM_STRICT_SCHEMA=false`
- Agents must call `MemoryManager`, not Hindsight REST/SDK directly.

## Not in M2 (by design)

- LangGraph / agent wiring (M3/M4)
- Graphiti
- Async retain queue / SQLite event ledger
- MCP as primary memory API

## Next

M3–M7 completed (see sibling acceptance docs).
