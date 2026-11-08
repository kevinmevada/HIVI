# HIVI Memory — M0 Compatibility Audit & Integration Spec

**Status:** Complete (audit only)  
**Date:** 2026-11-06  
**Rule:** No packages installed; no HIVI runtime code changed for memory.

---

## 1. Current repository state

| Area | Finding |
|------|---------|
| `src/hivi/memory/` | Empty stub (`__init__.py` only, 0 bytes) |
| `src/hivi/orchestrator/` | Empty stub |
| `src/hivi/storage/` | Empty stub |
| `src/hivi/security/` | Empty stub |
| `src/hivi/agents/` | Empty stub |
| SQLite project DB | None found (no app `.db` / `.sqlite` yet) |
| LangGraph graphs | Declared in deps; **no HIVI workflow graphs implemented** |
| Live chat / HAIE | `interfaces/hivi-chat/` (separate from `src/hivi`) |

**Conclusion:** The package layout matches the blueprint skeleton. Memory, orchestration, and storage are greenfield inside `src/hivi/`.

---

## 2. Dependency compatibility

| Package | In `pyproject.toml` / `uv.lock` | Notes |
|---------|----------------------------------|--------|
| Python | **3.13.9** (venv) | OK for LangGraph/Pydantic; validate Hindsight wheels on 3.13 before hard-pinning |
| `langgraph` | ≥1.2.19 (locked) | Present — use for checkpoints only |
| `langchain-core` | ≥1.7.3 (locked) | Present — keep as optional utilities, not full LangChain app |
| `pydantic` / `pydantic-settings` | Present | Use for memory DTOs + settings |
| `httpx` | Present | Prefer for Hindsight HTTP client if SDK is heavy |
| **Hindsight** | **Absent** | Do not add until M1 after install smoke |
| **Graphiti** | **Absent** | Deferred (M7) |
| Neo4j / FalkorDB | Absent | Not needed for M1–M6 |

---

## 3. Local inference endpoint (shared)

| Item | Value |
|------|--------|
| HIVI UI | `http://127.0.0.1:7860` |
| OpenAI-compatible backend | `http://127.0.0.1:8080/v1` |
| Current model (example) | Qwen3-Coder-30B-A3B Q4_K_M via llama.cpp |
| Engines | llama.cpp / Colibrì / AirLLM (vendored under `src/hivi/inference/`) |

### Locked inference policy for Hindsight

**Do not** use Hindsight’s built-in `llamacpp` provider (it would spawn a second server and may auto-download Gemma).

**Do** configure Hindsight as OpenAI-compatible client against HIVI’s existing server:

```text
HINDSIGHT_API_LLM_PROVIDER=openai
HINDSIGHT_API_LLM_BASE_URL=http://127.0.0.1:8080/v1
HINDSIGHT_API_LLM_MODEL=<id from /v1/models or "local">
HINDSIGHT_API_LLM_API_KEY=hivi-local   # dummy if server ignores auth
HINDSIGHT_API_LLM_MAX_CONCURRENT=1     # memory must not starve interactive chat
```

Optional later: dedicated low-priority retain/reflect caps  
`HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT=1`, reflect similarly.

**Embeddings:** prefer Hindsight `local` / `onnx` on **CPU** (`FORCE_CPU` / max concurrent 1 on Windows) so 8 GB VRAM stays for the coding model.

---

## 4. Hindsight deployment recommendation (Windows)

| Option | Verdict |
|--------|---------|
| Native `pip install hindsight-api` + embedded **pg0** | **Preferred for M1** — docs mark Windows + pg0 supported |
| Built-in `llamacpp` provider | **Rejected** for HIVI (second GPU/CPU inference process) |
| Docker/WSL2 | Fallback if native pg0 / ML extras fail |
| External PostgreSQL + pgvector | Only if embedded pg0 proves unstable |

**API default:** `http://localhost:8888`  
**Bank MCP (later):** `http://localhost:8888/mcp/{bank_id}/` — not M1.

Data dirs must live on NVMe (e.g. `%USERPROFILE%\.hindsight` or `C:\Users\mevad\Desktop\HIVI\data\hindsight`), **never** on USB `D:`.

---

## 5. Identity & bank naming (locked for M1)

| Concept | Convention |
|---------|------------|
| Application | `hivi` |
| First project room | `project_001` |
| Hindsight bank ID | `hivi_project_001` |
| Isolation | **One bank per project** (shared by all agents in that room) |
| Agent identity | Tags only — not separate banks |

### Required tags (HIVI convention)

```
project:001
department:{planning|architecture|development|review|testing|delivery}
agent:{role}_{nn}
task:{task_id}
type:{episodic|semantic|procedural|decision|failure|verification|constraint|open_question}
source:{artifact_or_event_id}
status:{unverified|supported|verified|superseded|disputed|invalidated}
```

---

## 6. Persistence ownership (locked)

| Store | Owns | Must not own |
|-------|------|--------------|
| **Hindsight** | Searchable long-term knowledge | Workflow checkpoints |
| **LangGraph** | Thread checkpoints / resume | Project knowledge archive |
| **SQLite** (`src/hivi/storage`) | Project/task/event metadata, permission rows, retention queue | Semantic search |
| **Filesystem** | Code, reports, datasets, evidence files | Agent coordination |

Evidence rule: every important memory carries provenance pointing at a filesystem or SQLite event ID. Extracted memory never outranks source artifacts.

---

## 7. Proposed `src/hivi/memory/` layout (implement from M2)

```
src/hivi/memory/
  __init__.py
  interfaces.py          # MemoryProvider protocol
  models.py              # Pydantic retain/recall/reflect/curate DTOs
  manager.py             # Gateway — only entry agents call
  scopes.py              # project → bank mapping
  context_builder.py
  event_selector.py
  provenance.py
  retention_queue.py
  policies.py
  curation.py
  providers/
    hindsight.py         # Adapter only; no SDK leakage to agents
```

Agents call `memory_manager.recall(...)` — never `hindsight_client` directly.

---

## 8. LangGraph boundary (M4, not M1)

Reusable nodes (when orchestrator exists):

1. `load_project_context` → recall + budgeted context  
2. `store_execution_memory` → selective retain (async queue)  
3. `review_project_memory` → reflect (rare)

Checkpoints remain LangGraph’s job; Hindsight is not a checkpoint store.

---

## 9. Resource priority (8 GB VRAM / 32 GB RAM)

| Workload | Priority |
|----------|----------|
| Interactive HIVI chat / coding agent | Highest |
| Dev / test / review agents | High |
| Memory recall | Medium |
| Retain / consolidation | Low |
| Reflect | Lowest |

Cap Hindsight LLM concurrency to **1** while a user-facing model is loaded.

---

## 10. Risks & blockers for M1

1. **Python 3.13 + Hindsight wheels** — confirm install; if blocked, use 3.11/3.12 side env or Docker.  
2. **VRAM contention** — coding MoE (Qwen3-Coder-30B) already owns GPU; memory LLM must share same server with strict concurrency, or use a tiny CPU-only model later.  
3. **Structured extraction quality** — Qwen3-Coder may be weak at fact JSON; benchmark retain quality before trusting memories.  
4. **Empty orchestrator** — memory gateway can be built and unit-tested before LangGraph graphs exist.  
5. **No SQLite yet** — M2/M3 should create `data/hivi/hivi.sqlite` for event ledger before async retain.

---

## 11. Acceptance criteria for M0 (this document)

- [x] Inspected `src/hivi/memory/` and related stubs  
- [x] Inspected `pyproject.toml` / `uv.lock`  
- [x] Identified local OpenAI-compatible endpoint for Hindsight  
- [x] Chose Windows native + pg0 first; Docker fallback  
- [x] Locked bank naming + tag schema  
- [x] Locked four-way persistence split  
- [x] No dependency installs performed  

---

## 12. Next milestone (M1) — only when approved

1. Isolated venv or optional-extra install of `hindsight-api` (do **not** pull `[local-llm]` unless needed).  
2. Start `hindsight-api` with embedded pg0 on NVMe.  
3. Point LLM at `http://127.0.0.1:8080/v1`.  
4. Health check `:8888`; create bank `hivi_project_001`.  
5. Manual retain/recall smoke (CLI or HTTP) — still no full adapter.  

**Do not start Graphiti. Do not add MCP as the primary path.**

---

## Locked baseline (section 16 of blueprint)

| Decision | Choice |
|----------|--------|
| Primary memory | Hindsight |
| Temporal graph | Graphiti later |
| Orchestration | LangGraph |
| Core | Python in `src/hivi/` |
| Hindsight DB | Embedded PostgreSQL (pg0) initially |
| App records | SQLite (to create) |
| Artifacts | Project filesystem |
| Isolation | One bank per project |
| Inference | HIVI-managed llama.cpp / Colibrì |
| Cloud inference | Not required |
