# HIVI — Repository Cleanup Audit (Stage A)

**Date:** 2026-11-06  
**Scope:** Full repository inventory and classification  
**Status:** Stage A complete; **Stage B Batch 1 executed 2026-11-06** (see §14).  
**Rule:** Further cleanup requires explicit approval of remaining batches.

---

## 1. Repository baseline

| Item | Value |
|------|--------|
| Root | `C:/Users/mevad/Desktop/HIVI` |
| Branch | `main` (tracking `origin/main`) |
| Uncommitted changes | **Preserve** — do not reset/overwrite |
| Dirty tracked files | `interfaces/hivi-chat/{index.html,registry.py,runtime.py,server.py}`, `pyproject.toml`, `scripts/download_hf_gguf.ps1`, `src/hivi/{memory,orchestrator,storage}/__init__.py` |
| Notable untracked | Memory vertical + tests, Core Integration docs, HAIE, smoke scripts, `Adaptive-LLM-Sampling-main/` |
| Offline unit tests (pre-cleanup) | `tests/memory` excluding live: **20 passed in 1.09s** (2026-11-06) |
| Live tests | **Not re-run** for this audit (stack was stopped by request) |

### Approximate on-disk footprint (working tree)

| Path | ~Size | Gitignored? | Cleanup in scope? |
|------|-------|-------------|-------------------|
| `Models/` | ~130 GB | Yes | **No** — user model library |
| `.venv-hindsight/` | ~1.7 GB | Local venv | Out of product source; local-only |
| `.venv/` | ~0.8 GB | Yes | Out of source cleanup |
| `data/P1_1/` (mostly runtime) | ~4.3 GB | Yes (`data/`) | **Protect** runtime binaries |
| `src/` (incl. vendored) | ~203 MB | Tracked | Yes — audit below |
| `.git/` | ~120 MB | — | Do not rewrite history |
| `Adaptive-LLM-Sampling-main/` | ~1.9 MB | Untracked | Candidate |
| Rest of repo (docs, scripts, interfaces, tests) | small | Mixed | Yes |

**Note:** Largest recoverable *tracked* source is vendored trees under `src/hivi/inference/` (~203 MB). Model weights and `data/` are protected / gitignored.

---

## 2. Architecture snapshot (what is actually active)

HIVI today is two parallel stacks:

1. **Core memory vertical (HIVI-owned):** `src/hivi/memory` + `orchestrator` + `storage` + `tests/memory` + smoke scripts. Talks to Hindsight (`:8888`) and OpenAI-compat inference (`:8080`).
2. **Chat / HAIE UI:** `interfaces/hivi-chat/` — FastAPI + HAIE planner; launches Colibrì or `llama-server` as subprocesses. **Does not import `hivi` yet** (C3 in Core Integration Spine).

Empty package stubs under `src/hivi/` (`agents`, `api`, `core`, …) are **planned namespaces** (README + C3–C5), not abandoned trash.

Authoritative docs (preserve):

- `docs/ARCHITECTURE_COMPLETENESS_AUDIT.md`
- `docs/CORE_INTEGRATION_SPINE.md`
- `docs/C1_LIVE_MEMORY_STABILIZATION.md`
- `docs/CONTEXTUAL_INTELLIGENCE_ACCEPTANCE.md`
- `src/hivi/memory/M1`–`M7` acceptance / Graphiti decision

---

## 3. Vendored repositories

### Classification summary

| Tree | ~Size | Tracked files | Classification | Rationale |
|------|-------|---------------|----------------|-----------|
| `src/hivi/inference/llama.cpp-master/` | ~174 MB | ~3644 | **REQUIRED SOURCE** | CMake source of built runtime in `data/P1_1/runtime/llama.cpp/`. No HIVI Python import; rebuild/provenance depend on it. |
| `src/hivi/inference/colibri-LLM-main/` | ~18 MB | ~648 | **UNCERTAIN** | Incomplete vs live install. Chat/HAIE use `Downloads\colibri-main\...`. P1.3 matrix still points at vendored `qwen36.exe`. |
| `src/hivi/inference/airllm-main/` | ~10 MB | ~99 | **UNUSED** | No imports; benches mark AirLLM N/A; `configs/inference.yaml` has `enabled: false`. |
| `src/hivi/inference/ollama/` | &lt;1 KB | stub | **UNUSED stub** | Not a clone; real Ollama is external daemon. Keep as namespace or replace later — low cost. |
| `Adaptive-LLM-Sampling-main/` (repo root) | ~1.9 MB | 0 (untracked) | **UNUSED** | No HIVI references; product sampling is HAIE. |
| `data/P1_1/runtime/llama.cpp/` | ~681 MB+ | gitignored | **EXTRACTED / PROTECT** | Active `llama-cli` / `llama-server` + CUDA DLLs. |

### Runtime path evidence (do not assume “no import” ⇒ unused)

| Consumer | Actual binary / path |
|----------|----------------------|
| `interfaces/hivi-chat/runtime.py` | Downloads Colibrì `coli` **or** `data/P1_1/runtime/.../llama-server.exe` |
| P1 benches / bootstrap | Inventory checks vendored trees; llama runtime under `data/P1_1` |
| `data/P1_3/.../interface_matrix.json` | Vendored Colibrì `qwen36.exe` (diverges from chat) |

### Unresolved (require approval before delete)

1. Keep `llama.cpp-master` forever vs re-fetch on rebuild only?
2. Canonical Colibrì: vendored tree vs Downloads — align scripts before removing either.
3. Is AirLLM still a product goal?
4. Archive vs delete P0/P1 scripts and `benchmarks/27b_comparison`?

---

## 4. HIVI-owned source (`src/hivi/`)

### Active

| Package | Modules | Role |
|---------|---------|------|
| `memory/` | manager, providers, policies, curation, queue, context_builder, … | Memory gateway + Hindsight adapter |
| `orchestrator/` | graphs, memory_nodes, preflight, project_room | LangGraph + ProjectRoom + preflight |
| `storage/` | sqlite_db, ledger | Event ledger + schema foundation (C4) |

### Interface / foundation — **KEEP** (even if empty)

`agents`, `api`, `core`, `evaluation`, `hardware`, `models`, `monitoring`, `retrieval`, `router`, `security`, `tools`, `inference/` (+ `ollama/` stub).

These match README layout and Core Integration Spine C3–C5. Removing them is an architecture decision, not cleanup.

### Cleanup signals (non-destructive)

- Unused imports (Ruff F401) in a few memory modules.
- Duplicated default Hindsight URL across `manager` / `preflight` / `hindsight` provider.
- Helpers with no call sites (e.g. `correction_to_dict`) — **UNCERTAIN**, keep for now.
- No confirmed abandoned HIVI-owned implementation modules.

---

## 5. Interfaces (`interfaces/hivi-chat/`)

| Path | Class | Action |
|------|--------|--------|
| `server.py`, `index.html`, `runtime.py`, `registry.py` | **Active** | Protect |
| `haie/**` (non-calibration) | **Active** | Protect |
| `haie/calibration/**` | **Experimental** | Keep while HAIE P1 scripts exist |
| `adaptive_sampling.py` | **Abandoned shim** | Candidate after confirm no external callers |
| `patch_ui.py` | **One-shot abandoned** | Candidate |
| `__pycache__/` | Generated | Safe local delete |

**Do not break** chat UI, HAIE, or backend launch while preparing C3.

---

## 6. Documentation

| Doc | Verdict |
|-----|---------|
| `docs/ARCHITECTURE_*`, `CORE_INTEGRATION_*`, `C1_*`, `CONTEXTUAL_*` | **Authoritative — keep** |
| `src/hivi/memory/M1`–`M6_ACCEPTANCE.md`, `M7_GRAPHITI_EVALUATION.md` | **Keep** (evidence / decision) |
| `M0_COMPATIBILITY_AUDIT.md` | **Keep as history**; content is a snapshot (outdated description of empty memory/) |
| Vendored upstream docs inside llama/colibri/airllm | Travel with those trees; keep licenses/attribution if trees remain |

No duplicate authoritative docs found under `docs/`. This audit file is additive.

---

## 7. Dependencies (`pyproject.toml`)

| Dependency | First-party use | Note |
|------------|-----------------|------|
| fastapi, uvicorn | Chat server | Keep |
| httpx | Memory, HAIE, preflight | Keep |
| langgraph, langgraph-checkpoint-sqlite | Orchestrator / ProjectRoom | Keep |
| pydantic | Memory models | Keep |
| huggingface-hub | Download / ops scripts | Keep |
| langchain-core | No direct import found | Likely transitive with LangGraph — **do not remove without lock analysis** |
| numpy, psutil, packaging, platformdirs, pydantic-settings, nvidia-ml-py, pyyaml, pyside6 | Little/no first-party import | **UNCERTAIN** — may be planned desktop/hardware or vendor tooling; Stage B needs per-dep proof |

Hatch wheel currently packages entire `src/hivi` including ~200 MB vendored trees — packaging exclude is a future hygiene item, not a delete.

`.venv-hindsight` is a separate local install (not declared in `pyproject.toml`).

---

## 8. Protected components (never delete in cleanup)

- `data/projects/`, project artifacts, evidence
- SQLite DBs under `data/hivi/`, Hindsight state, LangGraph checkpoints
- `Models/` and external/USB model libraries
- User configs / settings with local paths
- Active inference binaries (`data/P1_1/runtime/…`)
- Working chat + HAIE
- Memory / orchestrator / storage implementations + `tests/memory`, `tests/e2e`
- Accepted milestone reports and Core Integration docs
- Upstream LICENSE / copyright / attribution files for any retained vendor tree
- Uncommitted user work on `main` (commit or stash only with user request)

---

## 9. Proposed deletion lists

### A. Confirmed safe to remove (Stage B — still needs your explicit OK)

| Item | ~Recovery | Why safe | Risk |
|------|-----------|----------|------|
| `Adaptive-LLM-Sampling-main/` | ~1.9 MB | Untracked; unused; HAIE replaced it | None for HIVI runtime |
| `src/hivi/inference/airllm-main/` | ~10 MB | UNUSED; no runtime path | Loses offline AirLLM source if plans revive |
| `interfaces/hivi-chat/adaptive_sampling.py` | tiny | Dead re-export shim | None if HAIE remains |
| `interfaces/hivi-chat/patch_ui.py` | tiny | One-shot HTML rewriter | None |
| Local caches: `__pycache__/`, `.pytest_cache/`, `.ruff_cache/` | small | Regenerable | None |
| Regenerable smoke JSON e.g. `data/hivi/c1_retain_smoke.json` | tiny | Evidence duplicated in docs | None |
| Dated `data/p0_checklist_*.txt` extras (keep `*_latest` if desired) | small | Debug dumps | Low |
| Stale `benchmarks/**/logs/` / partial JSON | varies | Runtime noise | Low |

**Estimated easy recovery (confirmed candidates):** ~12–15 MB tracked/untracked source + caches (not counting gitignored `data/` logs).

### B. Requires approval (UNCERTAIN / high impact)

| Item | ~Size | Question |
|------|-------|----------|
| `src/hivi/inference/colibri-LLM-main/` | ~18 MB | Align on Downloads vs vendored first |
| `src/hivi/inference/llama.cpp-master/` | ~174 MB | Keep for rebuild, or document re-clone + rely on `data/P1_1/runtime`? |
| Empty package stubs under `src/hivi/` | negligible | Delete only if abandoning C3–C5 namespaces |
| `scripts/p0_*`, `p1_*`, `bench_27b/` | small code | Archive historical P-series? |
| `benchmarks/27b_comparison/` non-log artifacts | small | Keep SCOPE/reports? |
| Unused-looking pyproject deps (pyside6, pyyaml, …) | lock churn | Prove no planned/optional use |
| Hatch exclude of vendored trees from wheels | packaging | Separate PR |

### C. Do not remove

See §8. Also: do not “clean” `Models/`, USB libraries, or machines outside this repo.

---

## 10. Disk-space recovery estimate

| Scenario | Approx reclaim |
|----------|----------------|
| Confirmed safe batch only (§9A) | **~12–15 MB** (+ local caches) |
| + AirLLM already in §9A | included above |
| + Remove vendored Colibrì after path alignment | **+~18 MB** |
| + Remove `llama.cpp-master` after rebuild policy | **+~174 MB** (largest source win; **high risk** without rebuild story) |
| `data/` / `Models/` / venvs | **Out of scope** (multi-GB; protected or local) |

Cleanup will **not** make the checkout “small” while `Models/` and `data/P1_1/runtime` remain on disk — those dwarf source trees.

---

## 11. Before / after test results

| Suite | Before cleanup | After cleanup |
|-------|----------------|---------------|
| Offline `tests/memory` (no live) | **20 passed** (2026-11-06) | *N/A — Stage B not started* |
| Live Hindsight retain/recall | Previously green (C1 / bank 099); stack stopped | Not re-run |
| ProjectRoom E2E (C2) | Not green / aborted earlier | Not re-run |
| Import / lint / typecheck | Not fully run this audit | Deferred to Stage C |

**Do not claim live validation after cleanup until Stage C executes it.**

---

## 12. Stage B / C plan (awaiting approval)

### Stage B — incremental batches (suggest order)

1. **Batch 1 (lowest risk):** caches + `Adaptive-LLM-Sampling-main/` + `adaptive_sampling.py` + `patch_ui.py` + regenerable smoke JSON.  
2. **Batch 2:** `airllm-main/` + README/config path updates + LICENSE note if any attribution must remain elsewhere.  
3. **Batch 3 (approval gate):** Colibrì path unification, then possible vendored Colibrì removal.  
4. **Batch 4 (approval gate):** llama.cpp source policy.  
5. **Batch 5 (optional):** P0/P1 script archive; dependency prune with evidence.

Each batch: small diff → offline tests → report.

### Stage C — validation checklist

1. Import checks for `hivi.memory`, `hivi.orchestrator`, chat server module load  
2. Offline unit tests (`tests/memory`)  
3. Broader offline suite if present  
4. Ruff / mypy where configured  
5. Chat server startup (without requiring model load if possible)  
6. Inference backend startup (Colibrì or llama — as configured)  
7. Hindsight health  
8. Memory retain + recall smoke  
9. Confirm ProjectRoom modules + checkpoint paths intact  
10. Git status + disk usage before/after  

---

## 13. Concise cleanup summary

- **Audit complete; nothing deleted.**  
- Real product code is a thin **memory/orchestrator/storage** core + **hivi-chat/HAIE**; most of `src/hivi` is empty future namespaces or vendored inference trees.  
- Safest wins: untracked Adaptive-LLM tree, AirLLM vendor, dead chat shims, caches (~12–15 MB).  
- Largest source win (`llama.cpp-master` ~174 MB) needs an explicit rebuild policy — runtime already lives under `data/P1_1`.  
- Colibrì is **split-brained** (Downloads vs vendored) — fix paths before deleting either.  
- Do not remove empty package stubs or milestone docs without product decision.  
- **Next step:** approve Stage B Batch 1 (and optionally Batch 2). No cleanup until then.

---

## 14. Stage B Batch 1 execution report

**Approved:** Batch 1 only (2026-11-06).  
**Not touched:** `airllm-main/` (Batch 2), Colibrì/llama trees, empty stubs, deps, Models/data DBs.

### Pre-delete verification
- No Python imports of `adaptive_sampling` / `patch_ui` / `Adaptive-LLM-Sampling-main`.
- `index.html` has a comment mentioning Adaptive-LLM-Sampling presets only (not a runtime dependency).
- Smoke JSON regenerable; C1 evidence remains in `docs/C1_LIVE_MEMORY_STABILIZATION.md`.

### Deleted paths (19)
- `Adaptive-LLM-Sampling-main/`
- `interfaces/hivi-chat/adaptive_sampling.py`
- `interfaces/hivi-chat/patch_ui.py`
- `data/hivi/c1_retain_smoke.json`
- `interfaces/hivi-chat/__pycache__/`, `haie/__pycache__/`, `haie/calibration/__pycache__/`
- `scripts/p1_0/__pycache__/`, `scripts/p1_2/__pycache__/`
- `src/hivi/__pycache__/`, `memory/__pycache__/`, `memory/providers/__pycache__/`, `orchestrator/__pycache__/`, `storage/__pycache__/`
- `tests/__pycache__/`, `tests/e2e/__pycache__/`, `tests/memory/__pycache__/`
- `.pytest_cache/`, `.ruff_cache/`

`data/hivi/context_teach_recall.json` was already absent (incomplete prior run).

### Disk space recovered
**2,395,316 bytes (~2.28 MB)**

### Tests after cleanup
Offline `tests/memory` (excluding live): **20 passed in 1.01s**

### Note on user list vs Batch plan
User bullet list included `airllm-main/`; audit Batch 2 + “Do not touch Batch 2” prevailed — AirLLM tree **still present**.

---

## Appendix — inventory map

```
HIVI/
├── Adaptive-LLM-Sampling-main/     UNUSED (untracked) — Batch 1 candidate
├── Models/                         PROTECT (~130GB, gitignored)
├── data/                           PROTECT (gitignored; runtime + DBs)
├── docs/                           KEEP authoritative audits + this file
├── interfaces/hivi-chat/           KEEP UI+HAIE; drop dead shims with approval
├── scripts/                        KEEP smoke/HAIE/hindsight; P0/P1 archive TBD
├── src/hivi/
│   ├── memory|orchestrator|storage ACTIVE
│   ├── inference/
│   │   ├── llama.cpp-master/       REQUIRED SOURCE (~174MB)
│   │   ├── colibri-LLM-main/       UNCERTAIN (~18MB)
│   │   ├── airllm-main/            UNUSED (~10MB) — Batch 2 candidate
│   │   └── ollama/                 stub KEEP
│   └── [empty stubs]               KEEP for C3–C5
└── tests/memory|e2e                KEEP
```
