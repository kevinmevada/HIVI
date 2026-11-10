# HIVI

Local AI workbench for an 8 GB-VRAM laptop: a portable model library, a chat
UI that runs GGUF models on Colibrì or llama.cpp, and a project memory layer
(Hindsight + SQLite ledger + LangGraph) that survives restarts.

Honest status lives in `docs/ARCHITECTURE_COMPLETENESS_AUDIT.md` and the
milestone plan in `docs/CORE_INTEGRATION_SPINE.md`.

## What runs today

| Piece | Where | What it does |
|-------|-------|--------------|
| Chat control server | `interfaces/hivi-chat/server.py` | Model picker, starts/stops one backend on `:8080`, OpenAI-compatible proxy on `:7860/v1` |
| HAIE planner | `interfaces/hivi-chat/haie/` | Per-message intent → sampling params (fast regex → heuristic → LLM classify) |
| A1 router | `interfaces/hivi-chat/routing/` | Suggests a better-suited model for a prompt (UI shows a "Switch" hint) |
| Vision | registry + runtime + UI | Models with an `mmproj*.gguf` beside them load in llama.cpp with `--mmproj`; the UI's **Image** button sends pictures |
| Memory gateway | `:7860/memory/v1` | Hindsight's LLM endpoint; waits out model switches instead of failing |
| Memory core | `src/hivi/memory/` | Ledger-first retain, retry queue, per-project banks, corrections/supersession |
| ProjectRoom | `src/hivi/orchestrator/project_room.py` | LLM writes `calculator.py` from spec + recalled memory → subprocess tests → retry with failure feedback |

Not built yet: agent workforce, project/task database, security layer, supervisor.

## Layout

```
HIVI/
├── configs/                 # storage.yaml sets models_root / data_root
├── docs/                    # audit, milestones, BACKENDS.md
├── interfaces/hivi-chat/    # control server, UI, HAIE, router
├── scripts/                 # bootstrap, benchmarks, fetch_backends.ps1, launchers
├── src/hivi/
│   ├── paths.py             # every filesystem location (env-overridable)
│   ├── memory/              # Hindsight provider, governance, queue
│   ├── orchestrator/        # LangGraph graphs, ProjectRoom
│   ├── storage/             # SQLite ledger
│   └── inference/           # engines are fetched here, not committed
└── tests/
```

## Setup

```powershell
uv sync                                  # or: pip install -e ".[dev]"
.\scripts\fetch_backends.ps1             # Colibrì + llama.cpp source (see docs/BACKENDS.md)
```

Point HIVI at your binaries and models if they are not in the defaults:

| Variable | Default |
|----------|---------|
| `HIVI_MODELS_ROOT` | `Models/` (from `configs/storage.yaml`) |
| `HIVI_LLAMA_SERVER` | `data/P1_1/runtime/llama.cpp/bin/llama-server.exe`, else PATH |
| `HIVI_COLI` | `src/hivi/inference/colibri-LLM-main/c/coli` |
| `HIVI_BACKEND_PORT` | `8080` |

Full list: `docs/BACKENDS.md`. Nothing in the code refers to a specific user's
home folder.

## Run

```powershell
.\scripts\hivi_ui.ps1        # chat UI on http://127.0.0.1:7860
.\scripts\start_hindsight.ps1  # after the UI is up, so it uses the memory gateway
```

HIVI only stops processes it started (tracked in `data/run/hivi_backend.json`).
If something else already holds `:8080`, loading a model fails with the PID
instead of killing it.

## Tests

```powershell
pytest                       # offline suite (live tests skip without :8080/:8888)
.\scripts\run_project_room_e2e.ps1   # live ProjectRoom against real inference + Hindsight
```

## Model library

`HIVI USB Model Library Final Design.docx` describes the USB shelf and SHA-256
workflow: models are stored on USB, verified, and run from the NVMe.
