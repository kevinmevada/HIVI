# HIVI

<p align="center">
  <img src="interfaces/hivi-chat/static/brand/bunny.png" width="72" alt="HIVI bunny mark">
  <br>
  <strong>H I V I</strong>
  <br>
  <sub>Local intelligence · memory · execution</sub>
</p>

Local AI workbench for an 8 GB-VRAM laptop: a portable model library, a chat
UI that runs GGUF models on Colibrì or llama.cpp, and a project memory layer
(Hindsight + SQLite ledger + LangGraph) that survives restarts.

Honest status lives in `docs/ARCHITECTURE_COMPLETENESS_AUDIT.md` and the
milestone plan in `docs/CORE_INTEGRATION_SPINE.md`.

## Brand

HIVI's interface uses a clean system sans-serif stack: **SF Pro Display /
Inter / Segoe UI Variable Display / Segoe UI** for headings and the matching
text stack for body copy. The bunny is HIVI's primary symbol; the wordmark is
set in a bold display face with a warm-to-violet spectrum wave and soft glow.
Use the existing assets in
[`interfaces/hivi-chat/static/brand/`](interfaces/hivi-chat/static/brand)
when extending the UI so the symbol, typography, and color treatment stay
consistent.

## What runs today

| Piece | Where | What it does |
|-------|-------|--------------|
| Chat control server | `interfaces/hivi-chat/server.py` | Model picker, starts/stops one backend on `:8080`, OpenAI-compatible proxy on `:7860/v1` |
| HAIE planner | `interfaces/hivi-chat/haie/` | Per-message intent → sampling params using fast rules by default; optional LLM classification for ambiguous requests |
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
├── interfaces/hivi-desktop/ # single-window desktop launcher
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
| `HIVI_HAIE_SEMANTIC` | Off; set to `1` to enable the extra LLM intent-classification call |
| `HIVI_SCAN_TTL_S` | `10` seconds for the model-library scan cache |

Full list: `docs/BACKENDS.md`. Nothing in the code refers to a specific user's
home folder.

### HAIE behavior

HAIE classifies each message locally with rule-based matching by default, so a
second model request is not required before the actual reply. It recognizes
coding, architecture, math, research, explanation, writing, brainstorming,
greetings and chat, and carries short follow-ups such as "make it async" from
the previous user request. Set `HIVI_HAIE_SEMANTIC=1` when you prefer the
optional semantic LLM classifier for ambiguous prompts.

## Run

```powershell
.\scripts\hivi_ui.ps1        # chat UI on http://127.0.0.1:7860
.\scripts\start_hindsight.ps1  # after the UI is up, so it uses the memory gateway
uv run python interfaces/hivi-desktop/hivi_app.py  # desktop window / app-style browser fallback
```

The desktop launcher reuses a running HIVI server, or starts and stops one for
the lifetime of its window. On Windows, use the **HIVI** shortcut on the
desktop after creating it once:

```powershell
.\scripts\install_hivi_shortcut.ps1
```

Pin the **HIVI** shortcut itself to the taskbar. Do not pin the running
Python window; Windows uses the Python host icon for script-launched windows.

The desktop app starts the server and window in parallel, shows startup
progress, and enforces a single running HIVI window. Closing the window stops
only processes HIVI started; an unrelated process already using `:8080` is
left untouched. To keep the server running after closing the desktop window,
launch the app with `--keep-running`.

HIVI only stops processes it started (tracked in `data/run/hivi_backend.json`).
If something else already holds `:8080`, loading a model fails with the PID
instead of killing it.

## Tests

```powershell
pytest                       # offline suite (live tests skip without :8080/:8888)
pytest tests/desktop/test_hivi_app.py  # desktop launcher tests only
.\scripts\run_project_room_e2e.ps1   # live ProjectRoom against real inference + Hindsight
```

## Model library

`HIVI USB Model Library Final Design.docx` describes the USB shelf and SHA-256
workflow: models are stored on USB, verified, and run from the NVMe.
