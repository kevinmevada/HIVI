# HIVI

<p align="center">
  <img src="interfaces/hivi-chat/static/brand/bunny.png" width="72" alt="HIVI bunny mark">
  <br>
  <strong>H I V I</strong>
</p>

**A local-first AI workbench for Windows.** HIVI runs GGUF models on
[llama.cpp](https://github.com/ggml-org/llama.cpp) or
[Colibrì](https://github.com/JustVugg/colibri) on a single 8 GB-VRAM laptop,
and adds a chat UI, persistent on-device memory, and one OpenAI-compatible
endpoint that other tools can call. Nothing leaves your machine.

> **Status: working prototype.** The chat server, memory, routing, vision,
> image generation and desktop launcher are implemented and tested. The
> multi-agent "workforce" described in the design paper is **not** built yet
> (see [Not built yet](#not-built-yet)). Development and testing so far has been on
> one machine: Windows 11, RTX 5060 Laptop (8 GB VRAM), 32 GB RAM.

## What it does today

| Piece | Where | What it does |
|-------|-------|--------------|
| Chat control server | `interfaces/hivi-chat/server.py` | Model picker; starts and stops one inference backend on `:8080`; serves the UI and an OpenAI-compatible proxy at `:7860/v1`. |
| Engine selection | `interfaces/hivi-chat/registry.py` | Scans the model folder and picks an engine per file: Colibrì for Qwen3-family text models, llama.cpp for GPT-OSS, Gemma, vision, embedding and reranker models. Falls back to llama.cpp if Colibrì cannot load a file. |
| Model roles | `configs/model_roles.yaml`, `model_router.py` | Callers set `model` to a role or alias (e.g. `main-chef`, `hivi-heavy`); HIVI waits for in-flight requests, swaps models, then answers. See [`docs/MODEL_ROUTER.md`](docs/MODEL_ROUTER.md). |
| HAIE planner | `interfaces/hivi-chat/haie/` | Classifies each message (coding, math, research, writing, chat...) with local rules and sets sampling parameters. An optional LLM classifier is off by default. |
| Prompt router | `interfaces/hivi-chat/routing/` | Suggests a better-suited installed model; the UI shows a Switch / Dismiss hint and never switches on its own. |
| Vision | registry + runtime + UI | A model with an `mmproj*.gguf` beside it loads in llama.cpp with `--mmproj`; the **Image** button is enabled only for those models. |
| Image generation | `interfaces/hivi-chat/image_gen.py` | Local text-to-image through stable-diffusion.cpp (`sd-cli`). Unloads the chat model for the render and reloads it after. |
| Retrieval services | `interfaces/hivi-chat/aux_services.py` | Embedding (`:8081`) and reranker (`:8082`) `llama-server` side services, started on demand and stopped when idle. |
| Personal memory | `personal_memory.py`, `memory_engine.py` | Local SQLite memory: remember/forget (English and Hinglish), automatic learning after replies, recall in later chats, undo, temporary chats, import/export. |
| Chat history | `chat_history.py` | Chats saved to `data/memory/chats.db`; full-text search, pin, rename, delete, `Ctrl+Shift+H`. |
| Personality | `soul.py` | Rule-based mood reading that shapes tone only (not stored), plus thumbs-up/down feedback exported as JSONL. |
| Health and backups | `gpu_doctor.py`, `backups.py` | Live GPU/RAM/CPU/disk status in the UI; daily snapshots, manual backup, retention, one-click restore. Models are not included in backups. |
| Project memory | `src/hivi/memory/` | Ledger-first retain with a retry queue, per-project banks, and correction/supersession, backed by a [Hindsight](#hindsight-optional) server. |
| ProjectRoom | `src/hivi/orchestrator/project_room.py` | A LangGraph proof of concept: the model writes `calculator.py` from a fixed spec plus recalled memory, a subprocess test gates it, and failures are fed back for up to 3 attempts. |
| Desktop launcher | `interfaces/hivi-desktop/hivi_app.py` | Native window (pywebview + WebView2) that reuses or starts the server and stops only what it started. |

### Not built yet

Agent workforce and supervisor, project/task database, and a security layer.
The role table in `model_roles.yaml` exists as configuration only; no agents run
from it. ProjectRoom executes generated code in a separate interpreter with a
timeout, which is **not a sandbox**. Plans are tracked in
[`docs/FUTURE_FEATURES.md`](docs/FUTURE_FEATURES.md).

## Requirements

- Windows 10/11 (launch scripts are PowerShell; the desktop window uses WebView2)
- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Google Chrome, if you launch with `scripts/hivi_ui.ps1`
- GGUF models of your own, placed under `Models/` (or wherever `HIVI_MODELS_ROOT` points)
- A CUDA-capable GPU is strongly recommended; the project targets 8 GB VRAM

## Setup

```powershell
uv sync                                  # or: pip install -e ".[dev]"
.\scripts\fetch_backends.ps1             # clones Colibrì and llama.cpp source (see docs/BACKENDS.md)
```

Engines are not committed to this repo. You still need a built `llama-server`
(on `PATH`, or at `HIVI_LLAMA_SERVER`) and Colibrì **2.1.0 or newer**.

### Configuration

| Variable | Default |
|----------|---------|
| `HIVI_MODELS_ROOT` | `Models/` (from `configs/storage.yaml`) |
| `HIVI_DATA_ROOT` | `data/` |
| `HIVI_LLAMA_SERVER` | `data/P1_1/runtime/llama.cpp/bin/llama-server.exe`, else `PATH` |
| `HIVI_COLI` | `src/hivi/inference/colibri-LLM-main/c/coli` |
| `HIVI_BACKEND_PORT` | `8080` |
| `HIVI_HAIE_SEMANTIC` | off; `1` enables the extra LLM intent-classification call |

The full list (router limits, RAM reserve, side-service ports and timeouts) is in
[`docs/BACKENDS.md`](docs/BACKENDS.md) and [`docs/MODEL_ROUTER.md`](docs/MODEL_ROUTER.md).

**Image generation** is optional. Paths come from `configs/image.yaml`, otherwise
HIVI searches `Models/Image`. The `configs/image.yaml` in this repo contains
absolute paths from the author's machine; edit them or delete the file before use.

### Hindsight (optional)

Project memory uses a separate Hindsight server, which is not installed by
`uv sync`:

```powershell
uv venv .venv-hindsight
uv pip install --python .venv-hindsight hindsight-api
```

## Run

```powershell
.\scripts\hivi_ui.ps1           # chat UI at http://127.0.0.1:7860 (opens Chrome)
.\scripts\start_hindsight.ps1   # optional; start after the UI so it uses the memory gateway (:8888)
uv run python interfaces/hivi-desktop/hivi_app.py   # native desktop window instead of Chrome
.\scripts\install_hivi_shortcut.ps1                 # one-time: create a desktop shortcut
```

The desktop launcher reuses a running HIVI server or starts one, and stops only
processes it started (tracked in `data/run/hivi_backend.json`). If something else
holds `:8080`, loading a model fails with that process's PID rather than killing it.
Pass `--keep-running` to leave the server up after the window closes.

### Calling HIVI from other tools

```
POST http://127.0.0.1:7860/v1/chat/completions    {"model": "hivi-heavy", ...}
POST http://127.0.0.1:7860/v1/embeddings
POST http://127.0.0.1:7860/v1/rerank
GET  http://127.0.0.1:7860/api/roles              # what each role resolves to
```

Swaps take time (a ~20 GB model reloads from disk), so batch requests by model.

## Your data

Personal memory, chat history, feedback and backups stay on this PC under
`data/` (git-ignored). Use a temporary chat for a conversation that should neither
read nor write memory. Open **Memory & settings** from the gear button in the UI.
Backups default to `data/backups/` and can be pointed at a USB drive.

## Tests

```powershell
pytest                                   # offline suite; live tests skip without :8080 / :8888
pytest tests/desktop/test_hivi_app.py    # desktop launcher only
.\scripts\run_project_room_e2e.ps1       # live ProjectRoom against real inference + Hindsight
```

## Repository layout

```
configs/                  storage, model roles, routing, image generation
docs/                     backends, model router, memory stabilization notes, roadmap
interfaces/hivi-chat/     control server, UI, HAIE, routing, memory, backups, image gen
interfaces/hivi-desktop/  native desktop launcher
scripts/                  launchers, backend fetch, Hindsight start, P1.0 model-library audit
src/hivi/                 paths, memory core, LangGraph orchestrator, SQLite ledger
tests/                    chat, desktop, memory, routing, e2e
hivi-ieee-source/         LaTeX source of the design paper
```

## Model library

`HIVI USB Model Library Final Design.docx` describes the USB shelf: models live on
an exFAT USB drive with SHA-256 checksums, are verified after copying, and always
run from the NVMe copy. `scripts/p1_0/` is the read-only audit that inventories and
hash-checks a model library.

## Design paper

[`HIVI_Vision_Paper_IEEE.pdf`](HIVI_Vision_Paper_IEEE.pdf) lays out the full vision:
a personal assistant on consumer hardware, in nine build stages. The prototype
implements the foundation and the first two stages. LaTeX sources are in
[`hivi-ieee-source/`](hivi-ieee-source/).

## License

MIT, as declared in `pyproject.toml`.
