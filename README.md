# HIVI

Portable AI model library and inference runtime.

## Layout

```
HIVI/
├── .vscode/
├── configs/
├── src/
│   └── hivi/
│       ├── api/
│       ├── core/
│       ├── orchestrator/
│       ├── agents/
│       ├── router/
│       ├── models/
│       ├── inference/
│       │   ├── llama.cpp-master/   # vendored
│       │   ├── airllm-main/        # vendored
│       │   ├── colibri-LLM-main/   # vendored
│       │   └── ollama/             # thin client → local PC install
│       ├── tools/
│       ├── memory/
│       ├── retrieval/
│       ├── storage/
│       ├── hardware/
│       ├── security/
│       ├── evaluation/
│       └── monitoring/
├── interfaces/
├── experiments/
├── benchmarks/
├── tests/
├── scripts/
├── docs/
├── pyproject.toml
├── README.md
└── .gitignore
```

## Inference backends

| Backend | In repo? | Notes |
|---------|----------|--------|
| llama.cpp | yes (`llama.cpp-master/`) | default |
| AirLLM | yes (`airllm-main/`) | |
| Colibri | yes (`colibri-LLM-main/`) | |
| Ollama | no | uses the Ollama app already on the PC via `http://127.0.0.1:11434` |

## Setup

```bash
pip install -e ".[dev]"
```

## Design notes

See `HIVI USB Model Library Final Design.docx` for the USB shelf / checksum workflow.
Models live on USB (or disk); inference runs locally via backends under `src/hivi/inference/`.
