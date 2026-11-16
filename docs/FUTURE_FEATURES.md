# Future Features

This roadmap reflects the current HIVI architecture and keeps unfinished work
visible without treating plans as implemented behavior.

## Next stages

### Agent workforce

Add specialized agents and a supervisor that can delegate work, coordinate
results, and report failures clearly.

### Project and task database

Persist projects, tasks, execution history, and artifacts so work can resume
across sessions and remain inspectable.

### Security layer

Add explicit permissions, safe tool boundaries, input validation, and audit
events before enabling broader automated execution.

### Memory and ProjectRoom hardening

Continue improving retention governance, correction and supersession flows,
retry behavior, and end-to-end ProjectRoom reliability.

### Model and backend operations

Improve model discovery, health diagnostics, download verification, backend
selection, and performance reporting while keeping engines outside the
repository.

## Done in stage 2 (She knows you)

- **Personal memory (MEM-1 to MEM-3)**: `interfaces/hivi-chat/personal_memory.py`
  and `memory_engine.py`. SQLite at `data/memory/memory.db`; "remember / forget"
  in English and Hinglish; profile block and per-message recall in the system
  prompt (embedding service when running, word overlap otherwise); automatic
  learning after replies on an idle GPU with de-duplicate / replace and an Undo
  chip; temporary chat; export / import; the Memory screen (gear in the header).
- **Mood and character (SOUL-1, SOUL-2)**: `soul.py`. Rule-based mood reading
  that shapes tone only and is never stored; loyalty rules and the composed /
  dry-wit character in every HIVI chat; 👍 / 👎 with "say it like this instead",
  exported as JSONL for later preference tuning.
- **Local snapshots (BAK-1)**: `backups.py`. Daily snapshot of memory, feedback,
  settings and configs to `data/backups` or a folder you choose (USB drive);
  "Back up now", retention, one-click restore with a safety snapshot first.

- **Chat history (MEM-4)**: `chat_history.py`. Every chat is saved to
  `data/memory/chats.db` after each reply; sidebar (Ctrl+Shift+H) with
  day groups, full-text search, pin, rename, delete; included in snapshots.

Still ahead in this area: MEM-5 agents reading
preferences, SOUL-3 psychology library, SOUL-4 voice tone, SOUL-5 character
LoRA, BAK-2 encrypted / cloud backups, BAK-3 restore-points screen.

## Current foundation

The chat control server, HAIE planner, A1 router, vision path, memory gateway,
ProjectRoom, desktop launcher, and USB model-library workflow are active
foundations. See [`docs/BACKENDS.md`](BACKENDS.md) for backend setup and
[`README.md`](../README.md) for the current run and test commands.
