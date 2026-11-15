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

## Current foundation

The chat control server, HAIE planner, A1 router, vision path, memory gateway,
ProjectRoom, desktop launcher, and USB model-library workflow are active
foundations. See [`docs/BACKENDS.md`](BACKENDS.md) for backend setup and
[`README.md`](../README.md) for the current run and test commands.
