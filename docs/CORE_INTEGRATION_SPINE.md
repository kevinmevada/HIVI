# HIVI Core Integration Spine

**Status:** Active  
**Locked:** 2026-11-06  
**Source:** Architecture Completeness Audit + roadmap correction (M3–M7 = memory only)

## Non-goals (now)

- No new memory frameworks / Graphiti
- No second orchestration framework
- No full 42-role workforce

## Milestones

| ID | Deliverable | Measurable exit |
|----|-------------|-----------------|
| **C1** | Stabilize live Hindsight retain | **Done** — see `docs/C1_LIVE_MEMORY_STABILIZATION.md` (~109s smoke retain+recall on bank 099) |
| **C2** | Green live ProjectRoom | Live E2E: execute → retain → restart → recall evidence |
| **C3** | Connect HIVI Chat → core | Chat uses `MemoryManager` (+ orchestration); not a parallel runtime |
| **C4** | Project + Task entities | SQLite project identity, lifecycle, tasks, ownership, deps |
| **C5** | First dynamic workforce | Small agent registry + few roles (not all roles) |

## First measurable result (end of C3)

User enters a task in HIVI Chat → retrieve project memory → LangGraph execute → record result → survive restart → retrieve previous work.

## After C3

**Contextual intelligence acceptance** — see `docs/CONTEXTUAL_INTELLIGENCE_ACCEPTANCE.md` (30-item checklist + 5 practical tests). Memory green ≠ contextual intelligence.
