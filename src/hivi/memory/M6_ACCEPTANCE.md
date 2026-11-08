# HIVI Memory — M6 Acceptance

**Status:** Complete  
**Date:** 2026-11-06  

## Scenario: Calculator project

| # | Criterion | Result |
|---|-----------|--------|
| 1 | Complete project memory path | Pass — decisions + failures retained |
| 2 | Stop and restart HIVI | Pass — new `MemoryManager` + same SQLite/provider |
| 3 | Resume with context | Pass — LangGraph handover turn |
| 4 | Change requirements | Pass — supersede PySide6 → web UI |
| 5 | Old decisions not current | Pass — context omits superseded |
| 6 | Remember previous failures | Pass — division-by-zero recalled |
| 7 | Bank isolation | Pass — project 002 cannot see calculator facts |
| 8 | Queue recovery | Pass — pending survive reopen then drain |

## Exit

`tests/memory/test_m6_calculator_e2e.py` passes.

Note: Uses `InMemoryProvider` for deterministic CI-speed validation. Live Hindsight path remains covered by M1/M2 live tests.
