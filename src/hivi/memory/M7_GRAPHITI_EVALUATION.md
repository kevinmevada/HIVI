# HIVI Memory — M7 Graphiti Evaluation

**Status:** Complete (decision: defer install)  
**Date:** 2026-11-06  

## Question

Do we need Graphiti now for temporal knowledge graphs, or are Hindsight + HIVI ledger/corrections enough?

## What Graphiti would add

- Explicit temporal validity windows for facts (valid_from / valid_to)
- Entity relationship graphs (Neo4j / FalkorDB / Neptune)
- Historical queries (“what was true on Nov 12?”)

## What HIVI already has after M3–M6

| Need | Coverage |
|------|----------|
| Persistent project knowledge | Hindsight banks |
| Restart survival | Hindsight pg0 + SQLite ledger/queue |
| Supersession / invalidation | `CorrectionStore` + governance filter |
| Provenance / evidence | Ledger event IDs + artifact paths |
| Agent context budgets | `context_builder` |
| Workflow resume | LangGraph checkpoints (separate from semantic memory) |

## Gaps vs Graphiti

- No first-class bi-temporal edge model
- Historical “as-of” queries are approximate (ledger timestamps + superseded flags), not graph-walkable

## Resource cost (this laptop)

- Extra graph DB (Neo4j/FalkorDB)
- Extra embeddings/LLM traffic risk against 8 GB VRAM budget
- Second semantic store to keep consistent with Hindsight

## Decision

**Do not install Graphiti in this phase.**

Revisit when all of the following are true:

1. Agents repeatedly need as-of historical answers that ledger+Hindsight cannot satisfy.
2. A local graph backend is available on NVMe without starving coding inference.
3. Sync scope is limited to decisions / requirements / ownership (not every episodic retain).

## Exit

Evidence-based decision documented; Graphiti remains optional future extension.
