# HIVI Contextual Intelligence Acceptance

**Status:** Deferred until **after C2** (Live ProjectRoom)  
**Locked:** 2026-11-06  

## Rule

Passing memory retain/recall tests alone does **not** establish contextual intelligence.  
HIVI must understand which information is relevant, combine it with the current request, respect corrections, and avoid fabricating missing context.

## Checklist (0/30 until tested)

### 1. Understanding user context (0/5)
- [ ] Intent, not just keywords
- [ ] Follow-ups without re-explaining
- [ ] Preferences and constraints
- [ ] Instructions vs background
- [ ] Ambiguity → ask when needed

### 2. Conversation continuity (0/5)
- [ ] Same-conversation memory
- [ ] Earlier decisions reused
- [ ] References (“that project”, “previous model”)
- [ ] No re-asking answered questions
- [ ] Corrections recognized

### 3. Project context (0/5)
- [ ] Per-project context
- [ ] Objectives / requirements / progress
- [ ] Implementation decisions
- [ ] Task dependencies
- [ ] No cross-project mixing

### 4. Long-term memory (0/5)
- [ ] Cross-session facts
- [ ] Correct memory for the question
- [ ] Superseded / outdated handling
- [ ] Provenance preserved
- [ ] No invented memories on miss

### 5. Contextual reasoning (0/5)
- [ ] Combine multiple messages
- [ ] Context → appropriate response
- [ ] Contradictions recognized
- [ ] Adapt when evidence changes
- [ ] Knows when context is insufficient

### 6. Reliability (0/5)
- [ ] After application restart
- [ ] With selected local model
- [ ] Retrieval ≠ verified fact
- [ ] Hindsight failure graceful
- [ ] Isolation under concurrency

## First five practical tests

| Test | Example |
|------|---------|
| Follow-up context | Explain a model, then ask “Is it better than the previous one?” |
| Correction | Change a requirement; ask what the current requirement is |
| Cross-session recall | Restart HIVI; ask what was decided previously |
| Project isolation | Ask about Project A while in Project B |
| Contextual reasoning | Several constraints → recommend an implementation that satisfies all |

## Dependency order

1. **C2** — Live ProjectRoom (execution, persistence, recovery)  
2. **C3** — Chat → `MemoryManager` / orchestration  
3. **This checklist** — contextual intelligence acceptance (chat + memory + project room together)

Do not start this suite until C2 is green and chat is wired to core.
