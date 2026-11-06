# Phase 1 smoke status (after GPU free + Colibrì imports)

## Done
- Built `qwen38.exe` (CUDA_DLL=1) — note: dense Qwen3.8-27B GGUF uses **qwen36** per `docs/gguf.md` (`arch=qwen35`)
- Imported Model B → `RVN-Q6_K.colibri` (link, no-mtp)
- Imported Model C → `Qwen3.8-27B-Uncensored-Q6_K.colibri` (link, no-mtp)
- Model A already had `.colibri`

## Smoke — Model A only (Config A: ctx 4096)

| Runtime | Result | Load | Wall (short “pong”) | Peak VRAM | Peak GPU util | Gen speed |
|---------|--------|------|---------------------|-----------|---------------|-----------|
| **Colibrì 2.1.0** (`qwen36`) | PASS | 17.4 s; RSS 18.5 GB | **~38 s** | CUDA trunk placed (~5.8 GB budget); verified sm_120 init | (telemetry file empty; CUDA lines in log) | trunk on GPU, rest CPU |
| **llama.cpp CUDA** | PASS | included in wall | **~125 s** | **7684 MiB** | **100%** | prompt **11.8 t/s**, gen **0.4 t/s** |

Both runtimes **actually used CUDA** (not just “CUDA installed”). llama.cpp fills VRAM and hits 100% util but decode is very slow (hybrid). Colibrì completed the same smoke faster on wall time.

## Doctor warnings
B/C dense weights (~22 GB) exceed comfortable RAM budget (~21 GB) — expect memory pressure; monitor closely in full bench.

## Next (full Phase 3+)
5 measured runs × 3 models × 2 runtimes × configs A→B, then coding suite — say the word to continue the full matrix.
