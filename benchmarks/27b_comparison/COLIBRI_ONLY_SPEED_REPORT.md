# Colibrì-only 27B Speed Report

Generated: 2026-11-05T22:17:01.619061+00:00

## Settings
- Engine: Colibrì 2.1.0 (`coli run`, qwen36 for qwen35 GGUF)
- ctx=4096, ngen=64, temp=0.1
- Prompt: `Reply with exactly one word: pong`
- Measured runs per model: 5 (+1 warmup)

## Summary

| Model | Quant | Status | Load median (s) | Wall median (s) | Gen t/s median | Peak VRAM (MiB) | CUDA runs |
|-------|-------|--------|-----------------|-----------------|----------------|-----------------|-----------|
| Unsloth-Qwen3.8-27B-UD-Q5_K_M | Q5_K_M | COMPLETED | 15.1 | 38.0 | None | None | 5/5 |
| 0bserverx-RVN-Heretic-Q6_K | Q6_K | COMPLETED | 16.7 | 34.566 | None | 6294.0 | 5/5 |
| Coletti-Uncensored-Q6_K | Q6_K | COMPLETED | 16.6 | 33.301 | None | 6294.0 | 5/5 |

## Notes
- Each run is a fresh process (includes model load).
- Dense trunk is placed on CUDA when free VRAM allows; remaining work stays on CPU/RAM.
- JSON: `colibri_only_results.json` · CSV: `colibri_only_raw.csv`
