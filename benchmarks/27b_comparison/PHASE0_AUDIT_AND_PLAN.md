# Phase 0 — Environment Audit & Proposed Execution Plan

**Status:** COMPLETE (audit only — full benchmark not started)  
**Artifacts:** `environment_audit.json`, `model_inventory.json`  
**Date:** 2026-11-05

---

## Executive blockers (must resolve before Phase 3+)

1. **GPU is occupied** (~6.3 / 8.1 GiB used; `qwen36.exe` / Ollama / llama activity observed). Fair runs need a clean GPU. Will not kill user apps without permission.
2. **Colibrì `qwen38.exe` is missing.** Model A import maps to `colibri_gguf.arch = qwen35` / `model_type = qwen3_5_text`. Engine family is **qwen38**, not qwen36. Present: `qwen36.exe` + `coli_cuda.dll`. **Do not force qwen36 onto these 27B models.** Building `qwen38.exe` requires your approval (`make qwen38` / `CUDA_DLL=1` as applicable).
3. **Models B & C have no `.colibri` import yet.** Only Model A has `Qwen3.8-27B-UD-Q5_K_M.colibri`. Importing creates sidecar dirs (no GGUF overwrite) — needs explicit OK via `coli import`.
4. **Hardware reality:** 18–21 GB GGUFs on **8 GB VRAM** ⇒ llama.cpp will be **hybrid GPU+CPU**. Colibrì (if qwen38 works) streams/tiers; still RAM-heavy. Config C (16k) may be OUT OF MEMORY.

---

## Hardware (verified)

| Item | Measured |
|------|----------|
| CPU | Intel Core Ultra 9 275HX — **24 cores / 24 logical** |
| RAM | **31.46 GB** total (free fluctuated; was ~2 GB under load, ~21 GB after unload) |
| GPU | NVIDIA GeForce RTX 5060 Laptop — **8151 MiB** |
| Driver | **616.92** |
| CUDA Toolkit | **13.4** (nvcc V13.4.59) |
| Arch target | **sm_120** |

---

## Software (verified)

| Component | Path / version |
|-----------|----------------|
| Python | System 3.11.9; uv 3.13.9 |
| llama-cli | `HIVI\data\P1_1\runtime\llama.cpp\bin\llama-cli.exe` — **0.8.0-dev**, MSVC, **ggml-cuda.dll present** |
| llama-server | Present beside cli |
| Flash Attn / KV | Supported flags: `-fa`, `-ctk`/`-ctv`, `-ngl`, `-b`/`-ub` |
| Colibrì | **`C:\Users\mevad\Downloads\colibri-main\colibri-main`** — **v2.1.0** |
| Colibrì CUDA | `c\coli_cuda.dll` present; `c\qwen36.exe` present; **`c\qwen38.exe` ABSENT** |

CUDA DLL presence ≠ confirmed GPU decode for qwen38. That must be measured with `nvidia-smi` telemetry during generation.

---

## Model inventory (all three present)

| ID | Publisher | File | Size | Quant | Colibrì sidecar |
|----|-----------|------|------|-------|-----------------|
| **A** | Unsloth | `...\unsloth\qwen3.8-27b-q5_k_m\Qwen3.8-27B-UD-Q5_K_M.gguf` | **18.41 GB** | UD-Q5_K_M | **YES** (`.colibri`) |
| **B** | 0bserverx | `...\0bserverx\...\RVN-Q6_K.gguf` | **20.57 GB** | Q6_K | **NO** |
| **C** | jonathan-coletti | `...\jonathan-coletti\...\Qwen3.8-27B-Uncensored-Q6_K.gguf` | **20.89 GB** | Q6_K | **NO** |

### Model A Colibrì metadata (from import `config.json`)

- `model_type`: `qwen3_5_text`
- `architectures`: `Qwen3_5ForCausalLM`
- `colibri_gguf.arch`: **`qwen35`**
- Layers: **64**, hidden: **5120**, vocab: **248320**, max_pos: **262144**
- Hybrid layer_types: linear_attention + full_attention (interval 4)
- GGUF linked (mode `link`) — original file not duplicated/overwritten

Models B/C: assume same family only after GGUF metadata probe in Phase 1; templates/tokenizer may differ (Heretic/Uncensored).

---

## Runtime compatibility (preliminary)

| Model | llama.cpp CUDA | Colibrì |
|-------|----------------|---------|
| A Q5_K_M | **Likely YES** (GGUF + ggml-cuda build) — hybrid expected | **BLOCKED until `qwen38.exe` built**; import exists |
| B RVN Q6_K | **Likely YES** pending load test | Needs `coli import` + qwen38 |
| C Uncensored Q6_K | **Likely YES** pending load test | Needs `coli import` + qwen38 |

---

## Proposed configurations (Phase 2)

After GPU is free and engines ready:

| Config | ctx | llama.cpp intent | Colibrì intent |
|--------|-----|------------------|----------------|
| **A Conservative** | 4096 | `-ngl 99`, `-ctk q8_0 -ctv q8_0`, `-fa on`, `-b 512 -ub 128` | `--ctx 4096 --policy balanced --gpu` / auto-tier |
| **B Standard** | 8192 | same + max safe layers | `--ctx 8192 --policy quality` if RAM allows |
| **C Quality** | 16384 | only if peak RAM/VRAM safe | only if doctor/plan says fit |

Record **effective** settings from logs, not just requested flags. Abort Config C on OOM risk.

---

## Proposed execution order (after your approval)

1. Free GPU (you close Ollama/qwen36/llama, or authorize kill of named PIDs).
2. **Approve build** of `qwen38.exe` (+ CUDA DLL link if needed) in Downloads Colibrì tree.
3. **Approve** `coli import` for Models B and C (sidecar only).
4. Phase 1: `coli doctor/info` + llama load smoke per model; verify CUDA init via telemetry.
5. Phase 3: perf matrix (load, warm-up, 5 measured runs) — one model at a time.
6. Phase 4–5: coding suite + quality scores.
7. Phase 7: one-variable optimizations.
8. Phase 8: deliverables under `benchmarks\27b_comparison\`.

**Not started:** Phases 1–8 measurements (no invented numbers).

---

## Approvals needed to proceed

Reply with OK for any/all of:

1. Close/kill current GPU holders (`qwen36` / Ollama llama-server / leftover llama-cli) for clean baseline  
2. Build `qwen38.exe` (and CUDA-enabled variant if required)  
3. Run `coli import` on Models B and C (create `.colibri` sidecars; no GGUF modify)  
4. Start Phase 1 compatibility loads + Phase 3 speed matrix (Config A first)

---

## Final note

On this laptop, **no 27B GGUF will be “full GPU resident.”** The benchmark will measure which **engine+quant** is least bad for coding work under hybrid/tiered memory — that is still a valid, evidence-based result.
