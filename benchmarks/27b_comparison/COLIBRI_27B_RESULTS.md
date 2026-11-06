# Colibrì 27B speed results (Config A)

Scope: speed only. Engine: Colibrì **2.1.0** / **qwen36** (GGUF arch `qwen35`).  
Prompt: `Reply with exactly one word: pong` · ctx **4096** · ngen **64** · temp **0.1**

## Model A — Unsloth Qwen3.8-27B UD-Q5_K_M (COMPLETE)

| Metric | Result |
|--------|--------|
| Runs | warmup + **5/5 measured** — all OK |
| Load time | **median 15.1 s** (range 14.8–15.5 s measured) |
| RSS after load | **18.49 GB** |
| Approx. wall / run | **~40–45 s** (from log timestamps; first-pass wall field not saved) |
| CUDA | **Verified** — RTX 5060 Laptop, **sm_120** |
| GPU placement | Dense trunk → CUDA (~**5.8 GB** allowance) |
| On GPU | lm_head **0.97 GB** + DeltaNet proj **1.59 GB** + 127 trunk mats **3.19 GB** ≈ **~5.75 GB** |
| Note | Log says “experts on CPU”; this GGUF is **dense** (no MoE experts) |

Smoke (earlier same setup): wall **38.2 s**, load **17.4 s**.

## Model B — RVN Heretic Q6_K

- `.colibri` imported (`RVN-Q6_K.colibri`)
- Doctor: OK with RAM pressure warning (~22 GB dense vs ~21 GB budget)
- **Speed matrix not run yet** (bench still on llama.cpp path)

## Model C — Coletti Uncensored Q6_K

- `.colibri` imported (`Qwen3.8-27B-Uncensored-Q6_K.colibri`)
- Same doctor warning pattern
- **Speed matrix not run yet**

## Takeaway (Colibrì only, so far)

On this 8 GB laptop, **Colibrì + Model A** is usable: CUDA trunk offload confirmed, ~15 s load, ~40 s short reply. B/C should behave similarly but need measured runs (heavier Q6_K / more RAM risk).
