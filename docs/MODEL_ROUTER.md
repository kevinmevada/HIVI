# Model router and retrieval services

HIVI exposes one OpenAI-compatible endpoint for every agent:
`http://127.0.0.1:7860/v1`. Agents choose a model by **role**; HIVI loads it.

## How a request is routed

| Request | Goes to | Swaps the chat model? |
|---------|---------|-----------------------|
| `POST /v1/chat/completions` or `/v1/completions` with `model` = a role, alias, or registry id | main backend (port 8080), after loading that model | yes, if a different model is loaded |
| the same, with any other `model` value (the chat UI does this) | main backend, whatever is loaded | no |
| `POST /v1/embeddings` | embedding side service (port 8081) | no |
| `POST /v1/rerank` (also `/v1/reranking`, `/rerank`) | reranker side service (port 8082) | no |

Roles and aliases live in `configs/model_roles.yaml`. Edits apply on the next
request; no restart. Check what each role resolves to with `GET /api/roles`.

Responses from routed requests carry `X-HIVI-Model` (registry id) and
`X-HIVI-Model-Via` (`role:main-chef->hivi-heavy`, `alias:…`, or `id`).

## Swap safety

* Swaps are serialised. A request that needs another model waits its turn
  instead of failing with 409 (the UI's manual **Load** button still gets 409).
* A swap first waits for in-flight chat and memory requests to finish
  (`HIVI_SWITCH_DRAIN_S`, default 600 s), so one agent never cuts off another
  agent's reply.
* A request waits at most `HIVI_SWITCH_WAIT_S` (default 900 s) for its model.
* The `model` field is rewritten to the loaded engine's own id before
  forwarding, because Colibrì rejects names it doesn't know.
* If a load fails, the request gets HTTP 502 with the engine's error detail.

**Batch work by model.** A 20 GB model takes a while to reload. Run every
`hivi-heavy` step, then every `hivi-review` step, then return, instead of
alternating agents one call at a time.

## Retrieval side services

Embedding and reranker GGUFs are detected by name (`embed`, `rerank`), shown as
`kind: embedding|reranker` in `/api/models?kind=all`, and kept out of the chat
dropdown and the prompt router. Each runs in its own `llama-server`:

* started on first request, stopped after `HIVI_AUX_IDLE_S` (default 600 s) idle;
* CPU only by default (`HIVI_AUX_GPU_LAYERS=0`) so the GPU stays with the chat model;
* PIDs recorded in `data/run/hivi_aux_<kind>.json`; HIVI never kills a process it
  did not start, and reports a busy port instead of taking it.

`GET /api/aux` shows their state; `POST /api/aux/stop` stops both.

**Memory budget (32 GB RAM).** Qwen3.6-35B-A3B (~20.6 GB) plus both side
services on CPU (~2.3 GB each) uses ~25 GB before context caches and Windows.
If RAM gets tight, let the side services idle out or lower `HIVI_AUX_IDLE_S`.

## Settings

| Variable | Default | Meaning |
|----------|---------|---------|
| `HIVI_SWITCH_WAIT_S` | `900` | max wait for a routed request's model |
| `HIVI_SWITCH_DRAIN_S` | `600` | max wait for in-flight requests before a routed swap |
| `HIVI_EMBED_PORT` | `8081` | embedding service port |
| `HIVI_RERANK_PORT` | `8082` | reranker service port |
| `HIVI_EMBEDDING_MODEL` | smallest embedding GGUF | registry id to use |
| `HIVI_RERANKER_MODEL` | smallest reranker GGUF | registry id to use |
| `HIVI_AUX_GPU_LAYERS` | `0` | `-ngl` for both side services |
| `HIVI_AUX_IDLE_S` | `600` | idle seconds before a side service stops |

## Smoke test

```powershell
.\scripts\hivi_ui.ps1                       # window 1
.\scripts\test_model_router.ps1             # window 2
.\scripts\test_model_router.ps1 -Review     # also loads the review model
```
