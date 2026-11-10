"""Build HAIE V1.2 P1 labeled evaluation set (>=200). Does not touch production routing."""
from __future__ import annotations

import os

import json
from pathlib import Path

ROOT = Path(os.environ.get("HIVI_ROOT") or Path(__file__).resolve().parents[1])
OUT = ROOT / "data" / "haie" / "p1" / "labeled_set.jsonl"


def row(
    id_: str,
    split: str,
    prompt: str,
    gold_intents: list[str],
    gold_path: str,
    *,
    complexity: int = 2,
    deliverable: str = "reply",
    reasoning: str = "low",
    length: str = "medium",
    expected_tokens: int | None = None,
) -> dict:
    return {
        "id": id_,
        "split": split,
        "prompt": prompt,
        "gold_intents": gold_intents,
        "gold_path": gold_path,
        "gold_complexity": complexity,
        "gold_deliverable": deliverable,
        "gold_reasoning": reasoning,
        "gold_length": length,
        "gold_expected_tokens": expected_tokens,
    }


def build() -> list[dict]:
    rows: list[dict] = []

    # --- simple (greetings, tiny chat, arithmetic) ---
    greetings = ["Hey", "Hi", "Hello", "Yo", "Thanks", "Thank you", "Ok", "Okay", "hi!", "hello."]
    for i, g in enumerate(greetings):
        rows.append(row(f"simple_greet_{i:03d}", "simple", g, ["greeting"], "fast", complexity=1, length="short", expected_tokens=128))

    chats = [
        "How's it going?",
        "Good morning",
        "Nice to meet you",
        "What's up",
        "See you later",
        "Have a good day",
        "All good",
        "Sounds fine",
        "Cool",
        "Got it",
    ]
    for i, c in enumerate(chats):
        rows.append(row(f"simple_chat_{i:03d}", "simple", c, ["chat"], "fast", complexity=1, length="short"))

    maths = [
        ("What is 2+2?", 256),
        ("What is 9*9?", 256),
        ("Calculate 15-7", 256),
        ("Compute 100/4", 256),
        ("What is 12 x 3?", 256),
        ("What is 8+1?", 256),
        ("calculate 6*7", 256),
        ("What is 50-20?", 256),
        ("Compute 3+5", 256),
        ("What is 11*2?", 256),
        ("What is 4/2?", 256),
        ("calculate 19+1", 256),
        ("What is 7-3?", 256),
        ("Compute 9+10", 256),
        ("What is 5*5?", 256),
        ("What is 21+4?", 256),
        ("calculate 8/2", 256),
        ("What is 14-6?", 256),
        ("Compute 2*8", 256),
        ("What is 1+1?", 256),
    ]
    for i, (p, tok) in enumerate(maths):
        rows.append(row(f"simple_math_{i:03d}", "simple", p, ["math"], "fast", complexity=1, length="short", expected_tokens=tok, deliverable="solution", reasoning="none"))

    # --- coding (simple → semantic for non-hello-world) ---
    simple_code = [
        "Write Hello World in python",
        "Write a hello world in python",
        "Create hello world python",
        "Write Hello World",
        "write a simple hello world",
    ]
    for i, p in enumerate(simple_code):
        rows.append(row(f"coding_simple_{i:03d}", "coding", p, ["coding"], "fast", complexity=1, deliverable="code", length="short", expected_tokens=512))

    coding_sem = [
        ("Write a Python function to reverse a linked list", ["coding"], 3),
        ("Implement binary search in TypeScript", ["coding"], 3),
        ("Fix this null pointer in my Java service", ["coding"], 3),
        ("Write a regex to validate emails", ["coding"], 2),
        ("Create a REST endpoint for user login", ["coding"], 3),
        ("Implement LRU cache in Python", ["coding"], 3),
        ("Write unit tests for a payment parser", ["coding"], 3),
        ("Refactor this nested loop for readability", ["coding"], 3),
        ("Add retry with exponential backoff in Go", ["coding"], 3),
        ("Write a SQL query for top-N customers", ["coding"], 2),
        ("Implement merge sort in C++", ["coding"], 3),
        ("Create a Dockerfile for a FastAPI app", ["coding"], 2),
        ("Write a bash script to rotate logs", ["coding"], 2),
        ("Implement rate limiting middleware", ["coding"], 3),
        ("Parse JSONL into a pandas DataFrame", ["coding"], 2),
        ("Write a CLI that greps files recursively", ["coding"], 3),
        ("Implement websocket heartbeat in Node", ["coding"], 3),
        ("Add type hints to this Python module", ["coding"], 2),
        ("Write a migration to add a users table", ["coding"], 2),
        ("Implement JWT auth middleware", ["coding"], 3),
        ("Create a React hook for debounced input", ["coding"], 2),
        ("Write a Python dataclass for Order", ["coding"], 2),
        ("Implement circular buffer in Rust", ["coding"], 3),
        ("Fix race condition in this mutex code", ["coding"], 4),
        ("Write a generator that streams CSV rows", ["coding"], 2),
        ("Implement Dijkstra on an adjacency list", ["coding"], 3),
        ("Create a Makefile for a C project", ["coding"], 2),
        ("Write idempotent upsert SQL", ["coding"], 2),
        ("Implement bloom filter basics", ["coding"], 3),
        ("Add OpenTelemetry spans to this handler", ["coding"], 3),
        ("Write a pytest fixture for a temp DB", ["coding"], 2),
        ("Implement paginated API client", ["coding"], 3),
        ("Create a worker pool in Python", ["coding"], 3),
        ("Write a schema validator for events", ["coding"], 3),
        ("Implement soft-delete for ORM models", ["coding"], 2),
    ]
    for i, (p, intents, cx) in enumerate(coding_sem):
        rows.append(row(f"coding_sem_{i:03d}", "coding", p, intents, "semantic", complexity=cx, deliverable="code", reasoning="medium" if cx >= 3 else "low", length="medium"))

    # --- reasoning / explanation / math proofs / research-ish ---
    reasoning = [
        ("Explain how TCP congestion control works", ["explanation"], 3, "explanation"),
        ("Why does gradient vanishing happen in deep nets?", ["explanation"], 3, "explanation"),
        ("Explain CAP theorem with an example", ["explanation"], 3, "explanation"),
        ("How does a Bloom filter trade accuracy for space?", ["explanation"], 3, "explanation"),
        ("Explain async vs threads for I/O bound work", ["explanation"], 3, "explanation"),
        ("Walk through how a hash map handles collisions", ["explanation"], 3, "explanation"),
        ("Explain MVCC in PostgreSQL", ["explanation"], 3, "explanation"),
        ("Why is tail latency hard to control?", ["explanation"], 3, "explanation"),
        ("Explain speculative decoding for LLMs", ["explanation"], 3, "explanation"),
        ("How does backpressure work in streaming systems?", ["explanation"], 3, "explanation"),
        ("Prove that the square root of 2 is irrational", ["math"], 4, "proof"),
        ("Derive the quadratic formula step by step", ["math"], 3, "derivation"),
        ("Explain Bayes theorem with a medical test example", ["math", "explanation"], 3, "explanation"),
        ("What is the time complexity of heapify?", ["math", "explanation"], 2, "explanation"),
        ("Explain Big-O of quicksort average vs worst", ["math", "explanation"], 3, "explanation"),
        ("Research local LLM quantization tradeoffs for 8GB VRAM", ["research"], 4, "report"),
        ("Summarize differences between RAG and fine-tuning", ["research", "explanation"], 3, "comparison"),
        ("Compare Redis and Memcached for session cache", ["research", "explanation"], 3, "comparison"),
        ("Survey approaches to LLM tool calling", ["research"], 4, "survey"),
        ("Explain KV cache and why it grows with context", ["explanation"], 3, "explanation"),
        ("How do MoE routers pick experts?", ["explanation", "research"], 3, "explanation"),
        ("Explain flash attention at a high level", ["explanation"], 3, "explanation"),
        ("Why does MXFP4 help MoE models on consumer GPUs?", ["explanation", "research"], 3, "explanation"),
        ("Explain deadlock vs livelock with examples", ["explanation"], 3, "explanation"),
        ("How does a matching engine keep price-time priority?", ["explanation", "architecture"], 4, "explanation"),
        ("Explain CRDTs for collaborative editing", ["explanation"], 3, "explanation"),
        ("What breaks first when context length grows?", ["explanation", "research"], 3, "explanation"),
        ("Explain temperature vs top-p sampling", ["explanation"], 2, "explanation"),
        ("How does speculative decoding interact with MoE?", ["explanation", "research"], 4, "explanation"),
        ("Explain why Haar wavelet is used in compression", ["explanation", "math"], 3, "explanation"),
        ("Derive softmax and discuss numerical stability", ["math", "explanation"], 3, "derivation"),
        ("Explain Paxos roles briefly", ["explanation", "architecture"], 4, "explanation"),
        ("What is the difference between OLTP and OLAP?", ["explanation"], 2, "explanation"),
        ("Explain vector clocks for causality", ["explanation"], 3, "explanation"),
        ("How does rope positional encoding work?", ["explanation"], 3, "explanation"),
    ]
    for i, (p, intents, cx, deliv) in enumerate(reasoning):
        rows.append(row(f"reasoning_{i:03d}", "reasoning", p, intents, "semantic", complexity=cx, deliverable=deliv, reasoning="high" if cx >= 4 else "medium", length="long" if cx >= 4 else "medium"))

    # --- multi-intent ---
    multi = [
        ("explain and optimize this python function for sorting", ["explanation", "coding"], 3),
        ("explain and also implement a cache and compare two approaches", ["explanation", "coding", "research"], 4),
        ("Design a trading engine and implement the matching loop", ["architecture", "coding"], 5),
        ("Compare MoE and Dense models for local 8GB inference", ["explanation", "research"], 3),
        ("Debug this complex concurrency problem in a thread pool", ["coding", "explanation"], 4),
        ("Explain the bug and then write a failing test", ["explanation", "coding"], 3),
        ("Design an API and generate OpenAPI stubs", ["architecture", "coding"], 4),
        ("Compare two sorting algorithms and implement the winner", ["research", "coding", "explanation"], 3),
        ("Explain Redis eviction and write a demo client", ["explanation", "coding"], 3),
        ("Brainstorm auth options then recommend one architecture", ["brainstorm", "architecture"], 3),
        ("Research vector DBs and sketch an ingestion pipeline", ["research", "architecture"], 4),
        ("Explain GC pauses and tune this JVM config", ["explanation", "coding"], 3),
        ("Compare Kafka vs NATS then outline a migration plan", ["research", "architecture"], 4),
        ("Implement pagination and explain cursor vs offset", ["coding", "explanation"], 3),
        ("Design a rate limiter and prove its bounds", ["architecture", "math"], 4),
        ("Write a profiler script and explain the hotspots", ["coding", "explanation"], 3),
        ("Compare SQLite and DuckDB for analytics on a laptop", ["research", "explanation"], 3),
        ("Explain TLS handshake and implement a health probe", ["explanation", "coding"], 3),
        ("Design offline sync and brainstorm conflict policies", ["architecture", "brainstorm"], 4),
        ("Optimize this SQL and explain the plan differences", ["coding", "explanation"], 3),
        ("Compare dense retrieval and hybrid search for docs", ["research", "explanation"], 3),
        ("Implement a circuit breaker and document failure modes", ["coding", "explanation"], 3),
        ("Design a feature store and compare online vs offline paths", ["architecture", "research"], 4),
        ("Explain deadlock detection and fix this lock order", ["explanation", "coding"], 4),
        ("Write a migration plan and implement the first step", ["architecture", "coding"], 3),
        ("Compare GGML quant formats and recommend one for 8GB", ["research", "explanation"], 3),
        ("Explain backtesting pitfalls and sketch a research notebook", ["explanation", "research"], 3),
        ("Design a chat UI and implement streaming tokens", ["architecture", "coding"], 3),
        ("Compare prompt caching strategies and implement one", ["research", "coding"], 3),
        ("Explain MoE load balancing and propose an architecture", ["explanation", "architecture"], 4),
        ("Implement eval harness and compare two models", ["coding", "research"], 3),
        ("Brainstorm product names then write a short pitch", ["brainstorm", "explanation"], 2),
        ("Design observability and add metrics to this service", ["architecture", "coding"], 3),
        ("Explain quantization error and show a tiny experiment", ["explanation", "math", "coding"], 3),
        ("Compare Colibrì and llama.cpp for this workload", ["research", "explanation"], 3),
    ]
    for i, (p, intents, cx) in enumerate(multi):
        rows.append(row(f"multi_{i:03d}", "multi_intent", p, intents, "semantic", complexity=cx, deliverable="mixed", reasoning="high" if cx >= 4 else "medium", length="long" if cx >= 4 else "medium"))

    # --- ambiguous ---
    ambiguous = [
        ("Can you help with this?", ["chat"], 2),
        ("Make it better", ["chat", "brainstorm"], 2),
        ("Fix it", ["coding", "chat"], 2),
        ("What do you think?", ["chat", "brainstorm"], 2),
        ("Do the usual thing", ["chat"], 2),
        ("Handle the edge cases", ["coding", "chat"], 2),
        ("Improve performance", ["coding", "explanation"], 3),
        ("Look into MoE stuff", ["research", "chat"], 2),
        ("Something about trading systems", ["architecture", "chat"], 2),
        ("Help me decide", ["brainstorm", "chat"], 2),
        ("Is this good enough?", ["chat", "explanation"], 2),
        ("Continue", ["chat"], 1),
        ("Same but for production", ["architecture", "chat"], 3),
        ("Make it production ready", ["architecture", "coding"], 3),
        ("Not sure if code or design", ["coding", "architecture"], 3),
        ("Maybe explain or just show code", ["explanation", "coding"], 2),
        ("Whatever works on 8GB", ["research", "chat"], 2),
        ("Clean this up", ["coding", "chat"], 2),
        ("Thoughts on the approach?", ["brainstorm", "explanation"], 2),
        ("Can we go deeper?", ["explanation", "chat"], 2),
        ("Need both theory and practice", ["explanation", "coding"], 3),
        ("Sort of a research question", ["research", "chat"], 2),
        ("Design-ish coding task", ["architecture", "coding"], 3),
        ("Quick math or intuition?", ["math", "explanation"], 2),
        ("Is Dense or MoE better here?", ["research", "explanation"], 3),
        ("Please advise", ["chat", "brainstorm"], 2),
        ("Take a look", ["chat"], 1),
        ("What's next?", ["chat", "brainstorm"], 2),
        ("Break it down", ["explanation"], 2),
        ("Keep it simple but complete", ["explanation", "chat"], 2),
        ("Architecture or just a script?", ["architecture", "coding"], 3),
        ("I need a solid answer", ["chat", "explanation"], 2),
        ("Context: local inference, limited VRAM", ["research", "chat"], 2),
        ("Either optimize or redesign", ["coding", "architecture"], 3),
        ("Ambiguous on purpose: explain cache", ["explanation"], 2),
    ]
    for i, (p, intents, cx) in enumerate(ambiguous):
        rows.append(row(f"ambig_{i:03d}", "ambiguous", p, intents, "semantic", complexity=cx, deliverable="reply", reasoning="medium", length="medium", expected_tokens=1024))

    # --- architecture dedicated ---
    arch = [
        "Design a trading engine",
        "Design a multi-tenant SaaS billing system",
        "Propose an architecture for real-time notifications",
        "Design an offline-first mobile sync stack",
        "Architect a feature flag service",
        "Design a local-first LLM chat app with model hot-swap",
        "Propose a data lakehouse layout for clickstream",
        "Design a job scheduler for GPU workers",
        "Architect a secure secrets distribution system",
        "Design a matching engine for limit orders",
        "Propose HAIE-like controller architecture for sampling",
        "Design a CDN edge caching policy service",
        "Architect event sourcing for an inventory system",
        "Design a plugin sandbox for untrusted tools",
        "Propose a blue/green deploy topology for ML inference",
    ]
    for i, p in enumerate(arch):
        rows.append(row(f"arch_{i:03d}", "architecture", p, ["architecture"], "semantic", complexity=4, deliverable="design", reasoning="high", length="long", expected_tokens=4096))

    return rows


def main() -> int:
    rows = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    splits: dict[str, int] = {}
    for r in rows:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    print(f"wrote {OUT} n={len(rows)} splits={splits}")
    if len(rows) < 200:
        print("ERROR: need >=200 labeled examples")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
