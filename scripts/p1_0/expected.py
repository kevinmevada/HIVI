"""Expected inventory — not proof of existence."""

from __future__ import annotations

PC_EXPECTED = [
    {
        "id": "M01",
        "name": "Qwen2.5-Coder-14B-Instruct-Abliterated",
        "format": "GGUF Q8_0",
        "match": ["qwen2.5-coder-14b", "abliterated", "q8_0"],
    },
    {
        "id": "M02",
        "name": "GLM-4.7-Flash-Uncensored-Heretic",
        "format": "GGUF Q6_K",
        "match": ["glm-4.7-flash", "q6_k"],
    },
    {
        "id": "M03",
        "name": "Qwen3.8-27B-RVN-Heretic",
        "format": "GGUF Q6_K",
        "match": ["0bserverx", "rvn-q6", "heretic-abliterated"],
    },
    {
        "id": "M04",
        "name": "Qwen3-8B-Hivemind-Heretic",
        "format": "GGUF Q6_K",
        "match": ["hivemind", "heretic", "q6_k"],
    },
    {
        "id": "M05",
        "name": "Qwen3.8-27B-Uncensored",
        "format": "GGUF Q6_K",
        "match": ["jonathan-coletti", "uncensored-q6"],
    },
    {
        "id": "M06",
        "name": "Qwen3.6-35B-A3B-Colibri",
        "format": "Safetensors / I4-GS64",
        "match": ["colibri", "kreuzzelg", "i4-gs64"],
    },
    {
        "id": "M07",
        "name": "Ornith-1.5-9B",
        "format": "GGUF Q6_K",
        "match": ["ornith-1.5-9b", "q6_k"],
    },
    {
        "id": "M08",
        "name": "Ornith-1.5-35B-A3B",
        "format": "GGUF Q5_K_M",
        "match": ["ornith-1.5-35b", "q5_k_m"],
    },
    {
        "id": "M09",
        "name": "Qwen3.8-27B-Unsloth",
        "format": "GGUF Q5_K_M",
        "match": ["unsloth", "qwen3.8-27b", "q5_k_m"],
    },
]

USB_EXPECTED = [
    {
        "id": "U01",
        "name": "Qwen3.5-9B-Abliterated",
        "format": "GGUF Q4_K_M",
        "match": ["qwen3.5-9b-abliterated", "q4_k_m"],
    },
    {
        "id": "U02",
        "name": "Tiny-Random-BERT",
        "format": "Safetensors / PyTorch",
        "match": ["tiny-random-bert"],
        "exclude_sub": ["onnx"],
    },
    {
        "id": "U03",
        "name": "Tiny-Random-BERT ONNX",
        "format": "ONNX",
        "match": ["tiny-random-bert", "onnx"],
    },
]

OLLAMA_EXPECTED = ["smollm:135m", "qwen3.5:2b", "nomic-embed-text:latest"]

WEIGHT_EXTS = {".gguf", ".safetensors", ".bin", ".pt", ".pth", ".onnx", ".h5"}
META_EXTS = {".json", ".txt", ".jinja", ".model", ".py", ".md"}
SKIP_DIR_NAMES = {".cache", ".git", "__pycache__", "blobs"}
