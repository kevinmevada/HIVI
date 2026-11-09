from types import SimpleNamespace

from routing.policy import RoutingPolicy
from routing.selector import select_model


def model(model_id: str, name: str, params: str, size_gb: float):
    return SimpleNamespace(
        id=model_id,
        name=name,
        path=f"Models/{model_id}.gguf",
        params=params,
        size_gb=size_gb,
        quant="Q4_K_M",
        engine="llama",
    )


def test_selector_prefers_coding_model():
    models = [
        model("general", "General model", "9B", 5),
        model("coder", "Qwen Coder", "14B", 10),
    ]
    decision = select_model("Implement this Python function", models, RoutingPolicy())
    assert decision.model.id == "coder"
    assert decision.classification.task == "coding"


def test_selector_respects_size_limit():
    models = [model("large", "Qwen Coder", "30B", 30)]
    decision = select_model(
        "Implement this Python function",
        models,
        RoutingPolicy(max_size_gb=10),
    )
    assert decision.model is None
