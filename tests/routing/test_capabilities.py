from types import SimpleNamespace

from routing.capabilities import model_capabilities


def test_coder_model_has_coding_capability():
    model = SimpleNamespace(
        name="DeepSeek Coder",
        path="Models/Coding/deepseek.gguf",
        params="16B",
    )
    assert {"general", "coding"} <= model_capabilities(model)
