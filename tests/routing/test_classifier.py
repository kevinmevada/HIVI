from routing.classifier import classify_prompt


def test_classifies_coding_prompt():
    result = classify_prompt("Debug this Python function and explain the bug")
    assert result.task == "coding"
    assert "coding" in result.capabilities


def test_defaults_to_general():
    result = classify_prompt("What is the capital of France?")
    assert result.task == "general"
    assert result.complexity == "standard"
