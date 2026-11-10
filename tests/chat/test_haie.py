"""HAIE: fast/heuristic paths, param policy, and graceful fallback."""
from __future__ import annotations

from haie import plan_request
from haie.fast_path import extract_last_user, try_fast_path
from haie.heuristic import try_heuristic_path
from haie.policy import derive_params
from haie.types import Understanding

DEAD_BACKEND = "http://127.0.0.1:9"  # nothing listens on the discard port


def test_greeting_and_simple_math_take_fast_path():
    assert try_fast_path("hi").intents == ["greeting"]
    assert try_fast_path("what is 2+2").intents == ["math"]


def test_task_verbs_never_take_chat_fast_path():
    assert try_fast_path("Create a parser") is None
    assert try_fast_path("Why does this fail") is None


def test_heuristic_handles_clear_single_intent():
    u = try_heuristic_path("Create a REST API in FastAPI with JWT auth")
    assert u is not None and u.intents == ["coding"]


def test_math_budget_stays_small_and_coding_is_cold():
    math = derive_params(Understanding(intents=["math"], complexity=1, deliverable="solution",
                                       reasoning="none", length="short", ambiguity=0.1,
                                       confidence=0.9, path="fast"))
    assert math.max_tokens <= 512 and math.thinking_budget_tokens == 0
    code = derive_params(Understanding(intents=["coding"], complexity=4, deliverable="code",
                                       reasoning="high", length="long", ambiguity=0.1,
                                       confidence=0.9, path="semantic"))
    assert code.temperature <= 0.18 and code.max_tokens >= 4096


def test_image_messages_still_yield_text():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is in this picture"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]}
    assert extract_last_user(body) == "what is in this picture"


def test_unreachable_backend_falls_back_and_client_values_win():
    body = {
        "model": "local",
        "temperature": 0.9,  # explicit client value must survive
        "messages": [{"role": "user", "content":
                      "I need help thinking about my project's next milestone given the "
                      "audit results and constraints we discussed, what would you weigh?"}],
    }
    adapted, meta = plan_request(body, engine="llama", backend_base=DEAD_BACKEND, model_id=None)
    assert meta.path in ("fallback", "heuristic", "fast")
    assert adapted["temperature"] == 0.9
    assert "max_tokens" in adapted
