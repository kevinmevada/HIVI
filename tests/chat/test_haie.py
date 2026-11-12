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


def test_uncensored_sampling_stays_adaptive():
    from haie.policy import adapt_for_uncensored

    def u(intents, complexity=3, reasoning="medium"):
        return Understanding(intents=intents, complexity=complexity, deliverable="reply",
                             reasoning=reasoning, length="medium", ambiguity=0.2,
                             confidence=0.9, path="heuristic")

    code_u, chat_u, idea_u = u(["coding"]), u(["chat"]), u(["brainstorm"])
    code = adapt_for_uncensored(derive_params(code_u), code_u)
    chat = adapt_for_uncensored(derive_params(chat_u), chat_u)
    idea = adapt_for_uncensored(derive_params(idea_u), idea_u)
    # still per-request: coding coolest, brainstorm warmest — not one fixed value
    assert 0.55 <= code.temperature < chat.temperature < idea.temperature <= 1.0
    assert code.enable_thinking is False and code.thinking_budget_tokens == 0  # ordinary task: direct
    low_u = u(["chat"], complexity=2, reasoning="low")
    assert adapt_for_uncensored(derive_params(low_u), low_u).enable_thinking is False
    # thinking headroom grows with the effort HAIE picked
    hi_u = u(["coding"], complexity=5, reasoning="high")
    hi = adapt_for_uncensored(derive_params(hi_u), hi_u)
    assert hi.enable_thinking is True and hi.max_tokens > code.max_tokens
    # trivial turns skip thinking for speed
    hello_u = u(["greeting"], complexity=1, reasoning="none")
    assert adapt_for_uncensored(derive_params(hello_u), hello_u).enable_thinking is False


def test_uncensored_request_body_and_client_override():
    body = {"model": "local",
            "messages": [{"role": "user", "content": "write a python function that parses a csv file"}]}
    normal, _ = plan_request(body, engine="llama", backend_base=DEAD_BACKEND, model_id=None)
    adapted, meta = plan_request(body, engine="llama", backend_base=DEAD_BACKEND, model_id=None,
                                 uncensored=True)
    assert normal["temperature"] < adapted["temperature"]  # re-centred, not replaced
    # simple request -> direct answer (fast); thinking is reserved for medium/high effort
    think = adapted["chat_template_kwargs"]["enable_thinking"]
    assert think is (meta.params["reasoning_effort"] == "high" or meta.complexity >= 4)
    assert "thinking_budget_tokens" not in adapted
    assert meta.to_headers()["X-HIVI-HAIE-Thinking"] == ("on" if think else "off")
    # explicit user/client values always win
    mine = dict(body, temperature=0.2, chat_template_kwargs={"enable_thinking": False})
    adapted2, _ = plan_request(mine, engine="llama", backend_base=DEAD_BACKEND, model_id=None,
                               uncensored=True)
    assert adapted2["temperature"] == 0.2
    assert adapted2["chat_template_kwargs"]["enable_thinking"] is False


def test_colibri_gets_thinking_as_top_level_fields():
    """coli serve ignores chat_template_kwargs; unset, Qwen3.8 thinks at xhigh every turn."""
    easy = {"model": "m", "messages": [{"role": "user", "content": "hi there"}]}
    out, meta = plan_request(easy, engine="colibri", backend_base=DEAD_BACKEND, model_id=None)
    assert out["enable_thinking"] is False and "reasoning_effort" not in out
    assert meta.to_headers()["X-HIVI-HAIE-Thinking"] == "off"

    hard = {"model": "m", "messages": [{"role": "user", "content":
            "Design a distributed event-sourcing architecture with CQRS, sagas, exactly-once "
            "delivery, multi-region failover and a migration plan from our monolith; compare "
            "three options with trade-offs and prove the consistency guarantees."}]}
    out, meta = plan_request(hard, engine="colibri", backend_base=DEAD_BACKEND, model_id=None)
    if out["enable_thinking"]:
        assert out["reasoning_effort"] in ("low", "medium", "high")
        assert out["max_tokens"] <= 6144  # coli serve runs at --ctx 8192

    # the UI's refusal retry sends chat_template_kwargs; Colibrì needs it top-level
    retry = dict(easy, chat_template_kwargs={"enable_thinking": True})
    out, _ = plan_request(retry, engine="colibri", backend_base=DEAD_BACKEND, model_id=None,
                          uncensored=True)
    assert out["enable_thinking"] is True and out["reasoning_effort"] in ("low", "medium", "high")

    # an explicit client value still wins
    mine = dict(easy, enable_thinking=True, reasoning_effort="xhigh")
    out, _ = plan_request(mine, engine="colibri", backend_base=DEAD_BACKEND, model_id=None)
    assert out["enable_thinking"] is True and out["reasoning_effort"] == "xhigh"
    # llama requests are unchanged by this
    out, _ = plan_request(easy, engine="llama", backend_base=DEAD_BACKEND, model_id=None)
    assert "enable_thinking" not in out


def test_adapt_thinking_rule_and_budget():
    from haie.policy import COLIBRI_MAX_TOKENS_CEILING, adapt_thinking
    from haie.types import SampleParams, Understanding

    base = SampleParams(temperature=0.3, top_p=0.9, top_k=40, max_tokens=4096,
                        thinking_budget_tokens=0, reasoning_effort="high")
    hard = adapt_thinking(base, Understanding(intents=["architecture"], complexity=4))
    assert hard.enable_thinking is True and hard.max_tokens == COLIBRI_MAX_TOKENS_CEILING
    easy = adapt_thinking(
        SampleParams(0.5, 0.9, 40, 512, 0, "low"), Understanding(intents=["chat"], complexity=1))
    assert easy.enable_thinking is False and easy.max_tokens == 512
    preset = adapt_thinking(SampleParams(0.5, 0.9, 40, 512, 0, "low", enable_thinking=True),
                            Understanding())
    assert preset.enable_thinking is True  # an earlier decision (uncensored path) is kept


def test_acknowledgements_get_a_short_budget_even_in_long_chats():
    """After "thank you" / "perfect" models re-sent the whole previous answer."""
    from haie.fast_path import is_acknowledgement

    for t in ["thank you", "thankyou", "perfect", "perfect.. thanks", "nice it works",
              "thanks bro", "Thanks a lot man!!", "badhiya bhai", "sahi hai yaar", "ok thanks",
              "works perfectly"]:
        assert is_acknowledgement(t), t
    # real requests, questions and bare "go ahead" answers are not
    for t in ["great, now add logging", "perfect, but make it async", "thanks, what about retries?",
              "ok do it", "yes", "ok", "yes add tests", "is it perfect", "perfect now write tests"]:
        assert not is_acknowledgement(t), t

    history = [{"role": "user", "content": "write a python scraper"},
               {"role": "assistant", "content": "Here is the script ..."}] * 6  # long chat
    body = {"model": "m", "messages": history + [{"role": "user", "content": "perfect thanks"}]}
    out, meta = plan_request(body, engine="llama", backend_base=DEAD_BACKEND, model_id=None,
                             uncensored=True)
    assert meta.path == "fast" and meta.deliverable == "acknowledgement"
    assert out["max_tokens"] <= 256
    assert out["chat_template_kwargs"]["enable_thinking"] is False


def test_short_followups_get_previous_exchange_as_classifier_context(monkeypatch):
    """ "make it async" can't be classified without knowing what "it" is."""
    import haie.semantic as semantic

    seen: list[str] = []

    def fake_call(*, client, url, model, prompt, system, max_tokens, prefill):
        seen.append(prompt)
        return None

    monkeypatch.setattr(semantic, "_one_classify_call", fake_call)
    history = [
        {"role": "user", "content": "write a python scraper for job postings\nwith retries"},
        {"role": "assistant", "content": "Here is the script:\n```python\nimport httpx\n```"},
    ]
    body = {"model": "m", "messages": history + [{"role": "user", "content": "make it async"}]}
    plan_request(body, engine="llama", backend_base="http://127.0.0.1:9", model_id=None)
    assert seen, "follow-up should reach the semantic classifier"
    prompt = seen[0]
    assert "previous user: write a python scraper for job postings with retries" in prompt
    assert "previous assistant: Here is the script:" in prompt
    assert prompt.endswith("Request to classify:\nmake it async")
    assert "make it async" not in prompt.split("Request to classify:")[0]  # not duplicated
    assert len(prompt) < 400

    # long requests stand alone: no history in the classifier prompt
    seen.clear()
    long_req = "Design a job-scraping pipeline " + "with queues, retries and storage " * 6
    semantic.classify_semantic(backend_base="http://127.0.0.1:9", model="m", user_text=long_req,
                               recent_context="previous user: x")
    assert seen and seen[0] == long_req.strip()[:500]

    # short chit-chat still takes the fast path; short follow-ups don't
    from haie.fast_path import try_fast_path
    assert try_fast_path("tell me a joke", history_len=3).intents == ["chat"]
    for t in ("make it async", "now do it in rust", "same for the other one"):
        assert try_fast_path(t, history_len=3) is None, t
