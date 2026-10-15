"""The OpenAI Responses backend: request shape, stream assembly, and provider symmetry.

The ChatGPT-plan preview rejects a long list of ordinary parameters outright. Sending one is
not a soft failure — the request 400s, the policy falls back to the scripted arm, and the run
completes looking normal while measuring the wrong thing. That makes the request shape worth
pinning line by line.
"""

from __future__ import annotations

import json

import pytest

from drugorch.llm.openai_backend import FORBIDDEN_ON_PLAN, OpenAIBackend, _consume
from drugorch.llm.response_cache import LLMUnavailable

SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string"}},
    "required": ["action"],
    "additionalProperties": False,
}


def _plan(**kw) -> OpenAIBackend:
    return OpenAIBackend(access_token="tok", route="chatgpt_plan", **kw)


def _key(**kw) -> OpenAIBackend:
    return OpenAIBackend(api_key="sk-test", route="api_key", **kw)


# --- the plan request shape ------------------------------------------------
def test_plan_requests_are_streamed_and_unstored():
    """Both are mandatory on this route; the API rejects the request without them."""
    body = _plan()._body("sys", "manifest", "state", SCHEMA)
    assert body["stream"] is True
    assert body["store"] is False


def test_plan_requests_omit_every_forbidden_parameter():
    body = _plan()._body("sys", "manifest", "state", SCHEMA)
    assert FORBIDDEN_ON_PLAN & set(body) == set()


def test_max_output_tokens_is_sent_only_on_the_api_key_route():
    """It is a normal parameter with a key and a rejected one on a plan."""
    assert "max_output_tokens" in _key()._body("s", "m", "p", SCHEMA)
    assert "max_output_tokens" not in _plan()._body("s", "m", "p", SCHEMA)


def test_system_guidance_goes_in_instructions_not_a_system_message():
    """Explicit {role: "system"} items are rejected on this route."""
    body = _plan()._body("SYSTEM TEXT", "MANIFEST", "STATE", SCHEMA)
    assert "SYSTEM TEXT" in body["instructions"]
    assert "MANIFEST" in body["instructions"]
    roles = [item.get("role") for item in body["input"]]
    assert "system" not in roles
    assert roles == ["user"]


def test_the_volatile_state_goes_last_in_the_input():
    body = _plan()._body("sys", "manifest", "THE STATE", SCHEMA)
    assert body["input"][-1]["content"][0]["text"] == "THE STATE"


def test_the_schema_is_attached_as_a_strict_json_schema():
    body = _plan()._body("s", "m", "p", SCHEMA)
    fmt = body["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["strict"] is True
    assert fmt["schema"] == SCHEMA


# --- identity and provenance -----------------------------------------------
def test_cache_keys_separate_the_two_auth_routes():
    """The plan route cannot pin sampling and the key route can, so a replay must not
    serve a decision recorded under the other one."""
    assert _plan().key() != _key().key()
    assert "chatgpt_plan" in _plan().key()
    assert "api_key" in _key().key()


def test_describe_reports_the_route_and_that_caching_is_unavailable_on_a_plan():
    d = _plan().describe()
    assert d["route"] == "chatgpt_plan"
    assert d["prompt_caching"] is False
    assert _key().describe()["prompt_caching"] is True


def test_describe_never_leaks_a_credential():
    plan = OpenAIBackend(access_token="SECRET-ACCESS-VALUE", route="chatgpt_plan")
    key = OpenAIBackend(api_key="sk-SECRET-KEY-VALUE", route="api_key")
    blob = json.dumps(plan.describe()) + json.dumps(key.describe())
    assert "SECRET" not in blob


def test_readiness_tracks_the_credential_the_route_actually_uses():
    """A ChatGPT install with a stray OPENAI_API_KEY is still not signed in."""
    assert OpenAIBackend(route="chatgpt_plan", api_key="sk-x").ready is False
    assert OpenAIBackend(route="chatgpt_plan", access_token="t").ready is True
    assert OpenAIBackend(route="api_key", access_token="t").ready is False


def test_a_missing_credential_names_the_right_fix():
    with pytest.raises(LLMUnavailable, match="not signed in to ChatGPT"):
        OpenAIBackend(route="chatgpt_plan").complete("s", "m", "p", SCHEMA)
    with pytest.raises(LLMUnavailable, match="OPENAI_API_KEY"):
        OpenAIBackend(route="api_key").complete("s", "m", "p", SCHEMA)


# --- stream assembly -------------------------------------------------------
def test_deltas_accumulate_into_the_response_text():
    chunks, meta = [], {}
    for part in ('{"act', 'ion":', ' "call"}'):
        _consume({"type": "response.output_text.delta", "delta": part}, chunks, meta)
    assert json.loads("".join(chunks)) == {"action": "call"}


def test_the_completed_event_supersedes_the_deltas():
    """The final object is authoritative; relying on deltas alone truncates on a dropped
    event, and a truncated decision is a silent fallback."""
    chunks, meta = ['{"partial'], {}
    _consume(
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"action":"stop"}'}],
                    }
                ],
                "usage": {
                    "input_tokens": 120,
                    "output_tokens": 30,
                    "input_tokens_details": {"cached_tokens": 64},
                },
            },
        },
        chunks,
        meta,
    )
    assert json.loads("".join(chunks)) == {"action": "stop"}
    assert meta["usage"]["input_tokens"] == 120
    assert meta["usage"]["cache_read_input_tokens"] == 64


def test_usage_field_names_match_the_anthropic_backend():
    """The LLM log reads one set of field names for either provider."""
    from drugorch.llm.backend import AnthropicBackend

    chunks, meta = [], {}
    _consume(
        {"type": "response.completed", "response": {"usage": {"input_tokens": 1}, "output": []}},
        chunks,
        meta,
    )
    assert {"input_tokens", "output_tokens", "cache_read_input_tokens"} <= set(meta["usage"])
    assert AnthropicBackend().describe()["backend"] == "anthropic"


def test_a_refusal_is_surfaced_rather_than_parsed_as_json():
    chunks, meta = [], {}
    with pytest.raises(LLMUnavailable, match="declined"):
        _consume(
            {
                "type": "response.completed",
                "response": {"output": [{"type": "message",
                                          "content": [{"type": "refusal", "refusal": "no"}]}]},
            },
            chunks,
            meta,
        )


def test_a_stream_error_event_raises():
    with pytest.raises(LLMUnavailable, match="boom"):
        _consume({"type": "error", "message": "boom"}, [], {})
