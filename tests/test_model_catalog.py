"""Model and reasoning-effort selection.

Effort is not a shared vocabulary across providers, and it is not even shared across models
within one provider: on a ChatGPT plan some models take ``ultra`` and others stop at ``max``,
while the public GPT-5.1 docs list ``none`` that these reject. Sending an effort a model does
not accept is a 400, which the run loop turns into a scripted fallback — so the run finishes
looking normal while measuring the wrong arm. That is why this is validated rather than
passed through.
"""

from __future__ import annotations

import pytest

from drugorch.config import Settings
from drugorch.llm.catalog import (
    ANTHROPIC_EFFORTS,
    ModelOption,
    _parse_api_models,
    _parse_plan_models,
    coerce_effort,
    efforts_for,
    valid_effort,
)

# The shape the ChatGPT plan route actually returns — not the public API's data/id shape.
PLAN_PAYLOAD = {
    "models": [
        {
            "slug": "gpt-6.1-sol",
            "display_name": "GPT-6.1-Sol",
            "description": "Latest workhorse model.",
            "default_reasoning_level": "low",
            "supported_reasoning_levels": [
                {"effort": "low", "description": "Fast responses"},
                {"effort": "medium", "description": "Balanced"},
                {"effort": "ultra", "description": "Maximum depth"},
            ],
        },
        {
            "slug": "gpt-6-luna",
            "display_name": "GPT-6-Luna",
            "default_reasoning_level": "medium",
            "supported_reasoning_levels": [
                {"effort": "low", "description": ""},
                {"effort": "medium", "description": ""},
            ],
        },
    ]
}


# --- parsing the two listing shapes ----------------------------------------
def test_plan_listing_is_parsed_from_slug_not_id():
    models = _parse_plan_models(PLAN_PAYLOAD, current="gpt-6-luna")
    assert [m.id for m in models] == ["gpt-6.1-sol", "gpt-6-luna"]
    assert models[0].label == "GPT-6.1-Sol"
    assert models[0].note == "Latest workhorse model."


def test_plan_listing_carries_per_model_efforts():
    """The whole point: these two models do not accept the same levels."""
    a, b = _parse_plan_models(PLAN_PAYLOAD, current="")
    assert a.efforts == ("low", "medium", "ultra")
    assert b.efforts == ("low", "medium")
    assert a.default_effort == "low"
    assert b.default_effort == "medium"
    assert dict(a.effort_notes)["ultra"] == "Maximum depth"


def test_the_configured_model_is_marked_recommended():
    models = _parse_plan_models(PLAN_PAYLOAD, current="gpt-6-luna")
    assert [m.id for m in models if m.recommended] == ["gpt-6-luna"]


def test_public_api_listing_is_parsed_from_data_and_id():
    payload = {"data": [{"id": "gpt-5.5"}, {"id": "text-embedding-3-large"}]}
    models = _parse_api_models(payload, current="gpt-5.5")
    assert [m.id for m in models] == ["gpt-5.5"]  # the embedding model is filtered out


def test_non_chat_models_are_filtered_from_the_public_listing():
    """A dropdown offering an embedding endpoint invites a run that dies on step one."""
    payload = {
        "data": [
            {"id": "gpt-5.5"},
            {"id": "whisper-1"},
            {"id": "dall-e-3"},
            {"id": "text-embedding-3-small"},
            {"id": "omni-moderation-latest"},
        ]
    }
    assert [m.id for m in _parse_api_models(payload, "")] == ["gpt-5.5"]


def test_a_malformed_row_is_skipped_rather_than_crashing_the_picker():
    models = _parse_plan_models({"models": [{"no_slug": True}, {"slug": "ok"}]}, "")
    assert [m.id for m in models] == ["ok"]


# --- validation ------------------------------------------------------------
def test_a_models_own_list_overrides_the_provider_wide_one():
    """`ultra` is not in the static OpenAI set but a model may still accept it."""
    assert valid_effort("chatgpt", "ultra", ("low", "medium", "ultra")) is True
    assert valid_effort("chatgpt", "ultra", ("low", "medium")) is False


def test_without_a_model_list_the_provider_set_is_used():
    assert valid_effort("anthropic", "max") is True
    assert valid_effort("anthropic", "nonsense") is False


def test_providers_do_not_share_an_effort_vocabulary():
    assert "max" in efforts_for("anthropic")
    assert "none" in efforts_for("chatgpt")
    assert "none" not in efforts_for("anthropic")


@pytest.mark.parametrize(
    ("provider", "asked", "expected"),
    [
        ("anthropic", "max", "max"),
        ("chatgpt", "max", "max"),
        ("chatgpt", "none", "none"),
        ("anthropic", "none", "medium"),  # OpenAI-only level, so fall back
        ("chatgpt", "nonsense", "medium"),
    ],
)
def test_coerce_falls_back_to_the_default_not_a_neighbour(provider, asked, expected):
    """Silently turning `max` into `high` would record an effort nobody chose."""
    assert coerce_effort(provider, asked) == expected


# --- settings --------------------------------------------------------------
def test_effort_is_stored_per_provider_so_a_switch_does_not_corrupt_it():
    s = Settings(llm_provider="anthropic")
    s.set_effort("max")
    assert s.effort_for_provider() == "max"

    s.llm_provider = "chatgpt"
    s.set_effort("low")
    assert s.effort_for_provider() == "low"

    # Going back must find the Anthropic value still intact, not "low".
    s.llm_provider = "anthropic"
    assert s.effort_for_provider() == "max"


def test_model_is_stored_per_provider_too():
    s = Settings(llm_provider="anthropic")
    s.set_model("claude-sonnet-5-5")
    s.llm_provider = "chatgpt"
    s.set_model("gpt-6-luna")
    assert s.model_for_provider() == "gpt-6-luna"
    s.llm_provider = "anthropic"
    assert s.model_for_provider() == "claude-sonnet-5-5"


def test_an_invalid_stored_effort_is_corrected_on_read():
    """`.env` is hand-edited, so an impossible value can arrive from outside the UI."""
    s = Settings(llm_provider="chatgpt", openai_effort="bogus")
    assert s.effort_for_provider() == "medium"


def test_model_and_effort_reach_provenance():
    s = Settings(llm_provider="anthropic", llm_model="claude-sonnet-5-5", llm_effort="high")
    d = s.describe()
    assert d["llm_provider"] == "anthropic"
    assert d["llm_model"] == "claude-sonnet-5-5"
    assert d["llm_effort"] == "high"


def test_backend_cache_key_changes_with_model_and_effort():
    """Otherwise a replay would serve decisions made by a different configuration."""
    from drugorch.llm import build_backend

    a = build_backend(Settings(llm_provider="anthropic", llm_model="claude-opus-5-5"))
    b = build_backend(Settings(llm_provider="anthropic", llm_model="claude-haiku-5-5"))
    c = build_backend(Settings(llm_provider="anthropic", llm_effort="max"))
    assert a.key() != b.key()
    assert a.key() != c.key()


def test_model_option_serializes_for_the_picker():
    d = ModelOption("m", "M", "note", True, ("low",), "low", (("low", "fast"),)).to_dict()
    assert d["efforts"] == ["low"]
    assert d["effort_notes"] == {"low": "fast"}
    assert set(ANTHROPIC_EFFORTS) >= {"low", "medium", "high"}
