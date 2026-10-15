"""What models and reasoning efforts each provider accepts, and which are worth picking.

Effort is not a shared vocabulary. Claude Opus 5.5 takes ``low`` through ``max``; GPT-5.1
takes ``none`` through ``high`` and rejects ``max`` outright, and ``xhigh`` only exists on
models after ``gpt-5.1-codex-max``. Offering one merged list would let a user choose an
effort their model rejects, and a rejected request is not a visible error -- the policy
degrades to the scripted arm and the run finishes looking entirely normal while measuring
the wrong thing. So the sets are kept apart and validated before anything is saved.

Model lists are fetched from the provider when a credential exists, because a hardcoded
catalogue is wrong the moment a model ships. The lists below are the fallback for a signed-out
install, and the API marks which of the two a response came from so the UI can say so rather
than implying it checked.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- reasoning effort ----------------------------------------------------------
#
# Ordered cheapest-to-most-thorough, which is the order they are shown in.

ANTHROPIC_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
OPENAI_EFFORTS: tuple[str, ...] = (
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
)
"""The union across OpenAI models, used only when the provider will not tell us what a
specific model takes. Per-model lists are narrower *and* wider than any guess: the ChatGPT
plan models accept ``max`` and ``ultra`` that the public GPT-5.1 docs do not list, and
reject ``none`` that they do."""

EFFORTS: dict[str, tuple[str, ...]] = {
    "anthropic": ANTHROPIC_EFFORTS,
    "openai": OPENAI_EFFORTS,
    "chatgpt": OPENAI_EFFORTS,
}

DEFAULT_EFFORT: dict[str, str] = {
    "anthropic": "medium",
    "openai": "medium",
    "chatgpt": "medium",
}

EFFORT_NOTES: dict[str, str] = {
    "none": "No reasoning. Fast and cheap, and routing decisions get noticeably worse.",
    "minimal": "Barely any reasoning. Not supported on every model.",
    "low": "Quick decisions. Fine for short runs where the next move is usually obvious.",
    "medium": "The default, and what the published comparisons here were run at.",
    "high": "More deliberation per decision. Slower and dearer; use when budgets are large.",
    "xhigh": "Deep reasoning. Only on newer models.",
    "max": "The most reasoning a model offers. Slow and expensive per decision.",
    "ultra": "Beyond max, where a model offers it. Expect seconds per decision.",
}


@dataclass(frozen=True, slots=True)
class ModelOption:
    id: str
    label: str = ""
    note: str = ""
    recommended: bool = False
    efforts: tuple[str, ...] = ()
    """Reasoning levels this specific model accepts, when the provider reports them."""

    default_effort: str = ""
    effort_notes: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label or self.id,
            "note": self.note,
            "recommended": self.recommended,
            "efforts": list(self.efforts),
            "default_effort": self.default_effort,
            "effort_notes": dict(self.effort_notes),
        }


# --- fallback catalogues --------------------------------------------------------
#
# Used only when no credential is available to ask with. Model names move fast enough that
# this list is a starting point, not a source of truth -- the ChatGPT plan route in
# particular serves an entirely different set from the public API.

FALLBACK_MODELS: dict[str, tuple[ModelOption, ...]] = {
    "anthropic": (
        ModelOption("claude-opus-5-5", "Claude Opus 5.5",
                    "Most capable. What every comparison in this repo was run on.", True),
        ModelOption("claude-sonnet-5-5", "Claude Sonnet 5.5",
                    "Faster and cheaper; a reasonable second arm."),
        ModelOption("claude-haiku-5-5", "Claude Haiku 5.5",
                    "Cheapest. Useful for checking the harness, not for a result."),
    ),
    "openai": (
        ModelOption("gpt-5.5", "GPT-5.5", "", True),
    ),
}
FALLBACK_MODELS["chatgpt"] = FALLBACK_MODELS["openai"]


def _parse_plan_models(payload: dict, current: str) -> list[ModelOption]:
    """The ChatGPT-plan listing, which is not the public API's shape.

    It returns ``{"models": [{"slug": ..., "supported_reasoning_levels": [...]}]}`` rather
    than ``{"data": [{"id": ...}]}``, and it carries per-model reasoning levels with their
    own descriptions -- better information than any table kept here could be, so it is used
    in preference to the static one.
    """
    out: list[ModelOption] = []
    for m in payload.get("models") or []:
        slug = m.get("slug")
        if not slug:
            continue
        levels = m.get("supported_reasoning_levels") or []
        efforts = tuple(lv.get("effort", "") for lv in levels if lv.get("effort"))
        notes = tuple(
            (lv["effort"], lv.get("description", "")) for lv in levels if lv.get("effort")
        )
        out.append(
            ModelOption(
                id=slug,
                label=m.get("display_name") or slug,
                note=m.get("description") or "",
                recommended=slug == current,
                efforts=efforts,
                default_effort=m.get("default_reasoning_level") or "",
                effort_notes=notes,
            )
        )
    return out


def _parse_api_models(payload: dict, current: str) -> list[ModelOption]:
    """The public API listing: ``{"data": [{"id": ...}]}`` with no reasoning metadata."""
    out = [
        ModelOption(m["id"], m["id"], recommended=m["id"] == current)
        for m in payload.get("data") or []
        if _is_chat_model(m.get("id", ""))
    ]
    out.sort(key=lambda m: m.id)
    return out


def efforts_for(provider: str) -> tuple[str, ...]:
    return EFFORTS.get(provider, ANTHROPIC_EFFORTS)


def valid_effort(provider: str, effort: str, model_efforts: tuple[str, ...] = ()) -> bool:
    """Whether this effort can be sent.

    When the provider has told us what a specific model accepts, that list is authoritative
    -- it is both narrower and wider than any table kept here, and the static set exists
    only for a signed-out install that cannot ask.
    """
    if model_efforts:
        return effort in model_efforts
    return effort in efforts_for(provider)


def model_efforts(provider: str, model: str, settings) -> tuple[str, ...]:
    """What this one model accepts, from the live listing, or ``()`` if unknown."""
    for m in live_models(provider, settings) or ():
        if m.id == model:
            return m.efforts
    return ()


def model_default_effort(provider: str, model: str, settings) -> str:
    """The model's own preferred level, which is a better landing place than ours."""
    for m in live_models(provider, settings) or ():
        if m.id == model and m.default_effort:
            return m.default_effort
    return DEFAULT_EFFORT.get(provider, "medium")


def coerce_effort(provider: str, effort: str) -> str:
    """The nearest acceptable effort, so switching provider never saves an invalid one.

    Falls back to the provider's default rather than to a neighbouring level: silently
    turning ``max`` into ``high`` would mean a run recorded an effort nobody chose.
    """
    return effort if valid_effort(provider, effort) else DEFAULT_EFFORT.get(provider, "medium")


def fallback_models(provider: str) -> list[dict]:
    return [m.to_dict() for m in FALLBACK_MODELS.get(provider, ())]


def live_models(provider: str, settings) -> list[ModelOption] | None:
    """The models this account can actually use, or ``None`` if we cannot ask."""
    try:
        if provider == "anthropic":
            key = settings.api_key()
            if not key:
                return None
            import anthropic

            rows = anthropic.Anthropic(api_key=key, timeout=20).models.list(limit=100)
            out = [
                ModelOption(m.id, getattr(m, "display_name", "") or m.id,
                            recommended=m.id == settings.llm_model)
                for m in getattr(rows, "data", [])
                if _is_chat_model(m.id)
            ]
            return out or None

        token = _openai_token(provider, settings)
        if not token:
            return None
        import httpx

        r = httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
        if r.status_code >= 400:
            return None
        payload = r.json()
        current = settings.openai_model
        out = (
            _parse_plan_models(payload, current)
            if "models" in payload
            else _parse_api_models(payload, current)
        )
        return out or None
    except Exception:  # noqa: BLE001 - a listing failure falls back, it does not break the page
        return None


def _openai_token(provider: str, settings) -> str | None:
    if provider == "openai":
        return settings.openai_key()
    from .chatgpt_auth import SessionStore, ensure_fresh

    session = ensure_fresh(SessionStore(settings.chatgpt_session_path))
    return session.access_token if session else None


_NOT_CHAT = (
    "embedding", "whisper", "tts", "dall-e", "moderation", "audio", "image", "realtime",
    "transcribe", "search", "rerank", "sora", "codex-mini",
)


def _is_chat_model(model_id: str) -> bool:
    low = model_id.lower()
    if any(bad in low for bad in _NOT_CHAT):
        return False
    return low.startswith(("gpt-", "claude-", "o1", "o3", "o4"))


def catalog(provider: str, settings) -> dict:
    """Everything the model/effort picker needs for one provider.

    Efforts are reported per model when the provider says what each model takes, and only
    fall back to the provider-wide set otherwise. That difference matters: on a ChatGPT plan
    the levels vary between models, and offering one a model rejects produces a 400 that the
    run loop turns into a silent scripted fallback.
    """
    live = live_models(provider, settings)
    models = live if live is not None else list(FALLBACK_MODELS.get(provider, ()))
    current = settings.llm_model if provider == "anthropic" else settings.openai_model

    return {
        "provider": provider,
        "models": [m.to_dict() for m in models],
        "source": "live" if live is not None else "fallback",
        "current_available": any(m.id == current for m in models) if models else True,
        "efforts": [
            {"id": e, "note": EFFORT_NOTES.get(e, "")} for e in efforts_for(provider)
        ],
        "default_effort": DEFAULT_EFFORT.get(provider, "medium"),
    }


__all__ = [
    "ANTHROPIC_EFFORTS",
    "DEFAULT_EFFORT",
    "EFFORTS",
    "EFFORT_NOTES",
    "OPENAI_EFFORTS",
    "ModelOption",
    "catalog",
    "coerce_effort",
    "efforts_for",
    "fallback_models",
    "live_models",
    "valid_effort",
]
