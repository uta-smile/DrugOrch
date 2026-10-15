"""Anthropic backend and the response cache that makes runs replayable."""


def build_backend(settings=None, **overrides):
    """The LLM backend this install is configured to use.

    One place decides the provider, so the policy, the CLI and the doctor all agree about
    which model is about to be charged -- and a run's provenance records it without the
    policy needing to know there is more than one.
    """
    from ..config import get_settings

    st = settings or get_settings()
    provider = overrides.pop("provider", None) or st.llm_provider

    if provider == "anthropic":
        from .backend import AnthropicBackend

        return AnthropicBackend(
            model=st.llm_model,
            effort=st.effort_for_provider(),
            max_tokens=st.llm_max_tokens,
            timeout_s=st.llm_timeout_s,
            api_key=st.api_key(),
            **overrides,
        )

    from .openai_backend import OpenAIBackend

    if provider == "openai":
        return OpenAIBackend(
            model=st.openai_model,
            effort=st.effort_for_provider(),
            max_tokens=st.llm_max_tokens,
            timeout_s=st.llm_timeout_s,
            api_key=st.openai_key(),
            route="api_key",
            **overrides,
        )

    if provider == "chatgpt":
        from .chatgpt_auth import SessionStore, ensure_fresh

        store = SessionStore(st.chatgpt_session_path)
        session = ensure_fresh(store)

        def _refresh():
            s = ensure_fresh(store)
            return s.access_token if s else None

        return OpenAIBackend(
            model=st.openai_model,
            effort=st.effort_for_provider(),
            max_tokens=st.llm_max_tokens,
            timeout_s=st.llm_timeout_s,
            access_token=session.access_token if session else None,
            on_refresh=_refresh,
            route="chatgpt_plan",
            **overrides,
        )

    raise ValueError(f"unknown llm provider {provider!r}")

