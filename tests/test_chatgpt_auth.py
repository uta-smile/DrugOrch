"""The Sign in with ChatGPT flow: request shape, the security checks, and token handling.

This is the one part of DrugOrch that holds a live credential for something the user pays
for, and the one part that cannot be driven end to end in a test — completing it needs a
human to click Approve. So the pieces that *can* be pinned are pinned hard: the exact
parameters OpenAI's preview requires, the checks that make the callback safe to trust, and
the rule that identity alone is not permission to spend someone's plan.
"""

from __future__ import annotations

import time
import urllib.parse

import pytest

from drugorch.llm.chatgpt_auth import (
    AUTHORIZE_URL,
    BOOTSTRAP_CLIENT_ID,
    PLAN_SCOPE,
    RESOURCE,
    SCOPES,
    ChatGPTSession,
    LoopbackServer,
    Pkce,
    SessionStore,
    authorize_url,
)


def _params(url: str) -> dict:
    return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(url).query).items()}


def _url(**kw) -> str:
    base = {
        "redirect_uri": "http://127.0.0.1:1455/auth/callback",
        "host_id": "urn:uuid:abc",
        "pkce": Pkce.generate(),
        "state": "st",
        "nonce": "no",
    }
    return authorize_url(**{**base, **kw})


# --- the authorize request -------------------------------------------------
def test_first_signin_uses_the_bootstrap_client_and_names_the_agent():
    """Registration is the only time agent_name_hint is allowed."""
    p = _params(_url())
    assert p["client_id"] == BOOTSTRAP_CLIENT_ID
    assert p["agent_name_hint"] == "DrugOrch"


def test_reauthorization_uses_the_issued_id_and_drops_the_name_hint():
    """Sending the hint again would register a second agent instead of reusing this one."""
    p = _params(_url(client_id="oaiapp_xyz"))
    assert p["client_id"] == "oaiapp_xyz"
    assert "agent_name_hint" not in p


def test_every_required_preview_parameter_is_present():
    p = _params(_url())
    assert p["response_type"] == "code"
    assert p["code_challenge_method"] == "S256"
    assert p["resource"] == RESOURCE
    assert p["ext_agent_host_id"] == "urn:uuid:abc"
    assert p["state"] and p["nonce"] and p["code_challenge"]


def test_plan_scope_is_requested():
    """Without this scope the user is signed in but nothing can be run on their plan."""
    assert PLAN_SCOPE in _params(_url())["scope"].split()
    assert "offline_access" in SCOPES  # or there is no refresh token to renew with


def test_authorize_url_points_at_the_documented_endpoint():
    assert _url().startswith(AUTHORIZE_URL + "?")


def test_hints_are_only_sent_on_reauthorization():
    fresh = _params(_url(login_hint="a@b.com", id_token_hint="tok"))
    assert "login_hint" not in fresh and "id_token_hint" not in fresh

    again = _params(_url(client_id="oaiapp_xyz", login_hint="a@b.com", id_token_hint="tok"))
    assert again["login_hint"] == "a@b.com"
    assert again["id_token_hint"] == "tok"


# --- PKCE ------------------------------------------------------------------
def test_pkce_challenge_is_unpadded_s256_of_the_verifier():
    import base64
    import hashlib

    p = Pkce.generate()
    expected = base64.urlsafe_b64encode(hashlib.sha256(p.verifier.encode()).digest())
    assert p.challenge == expected.decode().rstrip("=")
    assert "=" not in p.challenge


def test_each_attempt_gets_a_fresh_verifier():
    assert Pkce.generate().verifier != Pkce.generate().verifier


# --- the loopback listener -------------------------------------------------
def test_redirect_uri_matches_the_required_shape():
    """`localhost` and `/callback` are both rejected by OpenAI; only the port may vary."""
    server = LoopbackServer()
    try:
        uri = server.start()
        assert uri.startswith("http://127.0.0.1:")
        assert uri.endswith("/auth/callback")
        assert "localhost" not in uri
        assert server.port > 0
    finally:
        server.stop()


def test_loopback_captures_the_callback_parameters():
    import httpx

    server = LoopbackServer()
    try:
        server.start()
        r = httpx.get(
            f"http://127.0.0.1:{server.port}/auth/callback",
            params={"code": "C", "state": "S", "client_id": "oaiapp_1"},
            timeout=5,
        )
        assert r.status_code == 200
        got = server.wait(timeout_s=2)
        assert got["code"] == "C"
        assert got["state"] == "S"
        assert got["client_id"] == "oaiapp_1"
    finally:
        server.stop()


def test_loopback_ignores_a_request_to_another_path():
    """Anything on this port that is not the callback must not be mistaken for one."""
    import httpx

    server = LoopbackServer()
    try:
        server.start()
        r = httpx.get(f"http://127.0.0.1:{server.port}/not-the-callback?code=X", timeout=5)
        assert r.status_code == 404
        assert server.result == {}
    finally:
        server.stop()


# --- the session -----------------------------------------------------------
def test_identity_without_the_plan_scope_is_not_authorization():
    """Signing in proves who you are; it does not mean you agreed to spend your plan."""
    s = ChatGPTSession(client_id="c", access_token="t", scopes=("openid", "profile", "email"))
    assert not s.authorized_for_plan

    s2 = ChatGPTSession(client_id="c", access_token="t", scopes=("openid", PLAN_SCOPE))
    assert s2.authorized_for_plan


def test_expiry_leaves_a_margin_so_a_decision_cannot_straddle_it():
    nearly = ChatGPTSession(client_id="c", access_token="t", expires_at=time.time() + 30)
    assert nearly.expired  # inside the refresh margin, so renew before using
    fine = ChatGPTSession(client_id="c", access_token="t", expires_at=time.time() + 3600)
    assert not fine.expired


def test_public_view_never_exposes_tokens():
    """This is what the browser receives; a token here would be handed to any page script."""
    import json

    s = ChatGPTSession(
        client_id="oaiapp_1",
        access_token="SECRET-ACCESS",
        refresh_token="SECRET-REFRESH",
        id_token="SECRET-ID",
        account_email="a@b.com",
        scopes=(PLAN_SCOPE,),
    )
    blob = json.dumps(s.public())
    assert "SECRET" not in blob
    assert "a@b.com" in blob
    assert s.public()["authorized_for_plan"] is True


# --- storage ---------------------------------------------------------------
def test_session_round_trips_through_the_store(tmp_path):
    store = SessionStore(tmp_path / "auth.json")
    assert store.load() is None

    s = ChatGPTSession(client_id="oaiapp_1", access_token="t", scopes=(PLAN_SCOPE,))
    store.save(s)
    back = store.load()
    assert back is not None
    assert back.client_id == "oaiapp_1"
    assert back.authorized_for_plan

    store.clear()
    assert store.load() is None


def test_a_corrupt_token_file_means_sign_in_again_not_a_crash(tmp_path):
    p = tmp_path / "auth.json"
    p.write_text("{ not json", encoding="utf-8")
    assert SessionStore(p).load() is None


def test_host_id_is_stable_across_calls(tmp_path):
    """A new id per run would register a new agent every time and clutter the user's list."""
    store = SessionStore(tmp_path / "auth.json")
    first = store.host_id()
    assert first.startswith("urn:uuid:")
    assert store.host_id() == first
    assert SessionStore(tmp_path / "auth.json").host_id() == first


def test_refresh_without_a_refresh_token_is_a_clear_error(tmp_path):
    from drugorch.llm.chatgpt_auth import refresh
    from drugorch.llm.response_cache import LLMUnavailable

    s = ChatGPTSession(client_id="c", access_token="t", refresh_token="")
    with pytest.raises(LLMUnavailable, match="sign in again"):
        refresh(s)
