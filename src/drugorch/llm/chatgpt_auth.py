"""Sign in with ChatGPT, so a ChatGPT plan can drive the routing policy instead of an API key.

OpenAI's token-sharing preview lets a *locally hosted, open-source* app ask a user to
authorize it against their own ChatGPT plan. Eligible Responses API calls are then billed to
that plan rather than to an API organization, which is the difference between "this research
tool needs you to put a credit card on platform.openai.com" and "this research tool uses the
subscription you already pay for". DrugOrch is exactly the shape of app the preview is for:
it runs on the scientist's own machine and the source is in front of them.

The flow is OAuth 2.0 authorization code with PKCE, with one unusual step -- *dynamic client
registration*. The first sign-in is made with the well-known entry-point client id, and the
callback hands back an issued ``oaiapp_...`` id that identifies this install from then on.
That id is persisted; losing it means re-registering, not losing access.

Three details in here are not optional and are easy to get wrong:

* **The redirect URI must be ``http://127.0.0.1:<port>/auth/callback``.** Not ``localhost``,
  and not ``/callback``. Only the port may differ between sign-ins, and the exact string
  used in the authorize request must be repeated in the token exchange.
* **``chatgpt.tokens.use.direct`` must appear in the granted scopes.** A valid ID token only
  proves who signed in; it does not mean the user agreed to spend their plan. Checking the
  scope is what separates "signed in" from "authorized to run inference".
* **The ID token signature is verified against OpenAI's JWKS.** The access token would be
  rejected by the API anyway if it were forged, but an unverified ID token would let a
  hostile loopback response put any name and email into our own records.

The preview can change without notice, so every failure here raises :class:`LLMUnavailable`
with the server's own wording rather than a guess, and the policy degrades the way it does
for any other backend outage.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import http.server
import json
import os
import secrets
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .response_cache import LLMUnavailable

AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
DISCOVERY_URL = "https://auth.openai.com/.well-known/openid-configuration"
ISSUER = "https://auth.openai.com"
JWKS_URL = "https://auth.openai.com/.well-known/jwks.json"
"""Fallback only. The live value is read from the discovery document, because this is a
preview and an endpoint that moves would otherwise fail sign-in at the last step with a
signature error that looks like a security problem rather than a stale constant."""
RESOURCE = "https://api.openai.com/v1"

# The entry point for a first-time registration. It is deliberately not the id we keep: the
# callback issues a real `oaiapp_...` client id bound to this user and host, and that is
# what every later sign-in uses.
BOOTSTRAP_CLIENT_ID = "dynamic_agent_client"

SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
PLAN_SCOPE = "chatgpt.tokens.use.direct"

CALLBACK_PATH = "/auth/callback"
AGENT_NAME = "DrugOrch"

# Refresh a little before the server would reject the token, so a long docking round cannot
# straddle the expiry and fail a decision that was fine when it started.
REFRESH_MARGIN_S = 120.0


@dataclass(slots=True)
class ChatGPTSession:
    """Everything needed to keep using a plan, plus who it belongs to."""

    client_id: str
    access_token: str
    refresh_token: str = ""
    expires_at: float = 0.0
    id_token: str = ""
    account_email: str = ""
    account_name: str = ""
    scopes: tuple[str, ...] = ()
    host_id: str = ""

    @property
    def authorized_for_plan(self) -> bool:
        return PLAN_SCOPE in self.scopes

    @property
    def expired(self) -> bool:
        return bool(self.expires_at) and time.time() >= self.expires_at - REFRESH_MARGIN_S

    def to_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "id_token": self.id_token,
            "account_email": self.account_email,
            "account_name": self.account_name,
            "scopes": list(self.scopes),
            "host_id": self.host_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ChatGPTSession:
        return cls(
            client_id=d.get("client_id", ""),
            access_token=d.get("access_token", ""),
            refresh_token=d.get("refresh_token", ""),
            expires_at=float(d.get("expires_at") or 0.0),
            id_token=d.get("id_token", ""),
            account_email=d.get("account_email", ""),
            account_name=d.get("account_name", ""),
            scopes=tuple(d.get("scopes") or ()),
            host_id=d.get("host_id", ""),
        )

    def public(self) -> dict:
        """What the UI may see. Never the tokens."""
        return {
            "signed_in": bool(self.access_token),
            "account_email": self.account_email,
            "account_name": self.account_name,
            "client_id": self.client_id,
            "authorized_for_plan": self.authorized_for_plan,
            "expires_in_s": max(0, int(self.expires_at - time.time())) if self.expires_at else None,
            "scopes": list(self.scopes),
        }


# --- persistence ---------------------------------------------------------------
class SessionStore:
    """Tokens on disk, readable only by the user who signed in.

    This file is a live credential for someone's paid plan, so it is written 0600 where the
    platform honours it and kept out of the run directories that get shared alongside
    results.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> ChatGPTSession | None:
        if not self.path.is_file():
            return None
        try:
            return ChatGPTSession.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 - a corrupt token file means "sign in again"
            return None

    def save(self, session: ChatGPTSession) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")
        # Windows ACLs do not map onto POSIX modes, so this is best-effort there.
        with contextlib.suppress(OSError):
            self.path.chmod(0o600)

    def clear(self) -> None:
        if self.path.is_file():
            self.path.unlink()

    def host_id(self) -> str:
        """A stable identifier for this machine, created once and kept.

        OpenAI treats each host as a separate agent host. A value that changed per run would
        register a new agent every time and clutter the user's authorized-apps list.
        """
        p = self.path.parent / "host_id"
        if p.is_file():
            existing = p.read_text(encoding="utf-8").strip()
            if existing:
                return existing
        host = f"urn:uuid:{uuid.uuid4()}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(host, encoding="utf-8")
        return host


# --- PKCE ----------------------------------------------------------------------
def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


@dataclass(slots=True)
class Pkce:
    verifier: str
    challenge: str

    @classmethod
    def generate(cls) -> Pkce:
        verifier = _b64url(os.urandom(64))
        challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
        return cls(verifier=verifier, challenge=challenge)


# --- the loopback listener ------------------------------------------------------
class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Catches the single redirect OpenAI sends back to this machine."""

    result: dict = {}

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's interface
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != CALLBACK_PATH:
            self.send_response(404)
            self.end_headers()
            return
        params = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        type(self).result.update(params)

        ok = "error" not in params and "code" in params
        body = _CALLBACK_PAGE_OK if ok else _CALLBACK_PAGE_FAIL.format(
            error=params.get("error_description") or params.get("error") or "no code returned"
        )
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:  # noqa: D102 - silence the default logging
        return


_CALLBACK_PAGE_OK = """<!doctype html><meta charset="utf-8"><title>DrugOrch</title>
<body style="font:15px system-ui;margin:0;display:grid;place-items:center;height:100vh;
background:#0f1115;color:#e8eaed">
<div style="text-align:center;max-width:30rem;padding:2rem">
<div style="font-size:2rem;color:#36b37e">&#10003;</div>
<h1 style="font-size:1.1rem;font-weight:650">Signed in to DrugOrch</h1>
<p style="color:#9aa1ae;line-height:1.6">Your ChatGPT plan is now authorized for this
machine. You can close this tab and go back to DrugOrch.</p></div>"""

_CALLBACK_PAGE_FAIL = """<!doctype html><meta charset="utf-8"><title>DrugOrch</title>
<body style="font:15px system-ui;margin:0;display:grid;place-items:center;height:100vh;
background:#0f1115;color:#e8eaed">
<div style="text-align:center;max-width:30rem;padding:2rem">
<div style="font-size:2rem;color:#e5484d">&#10007;</div>
<h1 style="font-size:1.1rem;font-weight:650">Sign-in did not complete</h1>
<p style="color:#9aa1ae;line-height:1.6">{error}</p></div>"""


@dataclass
class LoopbackServer:
    """A one-shot HTTP listener on 127.0.0.1, started before the browser opens."""

    port: int = 0
    _httpd: Any = field(default=None, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def start(self) -> str:
        handler = type("_H", (_CallbackHandler,), {"result": {}})
        self._httpd = http.server.HTTPServer(("127.0.0.1", self.port), handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.redirect_uri

    @property
    def redirect_uri(self) -> str:
        return f"http://127.0.0.1:{self.port}{CALLBACK_PATH}"

    @property
    def result(self) -> dict:
        return dict(getattr(self._httpd.RequestHandlerClass, "result", {})) if self._httpd else {}

    def wait(self, timeout_s: float = 300.0) -> dict:
        """Block until the browser comes back, or give up."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            res = self.result
            if res:
                return res
            time.sleep(0.2)
        return {}

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


# --- the flow -------------------------------------------------------------------
def authorize_url(
    *,
    redirect_uri: str,
    host_id: str,
    pkce: Pkce,
    state: str,
    nonce: str,
    client_id: str | None = None,
    login_hint: str = "",
    id_token_hint: str = "",
) -> str:
    """Build the URL the user's browser opens.

    ``client_id=None`` means first-time registration, which is the only time
    ``agent_name_hint`` is sent -- on reauthorization it must be omitted or the request is
    treated as registering a second agent.
    """
    params: dict[str, str] = {
        "client_id": client_id or BOOTSTRAP_CLIENT_ID,
        "ext_agent_host_id": host_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": SCOPES,
        "resource": RESOURCE,
        "state": state,
        "nonce": nonce,
        "code_challenge": pkce.challenge,
        "code_challenge_method": "S256",
    }
    if client_id is None:
        params["agent_name_hint"] = AGENT_NAME
    else:
        if login_hint:
            params["login_hint"] = login_hint
        if id_token_hint:
            params["id_token_hint"] = id_token_hint
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


def exchange_code(
    *, code: str, client_id: str, redirect_uri: str, verifier: str, timeout_s: float = 30.0
) -> dict:
    """Trade the authorization code for tokens. No client secret: this is a public client."""
    try:
        r = httpx.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
                "resource": RESOURCE,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=timeout_s,
        )
    except httpx.HTTPError as exc:
        raise LLMUnavailable(f"could not reach OpenAI to exchange the code: {exc}") from exc
    if r.status_code >= 400:
        raise LLMUnavailable(f"token exchange failed ({r.status_code}): {_err(r)}")
    return r.json()


def refresh(session: ChatGPTSession, *, timeout_s: float = 30.0) -> ChatGPTSession:
    """Renew an access token. The preview makes this the app's job, not the SDK's."""
    if not session.refresh_token:
        raise LLMUnavailable("this ChatGPT session has no refresh token; sign in again")
    try:
        r = httpx.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "client_id": session.client_id,
                "refresh_token": session.refresh_token,
                "scope": SCOPES,
                "resource": RESOURCE,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=timeout_s,
        )
    except httpx.HTTPError as exc:
        raise LLMUnavailable(f"could not reach OpenAI to refresh the token: {exc}") from exc
    if r.status_code >= 400:
        raise LLMUnavailable(f"token refresh failed ({r.status_code}): {_err(r)}")
    return _session_from_tokens(r.json(), client_id=session.client_id, host_id=session.host_id,
                                previous=session)


def _err(r: httpx.Response) -> str:
    try:
        body = r.json()
        return str(body.get("error_description") or body.get("error") or body)
    except Exception:  # noqa: BLE001
        return r.text[:300]


def _session_from_tokens(
    tokens: dict, *, client_id: str, host_id: str, previous: ChatGPTSession | None = None
) -> ChatGPTSession:
    claims = _verify_id_token(tokens.get("id_token", ""), client_id) if tokens.get("id_token") else {}
    scopes = tuple((tokens.get("scope") or "").split()) or (previous.scopes if previous else ())
    return ChatGPTSession(
        client_id=client_id,
        access_token=tokens.get("access_token", ""),
        refresh_token=tokens.get("refresh_token") or (previous.refresh_token if previous else ""),
        expires_at=time.time() + float(tokens.get("expires_in") or 3600),
        id_token=tokens.get("id_token", "") or (previous.id_token if previous else ""),
        account_email=claims.get("email", "") or (previous.account_email if previous else ""),
        account_name=claims.get("name", "") or (previous.account_name if previous else ""),
        scopes=scopes,
        host_id=host_id,
    )


_jwks_cache: str = ""


def _jwks_url() -> str:
    """The signing-key endpoint, from OpenAI's discovery document.

    Cached for the process: it is one extra request on the single sign-in of a session, and
    hardcoding it is how this broke the first time.
    """
    global _jwks_cache  # noqa: PLW0603 - process-wide memo
    if _jwks_cache:
        return _jwks_cache
    try:
        r = httpx.get(DISCOVERY_URL, timeout=15)
        r.raise_for_status()
        _jwks_cache = str(r.json().get("jwks_uri") or JWKS_URL)
    except Exception:  # noqa: BLE001 - the documented value is a reasonable fallback
        _jwks_cache = JWKS_URL
    return _jwks_cache


def _verify_id_token(id_token: str, client_id: str, *, nonce: str = "") -> dict:
    """Verify the signature, issuer, audience and expiry against OpenAI's published keys.

    The access token is what actually buys inference and the API would reject a forged one,
    so this is not what stops someone spending the plan. It is what stops a hostile loopback
    response writing an arbitrary identity into the record of who authorized this install.
    """
    try:
        import jwt
        from jwt import PyJWKClient
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise LLMUnavailable("PyJWT is required to verify the ChatGPT ID token") from exc

    try:
        signing_key = PyJWKClient(_jwks_url()).get_signing_key_from_jwt(id_token)
        claims = jwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience=client_id,
            issuer=ISSUER,
            # A minute of slack against clock skew. Without it a token minted by a server
            # whose clock is a second ahead of this machine's is rejected as "not yet valid
            # (iat)" -- which presents as a failed sign-in with a correct token, and is
            # intermittent, so it reads as a network problem rather than a clock one.
            leeway=60,
            options={"verify_exp": True, "verify_iss": True, "verify_aud": True},
        )
    except Exception as exc:  # noqa: BLE001 - surfaced, never raised onward
        raise LLMUnavailable(f"the ChatGPT ID token did not verify: {exc}") from exc

    if nonce and claims.get("nonce") != nonce:
        raise LLMUnavailable("the ChatGPT ID token carried the wrong nonce")
    return claims


def sign_in(
    store: SessionStore,
    *,
    open_browser: bool = True,
    timeout_s: float = 300.0,
    on_url=None,
) -> ChatGPTSession:
    """Run the whole flow and return a usable session.

    Blocking by design: it is driven either from the CLI or from a background thread behind
    one HTTP request, and in both cases the caller is waiting for a human to click Approve.
    """
    existing = store.load()
    host = store.host_id() if not existing else (existing.host_id or store.host_id())

    server = LoopbackServer()
    redirect_uri = server.start()
    pkce = Pkce.generate()
    state = secrets.token_urlsafe(24)
    nonce = secrets.token_urlsafe(24)

    url = authorize_url(
        redirect_uri=redirect_uri,
        host_id=host,
        pkce=pkce,
        state=state,
        nonce=nonce,
        client_id=existing.client_id if existing and existing.client_id else None,
        login_hint=existing.account_email if existing else "",
        id_token_hint=existing.id_token if existing else "",
    )

    try:
        if on_url is not None:
            on_url(url)
        if open_browser:
            import webbrowser

            webbrowser.open(url)

        result = server.wait(timeout_s)
        if not result:
            raise LLMUnavailable(
                f"no response from the browser within {timeout_s:.0f}s -- sign-in cancelled"
            )
        if result.get("error"):
            raise LLMUnavailable(
                f"sign-in refused: {result.get('error_description') or result['error']}"
            )
        # Validate state before touching anything else in the response: a mismatch means
        # this callback is not the one we started, and its code must not be exchanged.
        if result.get("state") != state:
            raise LLMUnavailable("sign-in state did not match -- possible cross-site attempt")

        issued = result.get("client_id") or (existing.client_id if existing else "")
        if not issued:
            raise LLMUnavailable(
                "OpenAI did not issue a client id; registration did not complete"
            )
        if existing and existing.client_id and issued != existing.client_id:
            raise LLMUnavailable("the callback returned a different client id than requested")

        tokens = exchange_code(
            code=result["code"], client_id=issued, redirect_uri=redirect_uri, verifier=pkce.verifier
        )
    finally:
        server.stop()

    if tokens.get("id_token"):
        _verify_id_token(tokens["id_token"], issued, nonce=nonce)

    session = _session_from_tokens(tokens, client_id=issued, host_id=host)
    if not session.access_token:
        raise LLMUnavailable("OpenAI returned no access token")
    if not session.authorized_for_plan:
        # Signed in, but the plan-usage consent was not granted -- so there is identity and
        # no inference. Saying so now beats a 401 on the first routing decision.
        raise LLMUnavailable(
            "signed in, but ChatGPT plan usage was not granted. DrugOrch needs the "
            f"'{PLAN_SCOPE}' scope to run decisions on your plan."
        )
    store.save(session)
    return session


def ensure_fresh(store: SessionStore) -> ChatGPTSession | None:
    """The session to use right now, refreshed if it is close to expiring."""
    session = store.load()
    if session is None or not session.access_token:
        return None
    if not session.expired:
        return session
    refreshed = refresh(session)
    store.save(refreshed)
    return refreshed


__all__ = [
    "AGENT_NAME",
    "AUTHORIZE_URL",
    "BOOTSTRAP_CLIENT_ID",
    "PLAN_SCOPE",
    "RESOURCE",
    "SCOPES",
    "TOKEN_URL",
    "ChatGPTSession",
    "LoopbackServer",
    "Pkce",
    "SessionStore",
    "authorize_url",
    "ensure_fresh",
    "exchange_code",
    "refresh",
    "sign_in",
]
