"""The OpenAI Responses backend, usable with an API key or with a ChatGPT plan.

Same contract as the Anthropic backend -- one request in, one schema-valid JSON object out --
so :class:`~drugorch.policy.llm.LLMPolicy` does not know or care which provider decided. That
symmetry is the point: with two providers behind one interface, "does an LLM route better
than rules" and "does *this* LLM route better than that one" become separate, answerable
questions instead of one confounded one.

The ChatGPT-plan route is a preview with a narrow request shape, and most of the care in this
file is about respecting it:

* **``store: false`` and ``stream: true`` are mandatory.** Non-streaming is simply rejected,
  so the response is always assembled from the SSE event stream.
* **A long list of ordinary parameters must be omitted**, ``temperature`` and ``top_p``
  among them. There is no way to pin sampling, so a ChatGPT-plan arm is less reproducible
  than an API-key arm. The response cache still makes a *recorded* run replay exactly, but
  two fresh runs at the same seed can diverge -- see ``is_deterministic``.
* **``{"role": "system"}`` items are rejected.** System guidance goes in ``instructions``.
* **Prompt caching cannot be requested** (``prompt_cache_retention`` is not accepted), so the
  manifest prefix that the Anthropic path caches is simply re-sent here. Cost per decision
  is correspondingly higher, which is worth knowing before attributing a cost difference
  between arms to the routing rather than to the plumbing.

Token refresh is the app's responsibility on this route, so a 401 is retried exactly once
after refreshing, and only when a refresh callback was supplied.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from .response_cache import LLMUnavailable

RESPONSES_URL = "https://api.openai.com/v1/responses"

# Rejected outright on the ChatGPT-plan route. Listed rather than silently dropped so that a
# future change here is a deliberate edit and not an accident.
FORBIDDEN_ON_PLAN = frozenset(
    {
        "background", "conversation", "max_output_tokens", "max_tool_calls", "metadata",
        "moderation", "multi_agent", "prompt", "prompt_cache_retention", "safety_identifier",
        "temperature", "top_logprobs", "top_p", "truncation", "user", "previous_response_id",
    }
)


@dataclass
class OpenAIBackend:
    """One Responses request that must come back as schema-valid JSON.

    ``auth`` is either ``("api_key", key)`` or ``("chatgpt", access_token)``. The difference
    changes what may be sent, not how the result is read.
    """

    model: str = "gpt-5.1"
    effort: str = "medium"
    max_tokens: int = 8000
    timeout_s: float = 60.0
    api_key: str | None = None
    access_token: str | None = None
    route: str = "api_key"
    """``api_key`` or ``chatgpt_plan``. Declared by the factory rather than inferred from
    which credential happens to be present, so a signed-out ChatGPT install still reports
    itself as a plan backend with no session instead of masquerading as an API-key one."""

    on_refresh: Callable[[], str | None] | None = field(default=None, repr=False)
    """Called once on a 401 to obtain a fresh ChatGPT access token, or ``None``."""

    @property
    def uses_plan(self) -> bool:
        return self.route == "chatgpt_plan"

    @property
    def ready(self) -> bool:
        return bool(self.access_token if self.uses_plan else self.api_key)

    def key(self) -> str:
        """Identity of this backend; part of every response-cache key.

        The auth route is in the key because the plan route cannot pin sampling while the
        API-key route can. Two runs that differ in that respect must not share cached
        decisions, or a replay would quietly mix them.
        """
        return f"openai:{self.model}:effort={self.effort}:{self.route}"

    @property
    def is_deterministic(self) -> bool:
        return False

    def describe(self) -> dict:
        return {
            "backend": "openai",
            "model": self.model,
            "effort": self.effort,
            "route": self.route,
            "ready": self.ready,
            "max_tokens": self.max_tokens,
            "deterministic": self.is_deterministic,
            "prompt_caching": not self.uses_plan,
        }

    # --- request ---------------------------------------------------------------
    def _headers(self, token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    def _body(self, system: str, cached_prefix: str, prompt: str, schema: dict) -> dict:
        # The manifest is stable within a run and the state summary is not, so they stay in
        # that order even though this route cannot cache the prefix -- it keeps the prompt
        # byte-comparable with the Anthropic path, which matters when the two are compared.
        instructions = system if not cached_prefix else f"{system}\n\n{cached_prefix}"
        body: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "routing_decision",
                    "strict": True,
                    "schema": schema,
                }
            },
            "store": False,
            "stream": True,
        }
        if self.effort:
            body["reasoning"] = {"effort": self.effort}
        if not self.uses_plan:
            # Only the API-key route accepts these; on the plan route their presence is an
            # error rather than a no-op.
            body["max_output_tokens"] = self.max_tokens

        leaked = FORBIDDEN_ON_PLAN & set(body) if self.uses_plan else set()
        if leaked:  # pragma: no cover - a guard against future edits, not a runtime path
            raise LLMUnavailable(
                f"these parameters are not accepted on a ChatGPT plan: {sorted(leaked)}"
            )
        return body

    def complete(
        self, system: str, cached_prefix: str, prompt: str, schema: dict
    ) -> tuple[dict, dict]:
        """Return ``(parsed_json, metadata)`` or raise :class:`LLMUnavailable`."""
        token = self.access_token if self.uses_plan else self.api_key
        if not token:
            raise LLMUnavailable(
                "not signed in to ChatGPT -- authorize DrugOrch from the web interface"
                if self.uses_plan
                else "no OpenAI API key: set OPENAI_API_KEY, or sign in with ChatGPT"
            )

        body = self._body(system, cached_prefix, prompt, schema)
        try:
            text, meta = self._stream(body, token)
        except _Unauthorized:
            # One retry, and only with a refresher: repeating a genuinely bad credential
            # would just burn the user's time twice.
            if self.on_refresh is None:
                raise LLMUnavailable(
                    "OpenAI rejected the credential (401). Sign in with ChatGPT again."
                ) from None
            fresh = self.on_refresh()
            if not fresh:
                raise LLMUnavailable(
                    "the ChatGPT session expired and could not be refreshed; sign in again"
                ) from None
            self.access_token = fresh
            try:
                text, meta = self._stream(body, fresh)
            except _Unauthorized:
                raise LLMUnavailable(
                    "OpenAI rejected the refreshed ChatGPT credential; sign in again"
                ) from None

        if not text.strip():
            raise LLMUnavailable(f"empty response (status={meta.get('status')})")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"response was not valid JSON: {exc}") from exc
        return parsed, meta

    def _stream(self, body: dict, token: str) -> tuple[str, dict]:
        """Drive the SSE stream to ``response.completed`` and return the assembled text."""
        chunks: list[str] = []
        meta: dict[str, Any] = {"model": self.model, "effort": self.effort}
        try:
            with httpx.Client(timeout=self.timeout_s) as client, client.stream(
                "POST", RESPONSES_URL, json=body, headers=self._headers(token)
            ) as r:
                if r.status_code == 401:
                    r.read()
                    raise _Unauthorized
                if r.status_code >= 400:
                    r.read()
                    raise LLMUnavailable(
                        f"OpenAI returned {r.status_code}: {_error_text(r)}"
                    )
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        ev = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    _consume(ev, chunks, meta)
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"could not reach the OpenAI API: {exc}") from exc

        if meta.get("refusal"):
            raise LLMUnavailable(f"model declined the request: {meta['refusal']}")
        return "".join(chunks), meta


class _Unauthorized(Exception):
    """A 401, which is recoverable by refreshing exactly once."""


def _consume(ev: dict, chunks: list[str], meta: dict) -> None:
    """Fold one SSE event into the accumulating text and metadata.

    Deltas are taken when they arrive, but the completed response is preferred when it comes:
    the final object is authoritative, and relying only on deltas would silently truncate if
    the stream dropped an event.
    """
    kind = ev.get("type", "")
    if kind == "response.output_text.delta":
        chunks.append(ev.get("delta", "") or "")
    elif kind == "response.refusal.delta":
        meta["refusal"] = (meta.get("refusal") or "") + (ev.get("delta", "") or "")
    elif kind in ("response.completed", "response.incomplete"):
        resp = ev.get("response") or {}
        meta["status"] = resp.get("status")
        usage = resp.get("usage") or {}
        meta["usage"] = {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            # Named to match the Anthropic backend so the LLM log reads the same field for
            # either provider rather than special-casing per backend.
            "cache_read_input_tokens": (usage.get("input_tokens_details") or {}).get(
                "cached_tokens"
            ),
            "reasoning_tokens": (usage.get("output_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
        }
        final = _text_from_response(resp)
        if final:
            chunks.clear()
            chunks.append(final)
    elif kind == "error":
        raise LLMUnavailable(str(ev.get("message") or ev))


def _text_from_response(resp: dict) -> str:
    out = []
    for item in resp.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "output_text":
                out.append(part.get("text", ""))
            elif part.get("type") == "refusal":
                raise LLMUnavailable(f"model declined the request: {part.get('refusal')}")
    return "".join(out)


def _error_text(r: httpx.Response) -> str:
    try:
        body = r.json()
        err = body.get("error") or body
        return str(err.get("message") or err)[:400]
    except Exception:  # noqa: BLE001
        return r.text[:400]


__all__ = ["FORBIDDEN_ON_PLAN", "RESPONSES_URL", "OpenAIBackend"]
