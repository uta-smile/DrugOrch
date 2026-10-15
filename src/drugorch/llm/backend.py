"""The Anthropic backend: one request, one validated JSON object back.

Every API choice here follows from Claude Opus 5.5's actual surface rather than habit:

* **Structured outputs, not forced tool use.** ``tool_choice`` of ``any``/``tool`` returns a
  400 on this model. ``output_config.format`` with a JSON schema is the supported way to get
  a guaranteed shape, and it is a better fit anyway -- we want one object, not a tool call.
* **Adaptive thinking.** ``budget_tokens`` is removed on this model and ``disabled`` is
  rejected at every effort level, so effort is the only dial. It is set explicitly because
  Opus 5.5 defaults to ``medium`` where earlier models defaulted to ``high`` -- leaving it
  implicit would make runs incomparable across a model update.
* **The beta namespace.** Server-side refusal fallbacks need ``betas`` and ``fallbacks``,
  which only ``beta.messages.create`` accepts. A routing decision is a mundane request, but
  a safety classifier firing mid-run would otherwise end it, and routing around a refusal is
  not what this experiment is measuring.
* **Prompt caching.** The system prompt and the capability manifest are identical on every
  decision in a run while the state summary changes each time, so the stable part is marked
  cacheable and the volatile part goes last. ``cache_read_input_tokens`` is reported back so
  a silent cache invalidation shows up in the run's provenance rather than only in the bill.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .response_cache import LLMUnavailable

REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class AnthropicBackend:
    """Thin wrapper over one Messages request that must return schema-valid JSON."""

    model: str = "claude-opus-5-5"
    effort: str = "medium"
    max_tokens: int = 8000
    timeout_s: float = 60.0
    api_key: str | None = None

    def key(self) -> str:
        """Identity of this backend, part of every cache key."""
        return f"anthropic:{self.model}:effort={self.effort}"

    @property
    def is_deterministic(self) -> bool:
        return False

    def _client(self):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailable("the anthropic package is not installed") from exc
        if not self.api_key:
            raise LLMUnavailable(
                "no Anthropic API key: set ANTHROPIC_API_KEY in the environment or in .env"
            )
        return anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout_s)

    def complete(self, system: str, cached_prefix: str, prompt: str, schema: dict) -> tuple[dict, dict]:
        """Return ``(parsed_json, metadata)`` or raise :class:`LLMUnavailable`.

        ``cached_prefix`` is the part of the system prompt that does not change within a run
        -- it carries the cache breakpoint. ``prompt`` is the volatile state summary.
        """
        import anthropic

        client = self._client()
        system_blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
        if cached_prefix:
            system_blocks.append(
                {
                    "type": "text",
                    "text": cached_prefix,
                    "cache_control": {"type": "ephemeral"},
                }
            )

        try:
            resp = client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
                betas=[REFUSAL_FALLBACK_BETA],
                fallbacks="default",
                system=system_blocks,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIStatusError as exc:
            raise LLMUnavailable(f"API error {exc.status_code}: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailable(f"could not reach the API: {exc}") from exc

        # Always check this before reading content -- a refusal is an HTTP 200.
        if getattr(resp, "stop_reason", None) == "refusal":
            details = getattr(resp, "stop_details", None)
            cat = getattr(details, "category", None) if details else None
            raise LLMUnavailable(f"model declined the request (category={cat})")

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        if not text.strip():
            raise LLMUnavailable(f"empty response (stop_reason={resp.stop_reason})")

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"response was not valid JSON: {exc}") from exc

        usage = getattr(resp, "usage", None)
        meta = {
            "model": self.model,
            "effort": self.effort,
            "stop_reason": resp.stop_reason,
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(
                    usage, "cache_creation_input_tokens", None
                ),
            }
            if usage
            else {},
        }
        return parsed, meta

    def describe(self) -> dict:
        return {
            "backend": "anthropic",
            "model": self.model,
            "effort": self.effort,
            "max_tokens": self.max_tokens,
            "deterministic": self.is_deterministic,
        }


def verify_api_key(key: str, *, timeout_s: float = 20.0) -> tuple[bool, str, bool]:
    """Check that a key authenticates, without spending a token on it.

    ``models.list`` is the cheapest authenticated endpoint there is: it costs nothing and
    still proves the key is live and not revoked. The alternative -- a one-token message --
    would bill for every paste of a mistyped key.

    Returns ``(ok, message, reachable)``. ``reachable`` separates "this key is wrong" from
    "the API could not be contacted", because only the first is the user's mistake: an
    offline machine or a blocking proxy must not lock someone out of entering a key they
    know is good.

    Checking at entry rather than at first use matters because the run loop treats an API
    failure as a ``policy_fallback`` and carries on under the scripted policy -- a wrong key
    would otherwise surface as a run that quietly was not the experiment it claims to be.
    """
    cleaned = (key or "").strip()
    if not cleaned:
        return False, "no key given", True
    try:
        import anthropic
    except ImportError:  # pragma: no cover - the dependency is declared
        return False, "the anthropic package is not installed", True

    try:
        client = anthropic.Anthropic(api_key=cleaned, timeout=timeout_s)
        models = client.models.list(limit=1)
    except anthropic.AuthenticationError:
        return False, "the API rejected this key (401). Check it at console.anthropic.com.", True
    except anthropic.PermissionDeniedError:
        return False, "this key authenticated but is not permitted to use the Messages API.", True
    except anthropic.APIConnectionError as exc:
        return False, f"could not reach the Anthropic API: {exc}", False
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, never raised onward
        return False, f"{type(exc).__name__}: {exc}", False

    n = len(getattr(models, "data", []) or [])
    return True, f"key verified against the Anthropic API ({n or 'no'} models listed)", True
