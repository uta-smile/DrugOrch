"""The arm under test: Claude chooses which subsystem runs next.

The whole design is aimed at making this arm *comparable* to the rule-based ones rather than
advantaged over them. It therefore gets no privileged information: the prompt is built from
exactly ``RunState.summary()`` and the capability manifest, the same two things
:func:`~drugorch.policy.base.assess` reads. It also gets no privileged error handling --
every failure degrades to the scripted policy through
:class:`~drugorch.policy.base.SafePolicy`, and each such event is recorded, so a run where
the LLM silently failed half the time cannot be mistaken for one where it worked.

The response is constrained by a JSON schema generated from the registry, so the model
cannot name a subsystem that does not exist or invent a parameter. Values are clamped against
each :class:`~drugorch.subsystems.base.ParamSpec` afterwards anyway, because a schema
guarantees shape, not sense.

``considered`` is required by the schema. Asking the model to name what it rejected and why
is partly for the trace -- the UI draws those as ghost blocks -- and partly because a policy
that has to justify its non-choices is being asked a sharper question than one that only has
to announce a winner.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable
from typing import Any

from ..config import Settings, get_settings
from ..llm import build_backend
from ..llm.backend import AnthropicBackend
from ..llm.response_cache import LLMUnavailable, ResponseCache, request_key
from ..orchestrator.state import Alternative, Decision, RunState
from ..subsystems.base import Capability
from ..versions import PROMPT_V, SCHEMA_V
from .base import affordable

SYSTEM = """\
You are the routing controller for an automated drug-discovery system.

Each step you choose ONE subsystem to run next, or stop the run. You do not choose molecules
and you do not interpret chemistry results directly -- the subsystems do that. Your only job
is to decide where the remaining compute budget goes.

What matters:
- Budget is finite and subsystems differ in cost by more than an order of magnitude. Docking
  is roughly 20x the cost per molecule of the property panel and takes tens of seconds each.
- Preconditions are real. Docking needs a structure; generation works far better from known
  actives than from nothing.
- Cheap information that changes an expensive decision is usually worth buying first.
- Producing candidates you will never be able to evaluate is not progress. Near the end of a
  budget, prefer resolving what you already have.
- Stopping with budget unspent is a legitimate and sometimes correct choice. Spending the
  remainder on work that cannot change the outcome is not frugal, it is waste.

You must report the alternatives you rejected and why, not only your choice. Be specific and
quantitative in your reasoning: refer to the actual counts, costs and scores you were given.\
"""

PROMPT_NOTE = """\
This is the entire instruction the routing model receives, plus the capability manifest
below and the run state at each step. Nothing else is sent and nothing is hidden: the model
sees exactly `RunState.summary()`, which is the same view every rule-based arm is offered,
so the comparison between arms stays fair.

Editing the text here changes only the run you are about to start. It is not written to a
file, not stored in the environment, and does not affect any other run -- which is the
point: a prompt change is a change to the experiment, and an experiment whose prompt lives
in a config file cannot be told apart from the run before it.\
"""


def compose_system(extra: str = "", override: str = "") -> str:
    """The system text for one run: the standard prompt, replaced or appended to.

    Two different operations and they are not interchangeable. ``override`` replaces the
    controller instructions outright, which is the sharp tool -- a run started with a
    different system prompt is a different arm, not a tuned one. ``extra`` appends, which is
    the usual way to add a goal-specific constraint ("this target has no crystal structure;
    do not plan on docking") without discarding the budget reasoning the rest of the prompt
    carries.

    Both may be given at once and both are applied: the override becomes the base and the
    extra follows it. Dropping one silently would be the worse behaviour -- text an operator
    typed would vanish from the run with nothing anywhere saying it had been ignored. The
    web form only ever sends one, but the API accepts both and :func:`prompt_kind` names
    what happened either way.
    """
    base = override.strip() if override and override.strip() else SYSTEM
    tail = extra.strip() if extra else ""
    return f"{base}\n\n{tail}" if tail else base


def prompt_kind(extra: str = "", override: str = "") -> str:
    """How this run's instructions differ from the standard ones, in one word.

    Carried on every exchange and in the policy's provenance so a run given edited
    instructions cannot be mistaken for a standard one. Shared with the preview so the
    label on the setup page and the label in the run record are the same computation.
    """
    kinds = []
    if override and override.strip():
        kinds.append("replaced")
    if extra and extra.strip():
        kinds.append("appended")
    return "+".join(kinds) or "standard"


def _nullable(schema: dict) -> dict:
    """Allow ``null`` for a parameter that does not apply to the chosen subsystem.

    OpenAI's strict structured outputs require ``required`` to name *every* property, with
    no exceptions -- a property you simply leave out of ``required`` is a 400, not a
    tolerated optional. The sanctioned way to express "may be absent" is therefore a union
    with ``null``, which is what this does. Anthropic accepts both shapes, so one schema
    serves both providers rather than branching per backend and risking the two arms being
    asked subtly different questions.
    """
    out = dict(schema)
    t = out.get("type")
    if isinstance(t, str) and t != "null":
        out["type"] = [t, "null"]
    elif isinstance(t, list) and "null" not in t:
        out["type"] = [*t, "null"]
    # An enum has to admit null too, or the union is unsatisfiable for the absent case.
    if "enum" in out and None not in out["enum"]:
        out["enum"] = [*out["enum"], None]
    return out


def build_params_schema(manifest: list[Capability]) -> dict:
    """One params object covering every subsystem's declared parameters.

    Structured outputs want ``additionalProperties: false``, which rules out a free-form
    params object. A union of the declared parameters keeps the schema closed while letting
    the model fill in whichever ones apply to the subsystem it picked. Names are shared
    across subsystems where the meaning is shared (``smiles``), which is what we want.

    Every parameter is required *and* nullable: strict mode has no notion of optional, so
    "this one does not apply here" is expressed as an explicit null, which
    :meth:`LLMPolicy._to_decision` then drops before the subsystem ever sees it.
    """
    props: dict[str, Any] = {}
    for cap in manifest:
        for p in cap.params:
            if p.name not in props:
                props[p.name] = _nullable(p.json_schema())
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(props),
        "properties": props,
    }


def build_schema(manifest: list[Capability]) -> dict:
    keys = [c.key for c in manifest]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "subsystem", "params", "rationale", "expected_gain", "considered"],
        "properties": {
            "action": {
                "type": "string",
                "enum": ["call", "stop"],
                "description": "Run a subsystem, or end the run.",
            },
            "subsystem": {
                "type": ["string", "null"],
                "enum": [*keys, None],
                "description": "Which subsystem to run. Null when action is 'stop'.",
            },
            "params": build_params_schema(manifest),
            "rationale": {
                "type": "string",
                "description": "Why this is the best use of the next budget units. "
                "Refer to the specific numbers in the state.",
            },
            "expected_gain": {
                "type": "string",
                "description": "What you expect to learn or improve, concretely.",
            },
            "considered": {
                "type": "array",
                "description": "Every other option you weighed, including stopping.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["subsystem", "score", "reason"],
                    "properties": {
                        "subsystem": {"type": "string"},
                        "score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "How attractive this option was, 0 to 1.",
                        },
                        "reason": {"type": "string", "description": "Why you rejected it."},
                    },
                },
            },
        },
    }


def build_prompt(state: RunState, manifest: list[Capability]) -> str:
    """The volatile half of the request: this run's current situation."""
    summary = state.summary()
    affordable_now = {
        c.key: affordable(state, manifest, c.key) for c in manifest if state.is_available(c.key)
    }
    unusable = [k for k, ok in affordable_now.items() if not ok]
    lines = [
        "Current state of the run:",
        "",
        json.dumps(summary, indent=2, default=str),
        "",
    ]
    if unusable:
        lines.append(
            "These subsystems are available but cannot be afforded with the remaining "
            f"budget: {', '.join(unusable)}."
        )
        lines.append("")
    lines.append("Choose the next action.")
    return "\n".join(lines)


class LLMPolicy:
    """Routing decisions from Claude, cached and schema-constrained."""

    name = "llm"

    def __init__(
        self,
        settings: Settings | None = None,
        backend: AnthropicBackend | None = None,
        cache: ResponseCache | None = None,
        offline: bool | None = None,
        prompt_extra: str = "",
        prompt_override: str = "",
    ):
        self.settings = settings or get_settings()
        # Per-run, never persisted. See :func:`compose_system`.
        self.prompt_extra = prompt_extra or ""
        self.prompt_override = prompt_override or ""
        self.system = compose_system(self.prompt_extra, self.prompt_override)
        self.prompt_kind = prompt_kind(self.prompt_extra, self.prompt_override)
        # Which provider is behind this is the installation's choice, not the policy's: the
        # policy's job is to route, and it reads the same state and emits the same Decision
        # whichever model answers.
        self.backend = backend or build_backend(self.settings)
        self.settings.ensure_dirs()
        self.cache = cache if cache is not None else ResponseCache(self.settings.llm_cache_dir)
        self.offline = self.settings.offline if offline is None else offline
        self.log: list[dict] = []
        self.on_call: Callable[[dict], None] | None = None
        """Set by the orchestrator to stream each exchange as it happens.

        A hook rather than a protocol method because only this policy has exchanges to
        report, and widening the Policy interface for one implementation would put an empty
        method on every control arm.
        """

    def reset(self, state: RunState) -> None:
        self.log.clear()

    def _manifest_text(self, state: RunState, manifest: list[Capability]) -> str:
        """The stable half of the request: what exists and what it costs.

        Constant within a run, so it carries the cache breakpoint -- except that marking a
        subsystem unavailable changes it, which is correct: that is a real change to the
        menu and the model must see it, cache hit or not.
        """
        blocks = [c.manifest_line() for c in manifest if state.is_available(c.key)]
        gone = [
            f"- {c.key}: UNAVAILABLE for the rest of this run "
            f"({state.unavailable_reason.get(c.key, 'unknown reason')})"
            for c in manifest
            if not state.is_available(c.key)
        ]
        return "Available subsystems:\n\n" + "\n".join(blocks + gone)

    def decide(self, state: RunState, manifest: list[Capability]) -> Decision:
        schema = build_schema(manifest)
        cached_prefix = self._manifest_text(state, manifest)
        prompt = build_prompt(state, manifest)
        version = f"{PROMPT_V}/{SCHEMA_V}"
        key = request_key(self.backend.key(), self.system, cached_prefix + "\n\n" + prompt, schema, version)

        hit = self.cache.get(key)
        if hit is not None:
            payload, meta, from_cache = hit.get("response", {}), hit.get("usage", {}), True
        elif self.offline:
            raise LLMUnavailable(
                f"offline replay: no cached decision for step {state.step} (key {key})"
            )
        else:
            t0 = time.perf_counter()
            payload, meta = self.backend.complete(self.system, cached_prefix, prompt, schema)
            meta["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            from_cache = False
            self.cache.put(
                key,
                {
                    "backend": self.backend.key(),
                    "version": version,
                    "system": self.system,
                    "prompt": cached_prefix + "\n\n" + prompt,
                    "response": payload,
                    "usage": meta,
                    "step": state.step,
                },
            )

        decision = self._to_decision(payload, state, manifest)

        usage = meta.get("usage", {}) if isinstance(meta.get("usage"), dict) else {}
        entry = {
            "step": state.step,
            "cached": from_cache,
            "key": key,
            "model": meta.get("model", self.backend.model),
            "effort": meta.get("effort", self.backend.effort),
            "latency_ms": meta.get("latency_ms"),
            "stop_reason": meta.get("stop_reason"),
            "action": decision.action,
            "subsystem": decision.subsystem,
            "params": decision.params,
            "rationale": decision.rationale,
            "expected_gain": decision.expected_gain,
            "considered": [a.to_dict() for a in decision.considered],
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            # The full exchange, for the scientist who wants to see what was actually asked.
            # Served on demand, never streamed -- these are thousands of tokens each.
            "system": self.system,
            "prompt": cached_prefix + "\n\n" + prompt,
            "response": payload,
            "usage": meta,
            # So a run given edited instructions cannot read as a standard one. The cache
            # key already contains the system text, so a changed prompt also misses the
            # cache rather than replaying a decision made under different instructions.
            "prompt_customised": self.prompt_kind,
        }
        self.log.append(entry)
        if self.on_call is not None:
            # Observation must never break a run: a broken listener loses the live update,
            # not the decision.
            with contextlib.suppress(Exception):
                self.on_call(entry)
        return decision

    def _to_decision(
        self, payload: dict, state: RunState, manifest: list[Capability]
    ) -> Decision:
        """Validate and clamp a model response into a Decision.

        The schema guarantees the shape; this guarantees the sense. A subsystem that is
        unavailable, or a parameter outside its declared range, is corrected here rather
        than allowed to reach the run loop, and the correction is visible in the rationale.
        """
        action = payload.get("action", "call")
        rationale = str(payload.get("rationale", "")).strip()
        considered = tuple(
            Alternative(
                subsystem=str(a.get("subsystem", "")),
                score=float(a.get("score", 0.0)),
                reason=str(a.get("reason", "")),
            )
            for a in (payload.get("considered") or [])
            if a.get("subsystem")
        )

        if action == "stop":
            return Decision(
                action="stop",
                rationale=rationale or "model chose to stop",
                considered=considered,
                expected_gain=str(payload.get("expected_gain", "")),
                policy=self.name,
                source="llm",
                raw={"llm": payload},
            )

        key = payload.get("subsystem")
        cap = next((c for c in manifest if c.key == key), None)
        if cap is None:
            raise ValueError(f"model named an unknown subsystem {key!r}")
        if not state.is_available(cap.key):
            raise ValueError(
                f"model chose {cap.key}, which is unavailable: "
                f"{state.unavailable_reason.get(cap.key, 'unknown')}"
            )

        # Keep only the parameters this subsystem actually declares -- the schema's params
        # object is a union across all of them, so the model can legally emit a key that
        # belongs to a different subsystem.
        declared = {p.name for p in cap.params}
        # Drop nulls as well as foreign keys: strict mode makes the model emit every
        # parameter in the union, so the ones that do not apply arrive as explicit nulls
        # and would otherwise override a subsystem's own defaults with None.
        raw_params = {
            k: v
            for k, v in (payload.get("params") or {}).items()
            if k in declared and v is not None
        }
        params = cap.coerce_params(raw_params)

        return Decision(
            action="call",
            subsystem=cap.key,
            params=params,
            rationale=rationale,
            considered=considered,
            expected_gain=str(payload.get("expected_gain", "")),
            policy=self.name,
            source="llm",
            raw={"llm": payload},
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "backend": self.backend.describe(),
            "prompt_version": PROMPT_V,
            "prompt_kind": self.prompt_kind,
            "schema_version": SCHEMA_V,
            "offline": self.offline,
            "cache": self.cache.describe(),
            "n_decisions": len(self.log),
            "n_cached": sum(1 for e in self.log if e["cached"]),
            "tokens": self.token_totals(),
        }

    def token_totals(self) -> dict:
        """What this run actually cost at the API, summed over live (non-cached) calls."""

        def _sum(field: str) -> int:
            return sum(int(e.get(field) or 0) for e in self.log if not e["cached"])

        live = [e for e in self.log if not e["cached"]]
        lat = [e["latency_ms"] for e in live if e.get("latency_ms")]
        return {
            "input_tokens": _sum("input_tokens"),
            "output_tokens": _sum("output_tokens"),
            "cache_read_input_tokens": _sum("cache_read_input_tokens"),
            "cache_creation_input_tokens": _sum("cache_creation_input_tokens"),
            "n_live_calls": len(live),
            "n_cached_calls": len(self.log) - len(live),
            "total_latency_ms": round(sum(lat), 1),
            "mean_latency_ms": round(sum(lat) / len(lat), 1) if lat else None,
        }
