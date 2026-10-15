"""The uniform interface every discovery subsystem implements.

The orchestrator never knows what a subsystem does -- only what it declares in its
:class:`Capability`, what a call would cost, and whether it is currently usable. That is what
lets the routing policy be swapped out without touching anything below, and what lets a new
subsystem be added without changing the run loop.

Two invariants the rest of the system depends on:

* ``run()`` **never raises**. A subsystem that fails returns ``ok=False`` with an ``error``
  string. An exception escaping into the run loop would kill a run that could have routed
  around the failure -- and routing around failure is the behaviour under study.
* ``run()`` is synchronous and checks ``cancel`` between units of work. The orchestrator
  executes it on a worker thread; blocking the event loop would stall every open browser.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - the cycle only matters to type checkers
    from ..orchestrator.state import RunState

NOMINAL_BATCH = 8
"""Assumed batch size when a per-molecule call does not name its molecules."""

ProgressFn = Callable[[str, float], None]
"""``(message, fraction_complete)`` -- streamed to the browser as ``call_progress``."""


@dataclass(frozen=True, slots=True)
class Cost:
    """What a call consumed, or would consume.

    ``budget_units`` is the only field the ledger cares about; the rest are for the trace and
    for working out where wall time actually goes.
    """

    budget_units: float
    wall_s: float = 0.0
    n_unique_smiles: int = 0
    api_calls: int = 0
    cache_hits: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """One tunable parameter of a subsystem.

    This single declaration drives three things: the JSON schema the LLM policy is
    constrained to, the post-parse clamping that keeps a hallucinated value in range, and the
    form the web UI renders. Declaring a parameter anywhere else means one of the three
    will drift.
    """

    name: str
    type: Literal["int", "float", "bool", "str", "enum", "smiles_list"]
    default: Any
    lo: float | None = None
    hi: float | None = None
    choices: tuple[str, ...] = ()
    doc: str = ""

    def json_schema(self) -> dict:
        """This parameter as a JSON Schema fragment."""
        base: dict[str, Any] = {"description": self.doc}
        if self.type == "int":
            base["type"] = "integer"
            if self.lo is not None:
                base["minimum"] = int(self.lo)
            if self.hi is not None:
                base["maximum"] = int(self.hi)
        elif self.type == "float":
            base["type"] = "number"
            if self.lo is not None:
                base["minimum"] = self.lo
            if self.hi is not None:
                base["maximum"] = self.hi
        elif self.type == "bool":
            base["type"] = "boolean"
        elif self.type == "enum":
            base["type"] = "string"
            base["enum"] = list(self.choices)
        elif self.type == "smiles_list":
            base["type"] = "array"
            base["items"] = {"type": "string"}
        else:
            base["type"] = "string"
        return base

    def coerce(self, value: Any) -> Any:
        """Force a value into range, falling back to the default when it cannot be read.

        Never raises. A policy that proposes nonsense gets the default and a run that keeps
        going, which is strictly more useful than a crash -- the nonsense stays visible in
        the trace either way.
        """
        try:
            if self.type == "int":
                v = int(value)
                if self.lo is not None:
                    v = max(int(self.lo), v)
                if self.hi is not None:
                    v = min(int(self.hi), v)
                return v
            if self.type == "float":
                f = float(value)
                if self.lo is not None:
                    f = max(self.lo, f)
                if self.hi is not None:
                    f = min(self.hi, f)
                return f
            if self.type == "bool":
                if isinstance(value, str):
                    return value.strip().lower() in ("true", "1", "yes", "on")
                return bool(value)
            if self.type == "enum":
                s = str(value)
                return s if s in self.choices else self.default
            if self.type == "smiles_list":
                if isinstance(value, str):
                    return [value]
                return [str(x) for x in value]
            return str(value)
        except (TypeError, ValueError):
            return self.default


@dataclass(frozen=True, slots=True)
class Capability:
    """What a subsystem advertises to the policy.

    ``summary`` is the text the LLM actually reads, so it should say what the subsystem is
    *for* and when it is worth calling -- not how it is implemented.
    """

    key: str
    title: str
    summary: str
    requires: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    params: tuple[ParamSpec, ...] = ()
    unit_cost: float = 1.0
    per: Literal["call", "molecule"] = "call"
    typical_wall_s: float = 1.0
    version: str = "v1"

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def coerce_params(self, params: dict | None) -> dict:
        """Fill in defaults, clamp what is present, drop what is not declared."""
        given = params or {}
        out: dict[str, Any] = {}
        for p in self.params:
            out[p.name] = p.coerce(given[p.name]) if p.name in given else p.default
        return out

    def params_schema(self) -> dict:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {p.name: p.json_schema() for p in self.params},
        }

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "summary": self.summary,
            "requires": list(self.requires),
            "produces": list(self.produces),
            "params": [dict(asdict(p), choices=list(p.choices)) for p in self.params],
            "unit_cost": self.unit_cost,
            "per": self.per,
            "typical_wall_s": self.typical_wall_s,
            "version": self.version,
        }

    def manifest_line(self) -> str:
        """One compact block describing this subsystem, for the policy prompt."""
        lines = [f"- {self.key}: {self.title}", f"    {self.summary}"]
        lines.append(
            f"    cost: {self.unit_cost} budget units per {self.per}; "
            f"typically ~{self.typical_wall_s:.0f}s"
        )
        if self.requires:
            lines.append(f"    requires: {', '.join(self.requires)}")
        if self.produces:
            lines.append(f"    produces: {', '.join(self.produces)}")
        for p in self.params:
            if p.lo is not None or p.hi is not None:
                rng = f" [{p.lo}..{p.hi}]"
            elif p.choices:
                rng = " {" + "|".join(p.choices) + "}"
            else:
                rng = ""
            lines.append(f"    param {p.name}: {p.type}{rng} default={p.default!r} -- {p.doc}")
        return "\n".join(lines)


EXPLAIN_MAX_ITEMS = 24
"""Longest list `explain` may carry. Enough to see the shape of a batch, short enough that
a 40-step run does not turn its event log into a molecule dump."""


def clamp_explain(blob: dict, *, max_items: int = EXPLAIN_MAX_ITEMS) -> dict:
    """Trim an ``explain`` blob so it can ride the event stream.

    Long lists are truncated and annotated with what was dropped, rather than silently cut --
    a reader who sees 24 of 200 parents should know that is what they are looking at.
    """
    out: dict[str, Any] = {}
    for key, value in blob.items():
        if isinstance(value, list) and len(value) > max_items:
            out[key] = value[:max_items]
            out[f"{key}_truncated"] = len(value) - max_items
        elif isinstance(value, dict):
            out[key] = clamp_explain(value, max_items=max_items)
        else:
            out[key] = value
    return out


@dataclass(slots=True)
class SubsystemResult:
    """What one call produced.

    ``records`` is a flat list of ``{"smiles": ..., "props": {...}, "scores": {...}}`` so the
    orchestrator can merge any subsystem's output into a
    :class:`~drugorch.types.MoleculeRecord` without knowing which subsystem produced it.
    ``facts`` carries non-molecular output -- the target sequence, the PDB shortlist -- which
    lands in ``RunState.facts``.
    """

    subsystem: str
    ok: bool
    records: list[dict] = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    explain: dict = field(default_factory=dict)
    """How this call reached its result, in the subsystem's own vocabulary.

    ``note`` is one sentence and ``records`` are the output; neither says *how*. A reader
    looking at a generation block could see "20 new molecules from 12 parents" and not learn
    which parents, on what fitness, by which operator, or how many candidates were thrown
    away -- so they could not tell a generator that is working from one that is thrashing.

    This is that missing middle. It is deliberately unstructured across subsystems: each one
    reports what matters for *it*, and the UI renders a known shape per key with a generic
    key-value fallback, so a subsystem added later is explainable without touching the
    frontend. Keep it JSON-safe and bounded -- it rides the event stream (see
    :func:`clamp_explain`).
    """

    cost: Cost = field(default_factory=lambda: Cost(0.0))
    note: str = ""
    error: str | None = None

    @property
    def smiles(self) -> list[str]:
        return [r["smiles"] for r in self.records if r.get("smiles")]

    def summary(self, top_k: int = 5, score_key: str | None = None) -> dict:
        """The compact form carried on the ``call_finished`` event."""
        out: dict[str, Any] = {"n_records": len(self.records), "n_facts": len(self.facts)}
        if self.note:
            out["note"] = self.note
        if score_key:
            scored = [
                (r["smiles"], r.get("scores", {}).get(score_key))
                for r in self.records
                if r.get("scores", {}).get(score_key) is not None
            ]
            # Every score in this system is a maximization, docking included -- its values
            # are negated at the source precisely so this holds everywhere.
            scored.sort(key=lambda kv: kv[1], reverse=True)
            out["top"] = [{"smiles": s, score_key: round(v, 4)} for s, v in scored[:top_k]]
        return out

    @classmethod
    def failure(cls, subsystem: str, error: str, cost: Cost | None = None) -> SubsystemResult:
        return cls(subsystem=subsystem, ok=False, cost=cost or Cost(0.0), error=error)


@runtime_checkable
class Subsystem(Protocol):
    """The contract. See the module docstring for the two invariants."""

    capability: Capability

    def preflight(self, params: dict, state: RunState) -> tuple[bool, str]:
        """``(usable, reason)`` -- can this be called right now, given what the run knows?

        A ``False`` here is not an error; it is the orchestrator learning that the policy
        picked something whose preconditions are unmet, which is itself worth showing.
        """
        ...

    def estimate_cost(self, params: dict, state: RunState) -> Cost:
        """What this call would cost. Must not mutate anything."""
        ...

    def run(
        self,
        params: dict,
        state: RunState,
        *,
        cancel: threading.Event,
        progress: ProgressFn,
    ) -> SubsystemResult:
        """Do the work. Never raises; checks ``cancel`` between units."""
        ...

    def describe(self) -> dict:
        """JSON-safe provenance blob, including every version that affects the output."""
        ...

    def urgency(self, state: RunState, facts: dict) -> tuple[float, str]:
        """How much this subsystem wants to run right now, 0..1, and why.

        Only the rule-based policies read this, and only for subsystems they have no
        built-in rule for. It exists because those policies score by name -- ``if
        avail("docking")`` and three more like it -- so a subsystem they have never heard of
        was never scored, never chosen, and **never appeared in the rejected alternatives
        either**. A plugin was therefore not merely unroutable but invisible in the very
        trace this project exists to produce.

        Returning a low score is fine and normal. Being absent is not.
        """
        ...


class BaseSubsystem:
    """Convenience base: sane defaults for the parts most subsystems do not override."""

    capability: Capability

    def preflight(self, params: dict, state: RunState) -> tuple[bool, str]:
        return True, ""

    def estimate_cost(self, params: dict, state: RunState) -> Cost:
        """Default estimate. Subsystems that choose their own molecules should override it.

        The empty-``smiles`` case is the one that matters. It means "work on whatever needs
        it", which is the usual way a policy calls these -- and estimating zero for it would
        let pre-flight wave through a call that then spends real budget, overshooting the
        ceiling. So an unspecified batch is priced at :data:`NOMINAL_BATCH` rather than
        nothing: an overestimate stops a run early, an underestimate breaks the budget.
        """
        c = self.capability
        if c.per == "call":
            return Cost(budget_units=c.unit_cost)
        given = list(params.get("smiles") or [])
        if given:
            novel = state.budget.novel(c.key, given)
            n = len(novel)
        else:
            n = int(params.get("max_molecules") or NOMINAL_BATCH)
        return Cost(budget_units=c.unit_cost * n, n_unique_smiles=n)

    def describe(self) -> dict:
        return {"key": self.capability.key, "version": self.capability.version}

    def urgency(self, state: RunState, facts: dict) -> tuple[float, str]:
        """Neutral by default: worth considering, never preferred over a known rule.

        Deliberately below every positive score the scripted policy assigns, so adding a
        plugin cannot change which subsystem a rule-based arm picks in situations it already
        had an opinion about -- it only becomes visible, and wins when nothing else wants to
        run. A plugin with a real opinion overrides this.
        """
        return 0.15, "no rule for this subsystem; considered at a neutral default"

    def _safe(self, fn: Callable[[], SubsystemResult]) -> SubsystemResult:
        """Run ``fn``, converting any escaped exception into a failed result."""
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - not propagating is the whole point
            return SubsystemResult.failure(self.capability.key, f"{type(exc).__name__}: {exc}")
