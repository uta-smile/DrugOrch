"""The routing policy interface, and the shared scoring the baselines are built from.

A policy answers one question each step: *given what this run knows and what it has left to
spend, which subsystem should run next -- or should we stop?* It never picks a molecule; that
is the subsystem's job. Keeping those apart is what makes each a clean test of one idea, and
it is why a policy cannot accidentally win by cherry-picking answers.

Every policy sees exactly ``RunState.summary()`` and the capability manifest. Nothing else.
If the LLM arm could see more than the scripted control, a difference between them would not
be attributable to the routing.

Two hard rules:

* ``decide()`` **never raises.** :class:`SafePolicy` wraps every implementation so a bug or a
  network failure degrades to a sensible fallback decision instead of killing the run.
* ``decide()`` must populate ``Decision.considered``. The trace draws rejected alternatives
  from it, and a policy that reports only its choice is not auditable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..orchestrator.state import Alternative, Decision, RunState
from ..subsystems.base import Capability

if TYPE_CHECKING:  # pragma: no cover
    pass


@runtime_checkable
class Policy(Protocol):
    """What the orchestrator requires of a routing policy."""

    name: str

    def reset(self, state: RunState) -> None:
        """Clear per-run state. Stateful policies must implement this or they leak
        across seeds, which would silently destroy the pairing between ablation arms."""
        ...

    def decide(self, state: RunState, manifest: list[Capability]) -> Decision:
        """Choose the next action."""
        ...

    def describe(self) -> dict:
        """JSON-safe provenance blob."""
        ...


# --- shared situation assessment -----------------------------------------------
#
# Every rule-based arm reasons over the same handful of facts. Computing them in one place
# means the baselines differ in their *rules*, not in what they can see -- which is the
# whole point of a control.


def assess(state: RunState, manifest: list[Capability]) -> dict:
    """Readable facts about the run's situation, shared by every rule-based policy."""
    s = state.summary()
    mol = s["molecules"]
    obj = s["objective"]
    budget = s["budget"]
    keys = {c.key for c in manifest if state.is_available(c.key)}

    n_total = mol["total"]
    n_unprofiled = mol.get("without_properties", 0)
    n_unscored = mol.get(f"without_{state.objective_key}", 0)

    return {
        "available": keys,
        "has_identity": s["target"]["resolved"],
        "has_structure": bool(s["target"]["pdb_id"]),
        "has_actives": "known_actives" in s["facts_available"],
        "n_molecules": n_total,
        "n_unprofiled": n_unprofiled,
        "n_unscored": n_unscored,
        "best": obj["best"],
        "remaining": budget["remaining"],
        "fraction_spent": (budget["spent"] / budget["total"]) if budget["total"] else 1.0,
        "stalled_for": s["rounds_since_improvement"],
        "step": s["step"],
    }


def affordable(state: RunState, manifest: list[Capability], key: str) -> bool:
    """Whether even the cheapest form of this subsystem fits in what is left."""
    cap = next((c for c in manifest if c.key == key), None)
    if cap is None:
        return False
    floor = cap.unit_cost if cap.per == "call" else cap.unit_cost
    return state.budget.remaining >= floor


def score_unknown(
    state: RunState,
    manifest: list[Capability],
    facts: dict,
    scores: dict,
    reasons: dict,
    params: dict,
) -> None:
    """Score every available subsystem the caller had no rule for.

    The rule-based policies key their logic on the four built-in names. Without this, a
    subsystem discovered from a plugin is absent from `scores`, which means absent from
    `viable`, which means absent from `considered` -- unroutable and, worse, invisible in the
    trace. Each one is asked for its own urgency instead, so at minimum it appears and says
    why it lost.
    """
    registry = state.registry
    for cap in manifest:
        if cap.key in scores:
            continue
        if cap.key not in facts["available"] or not affordable(state, manifest, cap.key):
            continue
        sub = registry.get(cap.key) if registry is not None else None
        try:
            score, why = (
                sub.urgency(state, facts)
                if sub is not None and hasattr(sub, "urgency")
                else (0.15, "no rule for this subsystem; considered at a neutral default")
            )
        except Exception as exc:  # noqa: BLE001 - a bad plugin must not break routing
            score, why = 0.0, f"urgency() failed: {type(exc).__name__}"
        scores[cap.key] = max(0.0, min(1.0, float(score)))
        reasons[cap.key] = why
        params.setdefault(cap.key, {})


def alternatives_from_scores(
    scores: dict[str, float], reasons: dict[str, str], chosen: str | None
) -> tuple[Alternative, ...]:
    """Every option that was not chosen, strongest first.

    Reported for *all* policies, including the random one -- a random policy's "scores" are
    uniform, and showing that is informative rather than embarrassing.
    """
    items = [
        Alternative(subsystem=k, score=v, reason=reasons.get(k, ""))
        for k, v in scores.items()
        if k != chosen
    ]
    items.sort(key=lambda a: a.score, reverse=True)
    return tuple(items)


class SafePolicy:
    """Wraps a policy so a failure becomes a fallback decision, never an exception.

    The orchestrator calls this, not the policy directly. A routing experiment where one arm
    can crash the run and another cannot is not a fair comparison, so every arm gets the same
    safety net -- and when it fires, it is recorded as ``source="fallback"`` so the trace
    shows that the policy under test did not actually make that call.
    """

    def __init__(self, inner: Policy, fallback: Policy | None = None):
        self.inner = inner
        self.fallback = fallback
        self.name = getattr(inner, "name", type(inner).__name__)
        self.n_failures = 0
        self.failures: list[dict] = []

    def reset(self, state: RunState) -> None:
        self.n_failures = 0
        self.failures.clear()
        for p in (self.inner, self.fallback):
            if p is not None and hasattr(p, "reset"):
                p.reset(state)

    def decide(self, state: RunState, manifest: list[Capability]) -> Decision:
        try:
            d = self.inner.decide(state, manifest)
            if d is None:
                raise ValueError("policy returned None")
            return d
        except Exception as exc:  # noqa: BLE001 - degrading is the entire purpose
            why = f"{type(exc).__name__}: {exc}"
            self.n_failures += 1
            self.failures.append({"step": state.step, "error": why})
            if self.fallback is not None:
                try:
                    d = self.fallback.decide(state, manifest)
                    return Decision(
                        action=d.action,
                        subsystem=d.subsystem,
                        params=d.params,
                        rationale=f"[{self.name} failed: {why}] {d.rationale}",
                        considered=d.considered,
                        expected_gain=d.expected_gain,
                        policy=self.name,
                        source="fallback",
                    )
                except Exception:  # noqa: BLE001
                    pass
            return Decision.stop(
                f"policy {self.name} failed and no fallback succeeded: {why}",
                policy=self.name,
                source="fallback",
            )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "inner": self.inner.describe(),
            "fallback": self.fallback.describe() if self.fallback else None,
            "n_failures": self.n_failures,
            "failures": self.failures[:20],
        }
