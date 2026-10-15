"""Run state, and the decision type every policy speaks.

``RunState`` is the full picture; ``RunState.summary()`` is the *only* view a policy gets.
That restriction is deliberate and load-bearing: every arm in the ablation ladder -- random,
round-robin, scripted rules, the LLM -- must see byte-identical inputs, or a difference in
outcome cannot be attributed to the routing policy. Giving the LLM arm a richer view than the
scripted control would quietly invalidate the whole comparison.

The summary is also kept compact (a couple of thousand tokens) because it is re-sent on every
decision. Molecules are summarized statistically plus a small top-k, never dumped in full.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..budget import BudgetLedger
from ..types import PANEL_MARKER, MoleculeRecord, TargetSpec

RunStatus = Literal["pending", "running", "done", "cancelled", "failed"]


@dataclass(frozen=True, slots=True)
class Alternative:
    """A subsystem the policy weighed and did not pick.

    This is not decoration. The trace renders rejected alternatives as ghost blocks at the
    decision's position, and "it considered docking here and chose generation instead" is the
    single most informative thing a reader can learn from a routing trace. Every policy in
    the ladder must populate it -- rule-based ones from their own scoring -- otherwise the
    LLM arm would be the only one whose reasoning is visible, which would bias any human
    evaluation of the traces.
    """

    subsystem: str
    score: float
    reason: str = ""

    def to_dict(self) -> dict:
        return {"subsystem": self.subsystem, "score": round(self.score, 4), "reason": self.reason}


@dataclass(frozen=True, slots=True)
class Decision:
    """One routing choice: call a subsystem with these parameters, or stop."""

    action: Literal["call", "stop"]
    subsystem: str | None = None
    params: dict = field(default_factory=dict)
    rationale: str = ""
    considered: tuple[Alternative, ...] = ()
    expected_gain: str = ""
    policy: str = ""
    source: Literal["llm", "fallback", "rule"] = "rule"
    raw: dict | None = None

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "subsystem": self.subsystem,
            "params": self.params,
            "rationale": self.rationale,
            "considered": [a.to_dict() for a in self.considered],
            "expected_gain": self.expected_gain,
            "policy": self.policy,
            "source": self.source,
        }

    @classmethod
    def stop(cls, rationale: str, *, policy: str = "", source: str = "rule") -> Decision:
        return cls(action="stop", rationale=rationale, policy=policy, source=source)  # type: ignore[arg-type]


@dataclass(slots=True)
class DecisionRecord:
    """A decision plus what happened when it was carried out."""

    step: int
    decision: Decision
    latency_ms: float = 0.0
    executed: bool = False
    ok: bool | None = None
    cost_units: float = 0.0
    wall_s: float = 0.0
    error: str | None = None
    llm: dict | None = None
    state_seen: dict | None = None
    """Exactly what ``summary()`` returned when this decision was made.

    Kept out of the SSE stream -- a few KB per step would swamp the event log -- and served
    on demand instead. It is what makes a decision re-examinable months later: without it,
    reconstructing the inputs means replaying the whole run and trusting that it replayed
    identically.
    """

    best_after: float | None = None
    """Objective value once this step finished.

    Recorded per step rather than recomputed, because 'has the run improved lately' is a
    question about history and the live state only knows the present. Reading the current
    best inside a loop over past steps makes every step after the first look like a
    non-improvement, which silently ends runs that are still making progress.
    """

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            **self.decision.to_dict(),
            "latency_ms": round(self.latency_ms, 2),
            "executed": self.executed,
            "ok": self.ok,
            "cost_units": round(self.cost_units, 4),
            "wall_s": round(self.wall_s, 4),
            "error": self.error,
            "llm": self.llm,
            "best_after": self.best_after,
        }

    def to_dict_full(self) -> dict:
        """``to_dict`` plus the inputs the decision was made from."""
        return {**self.to_dict(), "state_seen": self.state_seen}


@dataclass
class RunState:
    """Everything one run knows. Mutated in place by the orchestrator."""

    run_id: str
    goal: str
    target: TargetSpec
    budget: BudgetLedger
    seed: int = 42
    step: int = 0
    status: RunStatus = "pending"
    molecules: dict[str, MoleculeRecord] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    history: list[DecisionRecord] = field(default_factory=list)
    available: dict[str, bool] = field(default_factory=dict)
    unavailable_reason: dict[str, str] = field(default_factory=dict)
    registry: Any = field(default=None, repr=False, compare=False)
    """The live registry, set by the orchestrator at start.

    Policies are given `summary()` and the capability manifest, deliberately -- it is what
    keeps every arm looking at identical inputs. This is the one exception and it is
    narrow: the rule-based arms need to ask a subsystem they have no rule for how urgent it
    considers itself, which is the only way a plugin appears in the trace at all. It is
    excluded from `summary()`, `snapshot()` and every persisted record.
    """

    holdout: frozenset[str] = field(default_factory=frozenset)
    """Canonical SMILES this run must never be shown.

    Set only by the benchmark. Retrieval filters against it, so a held-out active can reach
    the run exactly one way: the run proposed it. That is the whole basis of a rediscovery
    claim, and it is why the leak test is the single most important test in the harness.
    """

    generation: int = 0
    """How many GA generations have completed. Distinct from ``step``: a policy may call
    generation at steps 3, 7 and 11, which is three generations, not generations 3, 7 and 11."""

    objective_key: str = "vina_score"
    objective_maximize: bool = True
    """Docking scores are stored negated (see :mod:`drugorch.subsystems.docking`), so every
    score in the run is a maximization and no comparison needs a sign check."""
    started_at: float = field(default_factory=time.time)
    error: str | None = None

    # --- molecules -------------------------------------------------------------
    def add_records(self, records: list[dict], origin: str) -> list[str]:
        """Merge subsystem output into the molecule table. Returns newly-seen SMILES."""
        new: list[str] = []
        for r in records:
            smi = r.get("smiles")
            if not smi:
                continue
            rec = self.molecules.get(smi)
            if rec is None:
                rec = MoleculeRecord(
                    smiles=smi,
                    origin=origin,
                    parents=tuple(r.get("parents") or ()),
                    step_first_seen=self.step,
                )
                self.molecules[smi] = rec
                new.append(smi)
            rec.merge(props=r.get("props"), scores=r.get("scores"))
        return new

    def scored(self, key: str | None = None) -> list[tuple[str, float]]:
        """``(smiles, value)`` for molecules carrying the objective score."""
        k = key or self.objective_key
        out = [
            (s, float(rec.scores[k]))
            for s, rec in self.molecules.items()
            if rec.scores.get(k) is not None
        ]
        out.sort(key=lambda kv: kv[1], reverse=self.objective_maximize)
        return out

    def best(self, n: int = 10) -> list[tuple[str, float]]:
        return self.scored()[:n]

    @property
    def best_value(self) -> float | None:
        s = self.scored()
        return s[0][1] if s else None

    def unprofiled(self) -> list[str]:
        """Molecules with no computed property panel.

        Keyed on :data:`~drugorch.types.PANEL_MARKER` rather than on ``props`` being
        non-empty, because retrieval also writes provenance into ``props``.
        """
        return [s for s, rec in self.molecules.items() if PANEL_MARKER not in rec.props]

    def unscored(self, key: str | None = None) -> list[str]:
        k = key or self.objective_key
        return [s for s, rec in self.molecules.items() if rec.scores.get(k) is None]

    # --- availability ----------------------------------------------------------
    def mark_unavailable(self, subsystem: str, reason: str) -> None:
        """Take a subsystem out of play for the rest of the run.

        Used when a failure is structural rather than transient -- receptor preparation that
        cannot succeed for this target is not worth retrying ten times. The policy sees the
        reduced menu on its next decision and routes around it.
        """
        self.available[subsystem] = False
        self.unavailable_reason[subsystem] = reason

    def is_available(self, subsystem: str) -> bool:
        return self.available.get(subsystem, True)

    def usable_subsystems(self, all_keys: list[str]) -> list[str]:
        return [k for k in all_keys if self.is_available(k)]

    # --- progress --------------------------------------------------------------
    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    def rounds_since_improvement(self) -> int:
        """Steps since the objective last improved -- the usual signal that a run is stuck.

        Reads each step's recorded ``best_after``; see :attr:`DecisionRecord.best_after` for
        why the live best cannot be used here.
        """
        best_so_far: float | None = None
        last_improved = 0
        for rec in self.history:
            if not rec.executed or rec.best_after is None:
                continue
            v = rec.best_after
            better = (
                best_so_far is None
                or (v > best_so_far if self.objective_maximize else v < best_so_far)
            )
            if better:
                best_so_far = v
                last_improved = rec.step
        # A run that has not scored anything yet is not stalled, it is still setting up.
        if best_so_far is None:
            return 0
        return max(0, self.step - last_improved)

    def objective_curve(self) -> list[dict]:
        """Best-so-far against cumulative spend -- the score-vs-calls plot."""
        out, running = [], 0.0
        for rec in self.history:
            running += rec.cost_units
            out.append(
                {
                    "step": rec.step,
                    "spent": round(running, 4),
                    "best": rec.best_after,
                    "subsystem": rec.decision.subsystem,
                }
            )
        return out

    # --- the policy's view -----------------------------------------------------
    def summary(self, top_k: int = 8) -> dict:
        """The compact, policy-facing view. Every arm sees exactly this.

        Deliberately excludes: full molecule lists, raw sequences, previous rationales. A
        policy that could read its own earlier reasoning would drift into narrative
        consistency rather than responding to the state it is actually in.
        """
        scored = self.scored()
        prof = sum(1 for r in self.molecules.values() if PANEL_MARKER in r.props)
        return {
            "goal": self.goal,
            "step": self.step,
            "target": {
                "query": self.target.query,
                "resolved": self.target.resolved,
                "label": self.target.label,
                "uniprot": self.target.uniprot,
                "chembl_id": self.target.chembl_id,
                "pdb_id": self.target.pdb_id,
                "n_pdb_candidates": len(self.target.pdb_ids),
                "has_sequence": bool(self.target.sequence),
            },
            "budget": self.budget.snapshot(),
            "molecules": {
                "total": len(self.molecules),
                "with_properties": prof,
                "without_properties": len(self.molecules) - prof,
                f"with_{self.objective_key}": len(scored),
                f"without_{self.objective_key}": len(self.molecules) - len(scored),
                "by_origin": self._by_origin(),
            },
            "objective": {
                "key": self.objective_key,
                "maximize": self.objective_maximize,
                "best": round(scored[0][1], 4) if scored else None,
                "top": [
                    {"smiles": s, self.objective_key: round(v, 4)} for s, v in scored[:top_k]
                ],
            },
            "facts_available": sorted(self.facts),
            "unavailable": dict(self.unavailable_reason),
            "rounds_since_improvement": self.rounds_since_improvement(),
            "elapsed_s": round(self.elapsed_s, 1),
            "recent_calls": [
                {
                    "step": r.step,
                    "subsystem": r.decision.subsystem,
                    "ok": r.ok,
                    "cost_units": round(r.cost_units, 2),
                    "error": r.error,
                }
                for r in self.history[-6:]
                if r.executed
            ],
        }

    def _by_origin(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for rec in self.molecules.values():
            out[rec.origin] = out.get(rec.origin, 0) + 1
        return out

    # --- persistence -----------------------------------------------------------
    def snapshot(self) -> dict:
        """The full state the API serves and the UI renders."""
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "seed": self.seed,
            "step": self.step,
            "status": self.status,
            "target": self.target.to_dict(),
            "budget": self.budget.snapshot(),
            "budget_curve": self.budget.curve(),
            "objective_curve": self.objective_curve(),
            "objective": {"key": self.objective_key, "maximize": self.objective_maximize},
            "n_molecules": len(self.molecules),
            "best": [{"smiles": s, "value": round(v, 4)} for s, v in self.best(20)],
            "facts": {k: _trim_fact(v) for k, v in self.facts.items()},
            "available": dict(self.available),
            "unavailable_reason": dict(self.unavailable_reason),
            "history": [r.to_dict() for r in self.history],
            "elapsed_s": round(self.elapsed_s, 2),
            "error": self.error,
        }

    def describe(self) -> dict:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "seed": self.seed,
            "target": self.target.to_dict(),
            "budget": self.budget.describe(),
            "objective_key": self.objective_key,
        }


def _trim_fact(value: Any, limit: int = 4000) -> Any:
    """Keep the snapshot payload sane -- a protein sequence is not UI material."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... [{len(value)} chars total]"
    if isinstance(value, list) and len(value) > 200:
        return value[:200] + [f"... [{len(value)} items total]"]
    return value


@dataclass(slots=True)
class StepResult:
    """What one ``Orchestrator.step()`` did."""

    decision: Decision
    executed: bool
    result: Any = None
    terminated: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.to_dict(),
            "executed": self.executed,
            "terminated": self.terminated,
            "reason": self.reason,
        }


__all__ = [
    "Alternative",
    "Decision",
    "DecisionRecord",
    "RunState",
    "RunStatus",
    "StepResult",
    "asdict",
]
