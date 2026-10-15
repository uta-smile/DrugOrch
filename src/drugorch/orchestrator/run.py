"""The run loop: one decision, one call, one step at a time.

Everything here is shaped by a single constraint -- a run must be observable *while it is
happening*. A loop that runs to completion and then hands back a result can be fast and
correct and still useless, because a routing trace that only appears after a ten-minute
docking round is not a routing trace anyone will watch.

So the unit of execution is :meth:`Orchestrator.step`, which makes exactly one routing
decision and carries out at most one subsystem call, emitting events throughout. Driving the
loop -- on a thread for the CLI, on an asyncio task for the web server -- is someone else's
job. There is deliberately no ``run_to_completion`` here.

The budget rule is enforced in one place, :meth:`_charge`, and it is the rule that makes two
runs comparable: a subsystem is charged for the unique canonical molecules it has not already
been charged for in this run. A value-cache hit still costs, because the run did not know the
answer when it asked. The cache buys wall time, never budget.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..config import Settings, get_settings
from ..subsystems.base import SubsystemResult, clamp_explain
from ..subsystems.registry import Registry
from ..types import TargetSpec
from .events import EventLog
from .state import Decision, DecisionRecord, RunState, StepResult, _trim_fact
from .termination import Termination, TerminationCheck

if TYPE_CHECKING:  # pragma: no cover
    from ..policy.base import Policy

STUCK_LIMIT = 4
"""Consecutive executed steps that change nothing before a run is declared stuck."""


class Orchestrator:
    """Drives one run: policy, subsystems, budget, events."""

    def __init__(
        self,
        state: RunState,
        registry: Registry,
        policy: Policy,
        log: EventLog,
        *,
        settings: Settings | None = None,
        termination: TerminationCheck | None = None,
        cancel: threading.Event | None = None,
    ):
        self.state = state
        self.registry = registry
        self.policy = policy
        self.log = log
        self.settings = settings or get_settings()
        self.cancel = cancel or threading.Event()
        self.termination = termination or TerminationCheck(
            max_steps=self.settings.max_steps,
            no_progress_steps=self.settings.no_progress_steps,
            wall_clock_cap_s=self.settings.wall_clock_cap_s,
        )
        self._started = False
        self._finished = False
        self.stop_reason: Termination | None = None
        # Guards against a policy that proposes the same thing forever -- a real failure
        # mode for an LLM that has latched onto one idea. See :meth:`_note_progress`.
        self._repeat_signature: tuple | None = None
        self._stuck_count = 0

    # --- event helpers ---------------------------------------------------------
    def _emit(self, type_: str, data: dict | None = None) -> None:
        self.log.emit(
            type_,  # type: ignore[arg-type]
            step=self.state.step,
            status=self.state.status,
            budget=self.state.budget.snapshot(),
            data=data or {},
        )

    # --- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.state.registry = self.registry
        self.state.status = "running"
        self.state.started_at = time.time()
        for key in self.registry.keys:
            self.state.available.setdefault(key, True)
        if hasattr(self.policy, "reset"):
            self.policy.reset(self.state)
        self._wire_llm_log()
        self._emit(
            "run_started",
            {
                "goal": self.state.goal,
                "target": self.state.target.to_dict(),
                "seed": self.state.seed,
                "policy": self.policy.describe(),
                "subsystems": self.registry.to_dict(),
                "objective": {
                    "key": self.state.objective_key,
                    "maximize": self.state.objective_maximize,
                },
            },
        )

    def _wire_llm_log(self) -> None:
        """Stream each LLM exchange as it happens, if the policy has any.

        Reached through ``SafePolicy.inner`` because the orchestrator only ever holds the
        wrapper. Duck-typed on ``on_call`` rather than on the class, so a future backend
        gets live logging by exposing the same attribute and nothing here changes.

        What goes on the wire is the summary -- step, model, latency, tokens, choice. The
        prompt and response stay in the policy and are served from ``/api/runs/{id}/llm``:
        a few thousand tokens per decision would make the event log unusable, and the one
        place that needs them is a panel the user has to open anyway.
        """
        inner = getattr(self.policy, "inner", self.policy)
        if not hasattr(inner, "on_call"):
            return

        def _report(entry: dict) -> None:
            self._emit(
                "llm_call",
                {
                    k: entry.get(k)
                    for k in (
                        "step", "cached", "model", "effort", "latency_ms", "stop_reason",
                        "action", "subsystem", "rationale", "input_tokens",
                        "output_tokens", "cache_read_input_tokens",
                    )
                },
            )

        inner.on_call = _report

    def llm_log(self) -> list[dict]:
        """Every LLM exchange this run made, full prompts and responses included."""
        inner = getattr(self.policy, "inner", self.policy)
        return list(getattr(inner, "log", []) or [])

    def llm_totals(self) -> dict:
        inner = getattr(self.policy, "inner", self.policy)
        fn = getattr(inner, "token_totals", None)
        return fn() if callable(fn) else {}

    def finish(self, termination: Termination) -> None:
        if self._finished:
            return
        self._finished = True
        self.stop_reason = termination
        cancelled = self.cancel.is_set()
        self.state.status = "cancelled" if cancelled else "done"
        self._emit(
            "run_cancelled" if cancelled else "run_finished",
            {
                "code": "cancelled" if cancelled else termination.code,
                "reason": "cancelled by user" if cancelled else termination.reason,
                "steps": self.state.step,
                "n_molecules": len(self.state.molecules),
                "best": [
                    {"smiles": s, "value": round(v, 4)} for s, v in self.state.best(20)
                ],
                "elapsed_s": round(self.state.elapsed_s, 2),
            },
        )

    @property
    def finished(self) -> bool:
        return self._finished

    # --- one step --------------------------------------------------------------
    def step(self) -> StepResult:
        """Make one routing decision and carry out at most one subsystem call."""
        if not self._started:
            self.start()

        if self.cancel.is_set():
            t = Termination(True, "cancelled", "cancelled by user")
            self.finish(t)
            return StepResult(
                decision=Decision.stop("cancelled"), executed=False, terminated=True,
                reason=t.reason,
            )

        term = self.termination.check(self.state, self.registry.keys)
        if term.stop:
            self.finish(term)
            return StepResult(
                decision=Decision.stop(term.reason), executed=False, terminated=True,
                reason=term.reason,
            )

        self.state.step += 1
        manifest = [c for c in self.registry.capabilities()]

        # --- decide ------------------------------------------------------------
        self._emit("policy_thinking", {"policy": getattr(self.policy, "name", "policy")})

        # Capture the policy-facing view *before* deciding and keep it on the record. This
        # is the explainability primitive the rest of the system rests on: a rationale says
        # what the policy claims it did, `state_seen` says what it actually had to go on,
        # and a scientist can only reconsider a decision when they can see both. Recomputed
        # rather than threaded through `decide()` so the Policy protocol stays unchanged --
        # nothing mutates the state in between, so the two are the same object's value.
        state_seen = self.state.summary()

        t0 = time.perf_counter()
        decision = self.policy.decide(self.state, manifest)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        record = DecisionRecord(step=self.state.step, decision=decision, latency_ms=latency_ms)
        record.state_seen = state_seen
        self.state.history.append(record)

        if decision.source == "fallback":
            self._emit(
                "policy_fallback",
                {"policy": decision.policy, "rationale": decision.rationale},
            )

        self._emit(
            "decision",
            {
                **decision.to_dict(),
                "latency_ms": round(latency_ms, 2),
                "llm": (decision.raw or {}).get("llm") if decision.raw else None,
            },
        )

        if decision.action == "stop":
            t = Termination(True, "policy_stop", decision.rationale or "policy chose to stop")
            self.finish(t)
            return StepResult(decision=decision, executed=False, terminated=True, reason=t.reason)

        # --- resolve the subsystem --------------------------------------------
        sub = self.registry.get(decision.subsystem or "")
        if sub is None:
            record.error = f"unknown subsystem {decision.subsystem!r}"
            self._emit("warning", {"message": record.error, "subsystem": decision.subsystem})
            return StepResult(decision=decision, executed=False, reason=record.error)

        params = sub.capability.coerce_params(decision.params)

        signature = (decision.subsystem, tuple(sorted((k, repr(v)) for k, v in params.items())))
        fingerprint_before = self._fingerprint()

        # --- preflight ---------------------------------------------------------
        usable, why = sub.preflight(params, self.state)
        if not usable:
            record.error = why
            self._emit(
                "subsystem_error",
                {"subsystem": decision.subsystem, "stage": "preflight", "reason": why},
            )
            return StepResult(decision=decision, executed=False, reason=why)

        # --- budget ------------------------------------------------------------
        est = sub.estimate_cost(params, self.state)
        if est.budget_units > self.state.budget.remaining:
            self.state.mark_unavailable(
                decision.subsystem,
                f"needs {est.budget_units:.1f} units, only "
                f"{self.state.budget.remaining:.1f} remain",
            )
            self._emit(
                "budget_block",
                {
                    "subsystem": decision.subsystem,
                    "needed": round(est.budget_units, 3),
                    "remaining": round(self.state.budget.remaining, 3),
                },
            )
            record.error = "budget_block"
            return StepResult(decision=decision, executed=False, reason="insufficient budget")

        # --- run ---------------------------------------------------------------
        self._emit(
            "call_started",
            {
                "subsystem": decision.subsystem,
                "params": params,
                "estimated_cost": est.to_dict(),
            },
        )

        def progress(message: str, fraction: float) -> None:
            self._emit(
                "call_progress",
                {
                    "subsystem": decision.subsystem,
                    "message": message,
                    "fraction": round(float(fraction), 4),
                },
            )

        t1 = time.perf_counter()
        result = sub.run(params, self.state, cancel=self.cancel, progress=progress)
        wall = time.perf_counter() - t1

        charged = self._charge(decision.subsystem, sub, params, result, est)
        self._merge(decision.subsystem, result)

        record.executed = True
        record.ok = result.ok
        record.cost_units = charged
        record.wall_s = wall
        record.error = result.error
        record.best_after = self.state.best_value

        stuck = self._note_progress(signature, fingerprint_before)

        if not result.ok:
            self._emit(
                "subsystem_error",
                {
                    "subsystem": decision.subsystem,
                    "stage": "run",
                    "reason": result.error or "unknown failure",
                },
            )

        self._emit(
            "call_finished",
            {
                "subsystem": decision.subsystem,
                "ok": result.ok,
                "wall_s": round(wall, 3),
                "cost": {**result.cost.to_dict(), "budget_units": round(charged, 4)},
                "inputs": _trim_inputs(params),
                "summary": result.summary(score_key=self.state.objective_key),
                "artifacts": result.artifacts,
                # How the call reached its result. Clamped here rather than at the source so
                # a subsystem can build its story freely and one place decides what fits on
                # the wire.
                "explain": clamp_explain(result.explain) if result.explain else {},
                "note": result.note,
                "error": result.error,
            },
        )

        if stuck is not None:
            self._emit("warning", {"message": stuck})
            self.finish(Termination(True, "stuck", stuck))
            return StepResult(
                decision=decision, executed=True, result=result, terminated=True, reason=stuck
            )

        return StepResult(decision=decision, executed=True, result=result)

    # --- progress guard --------------------------------------------------------
    def _fingerprint(self) -> tuple:
        """A cheap summary of everything a useful call could change."""
        best = self.state.best_value
        return (
            len(self.state.molecules),
            len(self.state.scored()),
            round(best, 6) if best is not None else None,
            len(self.state.facts),
        )

    def _note_progress(self, signature: tuple, before: tuple) -> str | None:
        """Detect a run that is no longer getting anywhere. Returns a reason to stop.

        The test is whether the run *advanced*, not whether the request repeated. Two
        failure modes have to be caught by one rule:

        * a policy that proposes the same thing forever -- the classic stuck LLM;
        * a policy that alternates between several actions that all change nothing, which
          an identical-request check would never notice because the request keeps changing.

        So the counter is over consecutive executed steps that left the state untouched,
        whatever was asked for. Docking the next eight molecules looks identical on paper
        every time while steadily working through the backlog; that advances the state, so
        it resets the counter and runs as long as it is useful.
        """
        if self._fingerprint() != before:
            self._stuck_count = 0
            self._repeat_signature = signature
            return None

        self._stuck_count += 1
        same = signature == self._repeat_signature
        self._repeat_signature = signature
        if self._stuck_count < STUCK_LIMIT:
            return None
        if same:
            return (
                f"policy proposed {signature[0]} with identical parameters "
                f"{self._stuck_count} times running and the run state did not change"
            )
        return (
            f"{self._stuck_count} consecutive calls changed nothing about the run state; "
            "the policy is cycling between actions that cannot make progress"
        )

    # --- budget and merge ------------------------------------------------------
    def _charge(self, key: str, sub, params: dict, result: SubsystemResult, est) -> float:
        """Apply the budget rule. The one place spend is decided.

        Charged on what the subsystem was *asked* to do, not on what it managed to return --
        a docking run that produced four poses out of eight still consumed the compute for
        eight, and a policy that could dodge the bill by failing would be learning the wrong
        lesson.
        """
        cap = sub.capability
        if cap.per == "call":
            units = cap.unit_cost
            smiles: list[str] = []
        else:
            smiles = [s for s in (params.get("smiles") or []) if s] or result.smiles
            novel = self.state.budget.novel(key, smiles)
            units = cap.unit_cost * len(novel)
            # A molecule-priced call that was asked for nothing still had an estimate; trust
            # the estimate rather than charging zero for work that happened.
            if not novel and est.budget_units > 0:
                units = est.budget_units

        self.state.budget.charge(
            key, units, step=self.state.step, smiles=smiles, note=result.note[:120]
        )
        return units

    def _lineage(self, smiles: str) -> dict:
        """The provenance of one molecule, small enough to put on the wire."""
        rec = self.state.molecules.get(smiles)
        if rec is None:
            return {"smiles": smiles}
        out: dict[str, Any] = {"smiles": smiles, "origin": rec.origin}
        if rec.parents:
            out["parents"] = list(rec.parents)
        for k in ("operator", "generation", "predicted", "assay_type", "chembl_id"):
            if rec.props.get(k) is not None:
                out[k] = rec.props[k]
        if rec.scores:
            out["scores"] = {k: round(v, 4) for k, v in rec.scores.items() if v is not None}
        return out

    def _merge(self, key: str, result: SubsystemResult) -> None:
        """Fold a result into the run state and announce what changed."""
        if result.facts:
            update = result.facts.pop("_target_update", None)
            if update:
                self.state.target = TargetSpec.from_dict(update)
            # Only what genuinely changed. Docking re-publishes its receptor fact on every
            # call, so announcing "facts present" rather than "facts changed" produced six
            # identical events in a ten-step run and taught a reader nothing.
            changed = {
                k: v for k, v in result.facts.items() if self.state.facts.get(k) != v
            }
            self.state.facts.update(result.facts)
            if changed:
                self._emit(
                    "facts_updated",
                    {
                        "subsystem": key,
                        "keys": sorted(changed),
                        "facts": {k: _trim_fact(v) for k, v in changed.items()},
                    },
                )

        if result.records:
            before = self.state.best_value
            new = self.state.add_records(result.records, origin=key)
            if new:
                self._emit(
                    "molecules_added",
                    {
                        "subsystem": key,
                        "n_new": len(new),
                        "n_total": len(self.state.molecules),
                        "sample": new[:8],
                        # Lineage, not just names. `parents` reached molecules.csv and
                        # nothing else, so the trace had to guess its dependency edges from
                        # whichever block last produced anything.
                        "records": [
                            self._lineage(s)
                            for s in new[:12]
                        ],
                    },
                )
            after = self.state.best_value
            improved = after is not None and (
                before is None
                or (after > before if self.state.objective_maximize else after < before)
            )
            if improved:
                self._emit(
                    "best_updated",
                    {
                        "objective": self.state.objective_key,
                        "best": round(after, 4),
                        "previous": round(before, 4) if before is not None else None,
                        "top": [
                            {"smiles": s, "value": round(v, 4)} for s, v in self.state.best(10)
                        ],
                    },
                )


def _trim_inputs(params: dict, max_smiles: int = 12) -> dict:
    """Keep the event payload readable when a call was handed a hundred molecules."""
    out = dict(params)
    smiles = out.get("smiles")
    if isinstance(smiles, list) and len(smiles) > max_smiles:
        out["smiles"] = smiles[:max_smiles]
        out["smiles_truncated"] = len(smiles)
    return out


def drive(
    orch: Orchestrator,
    *,
    on_step: Callable[[StepResult], None] | None = None,
    max_steps: int | None = None,
) -> RunState:
    """Step an orchestrator until it terminates. For the CLI and for tests.

    The web server does not use this -- it awaits ``step()`` on an asyncio task so the loop
    can be cancelled between steps and the event loop is never blocked.
    """
    orch.start()
    taken = 0
    while not orch.finished:
        if max_steps is not None and taken >= max_steps:
            break
        res = orch.step()
        taken += 1
        if on_step is not None:
            on_step(res)
    return orch.state
