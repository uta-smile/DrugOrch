"""The run loop: determinism, termination, and the progress guard.

The determinism test is the one the paper rests on. If two runs of the same arm at the same
seed can diverge, no ablation comparing arms means anything.
"""

from __future__ import annotations

import threading

import pytest

from drugorch.budget import BudgetLedger
from drugorch.orchestrator.events import EventLog
from drugorch.orchestrator.run import Orchestrator, drive
from drugorch.orchestrator.state import RunState
from drugorch.policy import build_policy
from drugorch.subsystems.registry import Registry
from drugorch.types import TargetSpec

from .conftest import FakeSubsystem  # noqa: TID252


def make_run(registry, policy_name="fixed", budget=50.0, seed=42, max_steps=20):
    state = RunState(
        run_id=f"t_{policy_name}_{seed}",
        goal="test",
        target=TargetSpec(query="JAK2", uniprot="O60674", sequence="MMM", pdb_id="7LL4"),
        budget=BudgetLedger(total=budget),
        seed=seed,
    )
    log = EventLog(state.run_id)
    policy = build_policy(policy_name, seed=seed)
    orch = Orchestrator(state, registry, policy, log, cancel=threading.Event())
    orch.termination.max_steps = max_steps
    return orch


def fresh_registry():
    return Registry(
        [
            FakeSubsystem("retrieval", per="call", unit_cost=1.0),
            FakeSubsystem("properties", per="molecule", unit_cost=0.05),
            FakeSubsystem("docking", per="molecule", unit_cost=1.0),
            FakeSubsystem("generation", per="call", unit_cost=2.0),
        ]
    )


def decision_sequence(state) -> list[tuple]:
    return [
        (r.step, r.decision.action, r.decision.subsystem, tuple(sorted(r.decision.params.items())))
        for r in state.history
    ]


@pytest.mark.parametrize("policy", ["fixed", "scripted", "random", "roundrobin"])
def test_runs_are_reproducible_at_the_same_seed(policy):
    """Same arm, same seed, byte-identical decisions. The whole ablation depends on it."""
    a = drive(make_run(fresh_registry(), policy, seed=7))
    b = drive(make_run(fresh_registry(), policy, seed=7))
    assert decision_sequence(a) == decision_sequence(b)
    assert a.budget.spent == b.budget.spent


def test_random_policy_differs_across_seeds():
    """Otherwise the 'random' control is not actually a control."""
    a = drive(make_run(fresh_registry(), "random", seed=1))
    b = drive(make_run(fresh_registry(), "random", seed=99))
    assert decision_sequence(a) != decision_sequence(b)


def test_run_never_exceeds_its_budget_by_more_than_one_call():
    """Pre-flight refuses a call it cannot afford, so overspend is bounded."""
    orch = make_run(fresh_registry(), "scripted", budget=12.0)
    state = drive(orch)
    assert state.budget.spent <= 12.0


def test_step_is_incremental_and_cancellable():
    """A run must be stoppable between steps -- the web server depends on it."""
    orch = make_run(fresh_registry(), "fixed")
    orch.start()
    orch.step()
    assert orch.state.step == 1
    assert not orch.finished
    orch.cancel.set()
    orch.step()
    assert orch.finished
    assert orch.state.status == "cancelled"


def test_termination_reason_is_recorded():
    orch = make_run(fresh_registry(), "fixed", budget=4.0)
    drive(orch)
    assert orch.stop_reason is not None
    assert orch.stop_reason.code
    assert orch.stop_reason.reason


def test_progress_guard_allows_repeated_calls_that_advance():
    """Docking the next batch looks identical every time but is real progress.

    A guard that compared only the request would cut such a run off after three rounds.
    """
    orch = make_run(fresh_registry(), "scripted", budget=50.0, max_steps=12)
    state = drive(orch)
    docking_steps = [r for r in state.history if r.decision.subsystem == "docking" and r.executed]
    # The fake docking subsystem keeps returning records, so repeats must be tolerated.
    assert len(docking_steps) >= 3
    assert orch.stop_reason.code != "stuck"


def test_progress_guard_stops_a_policy_that_changes_nothing():
    """A subsystem that always fails must not be retried forever."""
    registry = Registry(
        [
            FakeSubsystem("retrieval", per="call", unit_cost=1.0, fail=True),
            FakeSubsystem("docking", per="molecule", unit_cost=1.0, fail=True),
        ]
    )
    orch = make_run(registry, "roundrobin", budget=100.0, max_steps=40)
    drive(orch)
    assert orch.finished
    assert orch.state.step < 40  # it gave up rather than burning every step


def test_a_policy_that_raises_falls_back_rather_than_killing_the_run():
    """Every arm gets the same safety net, or the comparison is not fair."""

    class Exploding:
        name = "exploding"

        def reset(self, state):
            pass

        def decide(self, state, manifest):
            raise RuntimeError("boom")

        def describe(self):
            return {"name": self.name}

    from drugorch.policy.base import SafePolicy
    from drugorch.policy.scripted import ScriptedPolicy

    registry = fresh_registry()
    state = RunState(
        run_id="t_explode",
        goal="test",
        target=TargetSpec(query="JAK2", uniprot="O60674", sequence="MMM"),
        budget=BudgetLedger(total=20.0),
    )
    policy = SafePolicy(Exploding(), ScriptedPolicy())
    orch = Orchestrator(state, registry, policy, EventLog("t_explode"))
    orch.termination.max_steps = 6
    drive(orch)

    assert state.history, "the run should have made decisions via the fallback"
    assert all(r.decision.source == "fallback" for r in state.history)
    assert policy.n_failures == len(state.history)


def test_events_carry_a_monotonic_sequence_and_budget():
    """Reconnection replay and the trace's x-axis both depend on this."""
    orch = make_run(fresh_registry(), "fixed")
    drive(orch)
    seqs = [e.seq for e in orch.log.events]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)
    assert all("total" in e.budget for e in orch.log.events)
