"""When a run should stop, and for which reason.

Separated from the run loop because the reason a run ended is a result, not an implementation
detail: "stopped because the policy chose to" and "stopped because it ran out of budget" are
different outcomes, and an ablation that does not distinguish them cannot tell a policy that
knows when to quit from one that simply spent everything.
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import RunState


@dataclass(frozen=True, slots=True)
class Termination:
    """Whether to stop, and the reason, in a form the UI can show verbatim."""

    stop: bool
    code: str = ""
    reason: str = ""

    @classmethod
    def go(cls) -> Termination:
        return cls(stop=False)


class TerminationCheck:
    """Every condition that ends a run except an explicit policy ``stop``."""

    def __init__(
        self,
        *,
        max_steps: int = 40,
        no_progress_steps: int = 6,
        wall_clock_cap_s: float = 3600.0,
        min_useful_units: float = 0.05,
    ):
        self.max_steps = max_steps
        self.no_progress_steps = no_progress_steps
        self.wall_clock_cap_s = wall_clock_cap_s
        self.min_useful_units = min_useful_units

    def check(self, state: RunState, available_keys: list[str]) -> Termination:
        if state.budget.remaining <= self.min_useful_units:
            return Termination(
                True,
                "budget_exhausted",
                f"budget exhausted: {state.budget.spent:.1f} of "
                f"{state.budget.total:.1f} units spent",
            )
        if state.step >= self.max_steps:
            return Termination(
                True, "max_steps", f"reached the step cap of {self.max_steps}"
            )
        if state.elapsed_s >= self.wall_clock_cap_s:
            return Termination(
                True,
                "wall_clock",
                f"reached the wall-clock cap of {self.wall_clock_cap_s:.0f}s",
            )
        if not [k for k in available_keys if state.is_available(k)]:
            return Termination(
                True,
                "no_subsystems",
                "every subsystem has become unavailable: "
                + "; ".join(f"{k} ({v})" for k, v in state.unavailable_reason.items()),
            )
        # Only applies once the run has actually produced a score to stall on -- a run that
        # has not started scoring yet is not stuck, it is still setting up.
        if state.best_value is not None and state.rounds_since_improvement() >= self.no_progress_steps:
            return Termination(
                True,
                "no_progress",
                f"no improvement in the objective for {self.no_progress_steps} steps",
            )
        return Termination.go()

    def describe(self) -> dict:
        return {
            "max_steps": self.max_steps,
            "no_progress_steps": self.no_progress_steps,
            "wall_clock_cap_s": self.wall_clock_cap_s,
        }
