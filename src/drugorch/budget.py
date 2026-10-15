"""The budget ledger: the one thing a run actually spends.

A subsystem charges for *unique canonical SMILES it has not already been charged for in this
run*. Re-proposing a molecule a subsystem has already scored teaches the run nothing new, so
it is free; a molecule the run has never sent to that subsystem costs, even if some earlier
run put the answer in the value cache, because this run did not know it.

That rule is what makes two runs comparable. It is also why :class:`BudgetLedger` tracks per
subsystem rather than globally -- docking a molecule the properties panel already saw is new
information and must cost.

Non-molecular work (a database query, one GA generation) charges a flat per-call rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExhausted(RuntimeError):
    """Raised only by :meth:`BudgetLedger.charge` when ``strict`` is set."""


@dataclass
class Charge:
    """One budget-consuming event, in the order the run made them."""

    step: int
    subsystem: str
    units: float
    n_unique_smiles: int
    note: str = ""


@dataclass
class BudgetLedger:
    """Tracks what a run has spent and what it may still spend."""

    total: float
    spent: float = 0.0
    by_subsystem: dict[str, float] = field(default_factory=dict)
    charges: list[Charge] = field(default_factory=list)
    # subsystem -> canonical SMILES it has already been charged for this run
    _charged: dict[str, set[str]] = field(default_factory=dict, repr=False)

    @property
    def remaining(self) -> float:
        return max(0.0, self.total - self.spent)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0.0

    def novel(self, subsystem: str, smiles: list[str]) -> list[str]:
        """Which of these canonical SMILES this subsystem has not yet been charged for.

        Order preserved, duplicates within the batch collapsed.
        """
        seen = self._charged.setdefault(subsystem, set())
        out: list[str] = []
        local: set[str] = set()
        for s in smiles:
            if s in seen or s in local:
                continue
            local.add(s)
            out.append(s)
        return out

    def quote(
        self,
        subsystem: str,
        *,
        per_call: float = 0.0,
        per_molecule: float = 0.0,
        smiles: list[str] | None = None,
    ) -> tuple[float, list[str]]:
        """What a call *would* cost, and which molecules it would charge for.

        Used both by the policy (to judge whether an action fits the budget) and by the
        orchestrator's pre-flight check. Calling this has no effect on the ledger.
        """
        novel = self.novel(subsystem, smiles or [])
        return per_call + per_molecule * len(novel), novel

    def can_afford(self, units: float) -> bool:
        return units <= self.remaining

    def charge(
        self,
        subsystem: str,
        units: float,
        *,
        step: int = 0,
        smiles: list[str] | None = None,
        note: str = "",
        strict: bool = False,
    ) -> Charge:
        """Record a spend and mark its molecules as charged for this subsystem.

        Overspending is recorded rather than refused by default: a subsystem that ran has
        consumed real compute whether or not the estimate was right, and silently discarding
        that would make the ledger a lie. The orchestrator's pre-flight check is what keeps
        a call from starting when it cannot be afforded.
        """
        if strict and units > self.remaining:
            raise BudgetExhausted(
                f"{subsystem}: needs {units:.2f} units, {self.remaining:.2f} remaining"
            )
        novel = self.novel(subsystem, smiles or [])
        self._charged.setdefault(subsystem, set()).update(novel)
        self.spent += units
        self.by_subsystem[subsystem] = self.by_subsystem.get(subsystem, 0.0) + units
        c = Charge(step=step, subsystem=subsystem, units=units, n_unique_smiles=len(novel), note=note)
        self.charges.append(c)
        return c

    def snapshot(self) -> dict:
        """The budget block carried on every streamed event."""
        return {
            "total": round(self.total, 4),
            "spent": round(self.spent, 4),
            "remaining": round(self.remaining, 4),
            "by_subsystem": {k: round(v, 4) for k, v in sorted(self.by_subsystem.items())},
        }

    def curve(self) -> list[dict]:
        """Cumulative spend per charge -- the x-axis the routing trace is drawn against."""
        out, running = [], 0.0
        for c in self.charges:
            running += c.units
            out.append(
                {
                    "step": c.step,
                    "subsystem": c.subsystem,
                    "units": round(c.units, 4),
                    "cumulative": round(running, 4),
                    "n_unique_smiles": c.n_unique_smiles,
                }
            )
        return out

    def describe(self) -> dict:
        return {"total": self.total, "spent": round(self.spent, 4), "n_charges": len(self.charges)}
