"""The budget rule, which is what makes two runs comparable.

If these break, every cost curve in the project is meaningless, so they are worth stating
explicitly rather than leaving implicit in the orchestrator.
"""

from __future__ import annotations

from drugorch.budget import BudgetLedger


def test_repeated_molecules_are_free_within_a_subsystem():
    """Re-proposing a molecule teaches the run nothing, so it must not be charged twice."""
    b = BudgetLedger(total=100.0)
    b.charge("docking", 2.0, smiles=["A", "B"])
    assert b.spent == 2.0
    # The same two molecules again: nothing new to learn, nothing new to pay.
    assert b.novel("docking", ["A", "B"]) == []
    assert b.novel("docking", ["A", "C"]) == ["C"]


def test_subsystems_charge_independently():
    """Docking a molecule the property panel already saw is new information, so it costs."""
    b = BudgetLedger(total=100.0)
    b.charge("properties", 0.1, smiles=["A", "B"])
    assert b.novel("docking", ["A", "B"]) == ["A", "B"]


def test_duplicates_within_one_batch_collapse():
    b = BudgetLedger(total=100.0)
    assert b.novel("docking", ["A", "A", "B", "A"]) == ["A", "B"]


def test_quote_does_not_mutate_the_ledger():
    """A policy must be able to price an action without paying for it."""
    b = BudgetLedger(total=100.0)
    units, novel = b.quote("docking", per_molecule=1.0, smiles=["A", "B", "C"])
    assert units == 3.0
    assert novel == ["A", "B", "C"]
    assert b.spent == 0.0
    # Quoting twice gives the same answer -- it left no trace.
    assert b.quote("docking", per_molecule=1.0, smiles=["A", "B", "C"])[0] == 3.0


def test_curve_is_cumulative():
    b = BudgetLedger(total=100.0)
    b.charge("retrieval", 1.0, step=1)
    b.charge("properties", 5.0, step=2)
    b.charge("docking", 8.0, step=3)
    assert [c["cumulative"] for c in b.curve()] == [1.0, 6.0, 14.0]
    assert b.remaining == 86.0


def test_remaining_never_goes_negative():
    """An overspend is recorded honestly, but 'remaining' stays meaningful."""
    b = BudgetLedger(total=10.0)
    b.charge("docking", 25.0)
    assert b.spent == 25.0
    assert b.remaining == 0.0
    assert b.exhausted
