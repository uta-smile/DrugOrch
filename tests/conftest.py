"""Shared fixtures. Nothing here touches the network -- see ``fake_registry``."""

from __future__ import annotations

import threading

import pytest

from drugorch.budget import BudgetLedger
from drugorch.orchestrator.state import RunState
from drugorch.subsystems.base import (
    BaseSubsystem,
    Capability,
    Cost,
    ParamSpec,
    SubsystemResult,
)
from drugorch.subsystems.registry import Registry
from drugorch.types import TargetSpec

# A few real molecules, so the chemistry layers get something they can actually parse.
SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CC(=O)Nc1ccc(O)cc1",
    "N#CC[C@H](C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1",
    "OC(=O)C1=CC=CC=C1O",
    "CN1CCC[C@H]1c1cccnc1",
    "COc1ccccc1-c1ccc2cnc(Nc3ccc(N4CCN(C)CC4)cc3)nn12",
    "Cc1ccc2nc(-c3ccc(NC(=O)N4CCN(C)CC4)cc3)sc2c1",
    "CC(C)(C)OC(=O)N1CCC(N)CC1",
    "Clc1ccc(cc1)C(=O)Nc1ccccc1",
    "CCOC(=O)c1ccccc1N",
]


class FakeSubsystem(BaseSubsystem):
    """A subsystem with no dependencies, so the run loop can be tested on its own.

    Deterministic: given the same state it emits the same records, which is what lets the
    reproducibility tests isolate the orchestrator from the chemistry.
    """

    def __init__(self, key: str, *, per: str = "call", unit_cost: float = 1.0, fail: bool = False):
        self.capability = Capability(
            key=key,
            title=f"fake {key}",
            summary=f"test double for {key}",
            params=(
                ParamSpec(name="smiles", type="smiles_list", default=[], doc="molecules"),
                ParamSpec(name="n", type="int", default=6, lo=1, hi=12, doc="how many"),
            ),
            unit_cost=unit_cost,
            per=per,  # type: ignore[arg-type]
            version="test",
        )
        self.fail = fail
        self.calls = 0

    def run(self, params, state, *, cancel, progress):
        self.calls += 1
        if self.fail:
            return SubsystemResult.failure(self.capability.key, "deliberate test failure")
        progress("working", 0.5)
        n = int(params.get("n", 2))
        # Only the scoring subsystem produces the objective, as in the real system: if every
        # subsystem emitted a score, nothing would ever be 'unscored' and a policy would have
        # no reason to dock anything.
        if self.capability.key == "docking":
            targets = [s for s in (params.get("smiles") or []) if s] or state.unscored("vina_score")
            targets = targets[:n] or SMILES[:n]
            records = [
                {"smiles": s, "scores": {"vina_score": 5.0 + i + state.step * 0.1}}
                for i, s in enumerate(targets)
            ]
        elif self.capability.key == "generation":
            # A generator that returned the same molecules every round would leave the run
            # state unchanged and (correctly) trip the progress guard, so the double emits
            # genuinely new ones -- valid alkyl benzoates, unique per step.
            base = state.step * 20
            records = [
                {"smiles": "C" * (base + i + 2) + "OC(=O)c1ccccc1"} for i in range(n)
            ]
        else:
            records = [
                {"smiles": s, "props": {"panel_version": "test"}}
                for s in state.unprofiled()[:n] or SMILES[:n]
            ]

        # Retrieval publishes the facts the policies key off. Without them a policy would
        # keep asking for data it already has -- which is correct behaviour on its part, and
        # a broken test double on ours.
        facts = {}
        if self.capability.key == "retrieval":
            facts = {
                "target_identity": {"uniprot": "O60674", "gene": "JAK2"},
                "known_actives": {"n": len(records), "chembl_target": "CHEMBL2971"},
                "pdb_shortlist": [
                    {"pdb_id": "7LL4", "ligand_code": "Y5D", "score": 14.2, "reason": "test"}
                ],
            }

        return SubsystemResult(
            subsystem=self.capability.key,
            ok=True,
            records=records,
            facts=facts,
            cost=Cost(budget_units=0.0, wall_s=0.001, n_unique_smiles=len(records)),
            note=f"produced {len(records)}",
        )


@pytest.fixture
def fake_registry() -> Registry:
    return Registry(
        [
            FakeSubsystem("retrieval", per="call", unit_cost=1.0),
            FakeSubsystem("properties", per="molecule", unit_cost=0.05),
            FakeSubsystem("docking", per="molecule", unit_cost=1.0),
            FakeSubsystem("generation", per="call", unit_cost=2.0),
        ]
    )


@pytest.fixture
def state() -> RunState:
    return RunState(
        run_id="test",
        goal="test goal",
        target=TargetSpec(query="JAK2", uniprot="O60674", sequence="MMM"),
        budget=BudgetLedger(total=50.0),
        seed=42,
    )


@pytest.fixture
def cancel() -> threading.Event:
    return threading.Event()
