"""An out-of-tree subsystem, loaded from a plugin directory rather than imported.

This exists to prove the seam. It is a deliberately small tool -- structural alert counts
from RDKit's Brenk filter catalogue, which flags substructures a medicinal chemist would
avoid -- so that nothing about it distracts from the thing being demonstrated: a subsystem
living outside the package appears in the capability manifest, in the LLM's action enum, in
the routing trace with its own lane, and in the run record with its provenance, without a
single line of DrugOrch changing.

It also exercises ``urgency()``, which is what stops the rule-based policies from erasing a
subsystem they have no hand-written rule for.
"""

from __future__ import annotations

import threading
import time

from drugorch.subsystems.base import (
    BaseSubsystem,
    Capability,
    Cost,
    ParamSpec,
    ProgressFn,
    SubsystemResult,
)

VERSION = "v1"
SCORE_KEY = "structural_alerts"


def _catalog():
    """The Brenk alert set, built once. Imported lazily so a missing RDKit is a load
    failure with a readable message rather than an import crash at discovery time."""
    from rdkit.Chem import FilterCatalog

    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(params)


CAPABILITY = Capability(
    key="toxicity",
    title="Structural alert screening (Brenk)",
    summary=(
        "Count known-problematic substructures on molecules -- reactive groups, metabolic "
        "liabilities, and motifs that make a compound hard to develop. Local and nearly "
        "free. It says nothing about potency, only about whether a molecule is worth "
        "carrying forward at all."
    ),
    requires=("molecules",),
    produces=("structural_alerts",),
    params=(
        ParamSpec(
            name="smiles",
            type="smiles_list",
            default=[],
            doc="Molecules to screen. Leave empty to screen everything not yet screened.",
        ),
    ),
    unit_cost=0.02,
    per="molecule",
    typical_wall_s=0.5,
    version=VERSION,
)


class ToxicitySubsystem(BaseSubsystem):
    capability = CAPABILITY

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._cat = None

    def _targets(self, params: dict, state) -> list[str]:
        given = [s for s in (params.get("smiles") or []) if s]
        if given:
            return given
        return [
            s for s, rec in state.molecules.items() if rec.props.get(SCORE_KEY) is None
        ]

    def preflight(self, params: dict, state) -> tuple[bool, str]:
        if not self._targets(params, state):
            return False, "every molecule has already been screened"
        return True, ""

    def urgency(self, state, facts: dict) -> tuple[float, str]:
        """Cheap and informative, but never the thing that moves the objective.

        Scored above the neutral default when there is a real backlog so the policy has a
        reason to pick it, and well below docking so it cannot displace the measurement the
        run actually turns on.
        """
        n = len([s for s, rec in state.molecules.items() if rec.props.get(SCORE_KEY) is None])
        if n == 0:
            return 0.0, "every molecule has been screened"
        if n >= 10:
            return 0.45, f"{n} molecules unscreened; nearly free and filters the pool early"
        return 0.2, f"only {n} molecules unscreened; the call is barely worth its overhead"

    def run(
        self, params: dict, state, *, cancel: threading.Event, progress: ProgressFn
    ) -> SubsystemResult:
        return self._safe(lambda: self._run(params, state, cancel, progress))

    def _run(self, params: dict, state, cancel, progress) -> SubsystemResult:
        from drugorch.chem import to_mol

        t0 = time.perf_counter()
        if self._cat is None:
            self._cat = _catalog()

        targets = self._targets(params, state)
        records: list[dict] = []
        hits: dict[str, int] = {}
        clean = 0

        for i, smi in enumerate(targets):
            if cancel.is_set():
                break
            mol = to_mol(smi)
            if mol is None:
                continue
            matches = self._cat.GetMatches(mol)
            names = [m.GetDescription() for m in matches]
            for n in names:
                hits[n] = hits.get(n, 0) + 1
            if not names:
                clean += 1
            records.append(
                {
                    "smiles": smi,
                    "props": {SCORE_KEY: len(names), "alert_names": names[:6]},
                }
            )
            if i % 10 == 0:
                progress(f"{i}/{len(targets)} screened", i / max(len(targets), 1))

        worst = sorted(hits.items(), key=lambda kv: -kv[1])[:8]
        return SubsystemResult(
            subsystem=self.capability.key,
            ok=bool(records),
            records=records,
            explain={
                "catalog": "Brenk structural alerts",
                "screened": len(records),
                "clean": clean,
                "flagged": len(records) - clean,
                "most_common": [{"alert": n, "count": c} for n, c in worst],
                "rationale": (
                    f"Screened {len(records)} molecules against the Brenk alert set; "
                    f"{clean} came back clean and {len(records) - clean} carry at least one "
                    "flagged substructure."
                ),
            },
            cost=Cost(
                budget_units=0.0,
                wall_s=round(time.perf_counter() - t0, 3),
                n_unique_smiles=len(records),
            ),
            note=f"{len(records)} screened, {clean} clean, {len(records) - clean} flagged",
            error=None if records else "nothing screened",
        )

    def describe(self) -> dict:
        return {
            "key": self.capability.key,
            "version": VERSION,
            "catalog": "RDKit FilterCatalog BRENK",
        }


def build(ctx):
    """The factory every subsystem is constructed through, plugin or built-in."""
    return ToxicitySubsystem(config=ctx.config)
