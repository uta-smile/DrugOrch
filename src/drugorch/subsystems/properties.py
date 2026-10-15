"""The RDKit property panel: cheap, local, and the first thing worth spending on.

Every number here comes from RDKit on the CPU in roughly a millisecond, which is what makes
this subsystem interesting to a routing policy: it is nearly free, so a policy that *doesn't*
call it before spending a hundred budget units on docking is leaving information on the
table. That asymmetry is exactly the kind of decision the research question is about.

Three groups of output, kept separate because they answer different questions:

* **descriptors** -- what the molecule is (mass, lipophilicity, polarity, flexibility).
* **scores** -- single numbers meant to be optimized: QED (drug-likeness, 0-1 higher better)
  and synthetic accessibility (1-10, *lower* is easier to make).
* **flags** -- rule-of-thumb filters that a medicinal chemist would apply by eye. Lipinski
  and Veber are developability heuristics, not laws; PAINS marks substructures that produce
  false positives in assays. They are reported, never enforced -- deciding what to do about
  a PAINS hit is the policy's job, not this subsystem's.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from functools import lru_cache
from typing import TYPE_CHECKING

from rdkit import RDConfig
from rdkit.Chem import QED, Crippen, Descriptors, FilterCatalog, Lipinski, rdMolDescriptors

from ..cache import ValueCache
from ..chem import canonical, to_mol
from ..types import PANEL_MARKER
from ..versions import PANEL_V, STD_V
from .base import BaseSubsystem, Capability, Cost, ParamSpec, ProgressFn, SubsystemResult

if TYPE_CHECKING:  # pragma: no cover
    from ..orchestrator.state import RunState


@lru_cache(maxsize=1)
def _sascorer():
    """RDKit ships the synthetic-accessibility scorer as a contrib script, not a module.

    It lives inside the installed wheel but is not importable without putting its directory
    on ``sys.path`` first, so the import is deferred and memoised rather than done at module
    import time.
    """
    contrib = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if contrib not in sys.path:
        sys.path.append(contrib)
    import sascorer  # type: ignore[import-not-found]

    return sascorer


@lru_cache(maxsize=1)
def _pains_catalog() -> FilterCatalog.FilterCatalog:
    params = FilterCatalog.FilterCatalogParams()
    for cat in (
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A,
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B,
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C,
    ):
        params.AddCatalog(cat)
    return FilterCatalog.FilterCatalog(params)


# Descriptor name -> callable. Declared as data so the panel can be reported, versioned and
# iterated over without the computation and the documentation drifting apart.
DESCRIPTORS = {
    "mw": Descriptors.MolWt,
    "heavy_atoms": lambda m: m.GetNumHeavyAtoms(),
    "logp": Crippen.MolLogP,
    "tpsa": rdMolDescriptors.CalcTPSA,
    "hbd": Lipinski.NumHDonors,
    "hba": Lipinski.NumHAcceptors,
    "rotatable_bonds": Lipinski.NumRotatableBonds,
    "rings": rdMolDescriptors.CalcNumRings,
    "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings,
    "fsp3": rdMolDescriptors.CalcFractionCSP3,
    "stereocenters": lambda m: rdMolDescriptors.CalcNumAtomStereoCenters(m),
    "formal_charge": lambda m: sum(a.GetFormalCharge() for a in m.GetAtoms()),
}


# What each descriptor is, in one line, and the threshold that makes it a flag. Declared as
# data because the same table has to drive three things that must not drift apart: the
# computation, the `explain` blob a reader audits, and the UI's column headers. A panel whose
# documentation lives in a docstring and whose thresholds live in a function body is one
# where a changed cutoff is invisible to everyone downstream.
DESCRIPTOR_DOC = {
    "mw": ("Molecular weight", "g/mol", "Lipinski: > 500 is a violation"),
    "heavy_atoms": ("Heavy atoms", "count", ""),
    "logp": ("Lipophilicity (Crippen cLogP)", "", "Lipinski: > 5 is a violation"),
    "tpsa": ("Topological polar surface area", "A^2", "Veber: must be <= 140"),
    "hbd": ("Hydrogen-bond donors", "count", "Lipinski: > 5 is a violation"),
    "hba": ("Hydrogen-bond acceptors", "count", "Lipinski: > 10 is a violation"),
    "rotatable_bonds": ("Rotatable bonds", "count", "Veber: must be <= 10"),
    "rings": ("Rings", "count", ""),
    "aromatic_rings": ("Aromatic rings", "count", ""),
    "fsp3": ("Fraction of sp3 carbons", "0-1", "higher is usually more three-dimensional"),
    "stereocenters": ("Atom stereocentres", "count", ""),
    "formal_charge": ("Net formal charge", "", ""),
}

SCORE_DOC = {
    "qed": ("Drug-likeness (QED)", "0-1, higher is better"),
    "sa_score": ("Synthetic accessibility", "1-10, LOWER is easier to make"),
}

FLAG_DOC = {
    "lipinski_violations": (
        "Lipinski rule-of-five violations",
        "MW > 500, cLogP > 5, HBD > 5, HBA > 10. Two or more is the usual concern line.",
    ),
    "veber_pass": (
        "Veber oral-bioavailability heuristic",
        "Rotatable bonds <= 10 and TPSA <= 140.",
    ),
    "pains_alert": (
        "PAINS substructure",
        "A motif known to produce apparent activity across unrelated assays. Reported, "
        "never enforced.",
    ),
}


def lipinski_violations(d: dict) -> int:
    """Count of Lipinski rule-of-five violations. Two or more is the usual concern line."""
    return sum(
        [d["mw"] > 500.0, d["logp"] > 5.0, d["hbd"] > 5, d["hba"] > 10]
    )


def veber_pass(d: dict) -> bool:
    """Veber's oral-bioavailability heuristic: not too floppy, not too polar."""
    return d["rotatable_bonds"] <= 10 and d["tpsa"] <= 140.0


def compute_panel(smiles: str) -> dict | None:
    """The full panel for one molecule, or ``None`` if it cannot be parsed."""
    mol = to_mol(smiles)
    if mol is None:
        return None

    desc = {}
    for name, fn in DESCRIPTORS.items():
        try:
            v = fn(mol)
        except Exception:
            v = None
        desc[name] = round(float(v), 4) if isinstance(v, (int, float)) else v

    try:
        qed = round(float(QED.qed(mol)), 4)
    except Exception:
        qed = None
    try:
        sa = round(float(_sascorer().calculateScore(mol)), 4)
    except Exception:
        sa = None

    try:
        entry = _pains_catalog().GetFirstMatch(mol)
        pains = entry.GetDescription() if entry is not None else None
    except Exception:
        pains = None

    return {
        "descriptors": desc,
        "scores": {"qed": qed, "sa_score": sa},
        "flags": {
            "lipinski_violations": lipinski_violations(desc),
            "veber_pass": veber_pass(desc),
            "pains_alert": pains,
        },
    }


def _tripped(panels: list[dict], flag: str) -> int:
    """How many molecules a developability flag caught.

    Each flag reads differently and the differences are the point: Lipinski counts
    violations and two is the line, Veber reports a pass so the failure is the negation,
    and PAINS is a description string that is either present or absent.
    """
    if flag == "lipinski_violations":
        return sum(1 for p in panels if (p.get(flag) or 0) >= 2)
    if flag == "veber_pass":
        return sum(1 for p in panels if p.get(flag) is False)
    return sum(1 for p in panels if p.get(flag))


CAPABILITY = Capability(
    key="properties",
    title="Molecular property and developability panel",
    summary=(
        "Compute a fixed panel for a set of molecules. Descriptors: molecular weight, heavy "
        "atoms, cLogP, TPSA, H-bond donors and acceptors, rotatable bonds, rings, aromatic "
        "rings, fraction sp3, stereocentres, formal charge. Scores: QED drug-likeness (0-1, "
        "higher better) and synthetic accessibility (1-10, LOWER is easier to make). Flags: "
        "Lipinski violations, Veber pass, PAINS alert. Local and almost free -- there is "
        "rarely a reason not to run this on new molecules before spending real budget on "
        "them. Says nothing about whether a molecule binds the target."
    ),
    requires=(),
    produces=("properties", "qed", "sa_score", "developability_flags"),
    params=(
        ParamSpec(
            name="smiles",
            type="smiles_list",
            default=[],
            doc="Molecules to profile. Leave empty to profile every molecule in the run "
            "that has no property panel yet.",
        ),
    ),
    unit_cost=0.05,
    per="molecule",
    typical_wall_s=0.5,
    version=f"{PANEL_V}+{STD_V}",
)


class PropertiesSubsystem(BaseSubsystem):
    """Runs :func:`compute_panel` over a batch, with a cross-run value cache."""

    capability = CAPABILITY

    def __init__(self, cache: ValueCache | None = None):
        self.cache = cache

    @property
    def cache_key(self) -> str:
        return f"properties|{self.capability.version}"

    def _targets(self, params: dict, state: RunState) -> list[str]:
        """Which molecules this call is about.

        An empty ``smiles`` list means "everything in the run that still lacks a panel",
        which is the form a policy will usually want: it does not have to track which
        molecules it has already profiled.
        """
        given = [s for s in (params.get("smiles") or []) if s]
        if given:
            return [c for c in (canonical(s) for s in given) if c]
        return [
            s
            for s, rec in state.molecules.items()
            if rec.props.get(PANEL_MARKER) != self.capability.version
        ]

    def preflight(self, params: dict, state: RunState) -> tuple[bool, str]:
        if not self._targets(params, state):
            return False, "no molecules without a property panel; nothing to compute"
        return True, ""

    def estimate_cost(self, params: dict, state: RunState) -> Cost:
        novel = state.budget.novel(self.capability.key, self._targets(params, state))
        return Cost(
            budget_units=self.capability.unit_cost * len(novel),
            n_unique_smiles=len(novel),
            wall_s=0.002 * len(novel),
        )

    def run(
        self,
        params: dict,
        state: RunState,
        *,
        cancel: threading.Event,
        progress: ProgressFn,
    ) -> SubsystemResult:
        return self._safe(lambda: self._run(params, state, cancel, progress))

    def _run(
        self, params: dict, state: RunState, cancel: threading.Event, progress: ProgressFn
    ) -> SubsystemResult:
        t0 = time.perf_counter()
        targets = self._targets(params, state)
        if not targets:
            return SubsystemResult.failure(self.capability.key, "no molecules to profile")

        # What this call was asked about, and why these molecules. An empty `smiles` means
        # "whatever still lacks a panel", which is the usual way a policy calls this -- and
        # it is also the form that made a block in the trace unreadable, because the request
        # named no molecules and the result named no molecules either.
        asked = [s for s in (params.get("smiles") or []) if s]
        selection = (
            f"the {len(targets)} molecules named in the call"
            if asked
            else f"every molecule in the run without a current panel ({len(targets)} of "
            f"{len(state.molecules)})"
        )

        cached: dict[str, tuple] = {}
        if self.cache is not None:
            cached = self.cache.get_many(self.cache_key, targets)

        records: list[dict] = []
        to_write: dict[str, tuple] = {}
        n_failed = 0
        failures: list[str] = []

        for i, smi in enumerate(targets):
            if cancel.is_set():
                break
            if smi in cached:
                panel, ok = cached[smi]
            else:
                tm = time.perf_counter()
                panel = compute_panel(smi)
                ok = panel is not None
                to_write[smi] = (panel, ok, time.perf_counter() - tm)

            if not ok or panel is None:
                n_failed += 1
                if len(failures) < 8:
                    failures.append(smi)
                continue

            records.append(
                {
                    "smiles": smi,
                    "props": {
                        **panel["descriptors"],
                        **panel["flags"],
                        PANEL_MARKER: self.capability.version,
                    },
                    "scores": panel["scores"],
                }
            )
            if i % 25 == 0 or i == len(targets) - 1:
                progress(f"profiled {i + 1}/{len(targets)}", (i + 1) / len(targets))

        if self.cache is not None and to_write:
            self.cache.put_many(self.cache_key, to_write)

        wall = time.perf_counter() - t0
        note = f"profiled {len(records)} molecules"
        if n_failed:
            note += f"; {n_failed} unparseable"
        if cancel.is_set():
            note += "; cancelled early"

        return SubsystemResult(
            subsystem=self.capability.key,
            ok=bool(records),
            records=records,
            explain=self._explain(records, selection, targets, cached, failures, note),
            cost=Cost(
                budget_units=0.0,  # the orchestrator charges; the subsystem only reports work
                wall_s=round(wall, 4),
                n_unique_smiles=len(records),
                cache_hits=len(cached),
            ),
            note=note,
            error=None if records else "every molecule failed to parse",
        )

    def _explain(
        self,
        records: list[dict],
        selection: str,
        targets: list[str],
        cached: dict,
        failures: list[str],
        note: str,
    ) -> dict:
        """The panel, named and summarised, rather than a count of molecules profiled.

        "Profiled 20 molecules" is a true sentence that leaves a reader unable to say which
        properties were looked at, what the thresholds were, or whether the answers were any
        good. So this reports the panel itself -- every descriptor with its units and the
        cutoff that turns it into a flag -- alongside the distribution each one took and how
        many molecules tripped each developability rule.
        """
        panels = [
            {**r["props"], **r["scores"], "smiles": r["smiles"]} for r in records
        ]

        def spread(key: str) -> dict | None:
            vals = [p[key] for p in panels if isinstance(p.get(key), (int, float))]
            if not vals:
                return None
            vals.sort()
            mid = len(vals) // 2
            median = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
            return {
                "min": round(float(vals[0]), 3),
                "median": round(float(median), 3),
                "max": round(float(vals[-1]), 3),
            }

        pains_hits = [p for p in panels if p.get("pains_alert")]
        return {
            "rationale": note,
            "selection": selection,
            "batch": {
                "requested": len(targets),
                "profiled": len(records),
                "from_cache": len(cached),
                "computed": len(targets) - len(cached),
                "failed": len(failures),
                "failed_examples": failures,
            },
            # The panel itself. This is the answer to "what properties are we looking at".
            "panel": {
                "descriptors": [
                    {
                        "key": k,
                        "title": DESCRIPTOR_DOC[k][0],
                        "unit": DESCRIPTOR_DOC[k][1],
                        "threshold": DESCRIPTOR_DOC[k][2],
                        "spread": spread(k),
                    }
                    for k in DESCRIPTORS
                    if k in DESCRIPTOR_DOC
                ],
                "scores": [
                    {
                        "key": k,
                        "title": SCORE_DOC[k][0],
                        "scale": SCORE_DOC[k][1],
                        "spread": spread(k),
                    }
                    for k in SCORE_DOC
                ],
                "flags": [
                    {
                        "key": k,
                        "title": FLAG_DOC[k][0],
                        "rule": FLAG_DOC[k][1],
                        "n_tripped": _tripped(panels, k),
                        "of": len(panels),
                    }
                    for k in FLAG_DOC
                ],
            },
            "pains": [
                {"smiles": p["smiles"], "alert": p["pains_alert"]} for p in pains_hits[:8]
            ],
            # A sample, best-QED first: the one ordering that makes the table worth reading
            # rather than a slice of whatever order the molecules happened to arrive in.
            "molecules": [
                {
                    "smiles": p["smiles"],
                    "qed": p.get("qed"),
                    "sa_score": p.get("sa_score"),
                    "mw": p.get("mw"),
                    "logp": p.get("logp"),
                    "tpsa": p.get("tpsa"),
                    "hbd": p.get("hbd"),
                    "hba": p.get("hba"),
                    "rotatable_bonds": p.get("rotatable_bonds"),
                    "lipinski_violations": p.get("lipinski_violations"),
                    "veber_pass": p.get("veber_pass"),
                    "pains_alert": p.get("pains_alert"),
                }
                for p in sorted(
                    panels, key=lambda p: (p.get("qed") is None, -(p.get("qed") or 0.0))
                )
            ],
        }

    def describe(self) -> dict:
        return {
            "key": self.capability.key,
            "version": self.capability.version,
            "descriptors": sorted(DESCRIPTORS),
            "scores": ["qed", "sa_score"],
            "flags": ["lipinski_violations", "veber_pass", "pains_alert"],
            "cache": self.cache.describe() if self.cache else None,
        }
