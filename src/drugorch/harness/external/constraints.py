"""Evaluating a molecule against a MolDesignBench constraint set.

An instance's constraints arrive already structured — property bounds and required fragment
counts — so there is no natural-language parsing here. What there *is*, is the job of mapping
their vocabulary onto RDKit exactly, and that mapping is the whole correctness risk: fuzzy
name matching gets `sulfonamide` to `fr_amide`, `thiazole ring` to `fr_azo` and `thioether`
to `fr_ether`, all of which are wrong and none of which would raise. So every name below is
written out by hand and checked against the benchmark's own reference answers.

Four of their fourteen property types — ``Mutag``, ``logD``, ``BBBP``, ``logS`` — are ADMET
predictions rather than descriptors, and they are not a minor corner: they appear in 1,427 of
the 2,000 instances, leaving only 15% evaluable from RDKit alone. They are answered by the
``admet`` subsystem through :func:`admet_predictor`, which reaches ADMET-AI in its own
virtualenv (:mod:`drugorch.admet.worker` explains why it has to live there).

When no predictor is supplied those four are reported as :data:`UNAVAILABLE` and excluded
from the denominator — never estimated, and never scored as passed or failed. A wrong value
scores exactly like a right one, so a guess would corrupt the gate and the result together
while looking like a measurement. Which constraints were skipped therefore travels with every
number produced here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rdkit.Chem import QED, Crippen, Descriptors, Fragments, Lipinski, rdMolDescriptors

from ...admet import BENCH_COLUMNS
from ...chem import to_mol

# --- properties ------------------------------------------------------------------
PROPERTY_FNS = {
    "MW": Descriptors.MolWt,
    "logP": Crippen.MolLogP,
    "MR": Crippen.MolMR,
    "TPSA": rdMolDescriptors.CalcTPSA,
    # CalcNumHBA/HBD, not Lipinski.NumHAcceptors/NumHDonors. The two disagree -- Lipinski's
    # SMARTS is the more restrictive -- and the benchmark reads these from ADMET-AI's
    # physchem module, which calls these. Using Lipinski's left two reference answers in a
    # 400-sample short by exactly one acceptor, which is what a quiet definition mismatch
    # looks like: not an error, just a systematically different number.
    "HBD": rdMolDescriptors.CalcNumHBD,
    "HBA": rdMolDescriptors.CalcNumHBA,
    "rotB": Lipinski.NumRotatableBonds,
    "rings_total": rdMolDescriptors.CalcNumRings,
    "heavy_atoms": lambda m: m.GetNumHeavyAtoms(),
    "QED": QED.qed,
}

NEEDS_PREDICTOR = frozenset({"Mutag", "logD", "BBBP", "logS"})
"""ADMET endpoints, not descriptors. Reported as unavailable, never estimated."""

UNAVAILABLE = "unavailable"


# --- fragments --------------------------------------------------------------------
#
# Written out rather than matched. The commented-out guesses are what a substring match
# produces, and each one is a silent wrong answer.
FRAGMENT_FNS = {
    "benzene ring": Fragments.fr_benzene,
    "carbonyl o": Fragments.fr_C_O,
    "amide": Fragments.fr_amide,
    "pyridine ring": Fragments.fr_pyridine,
    "aliphatic hydroxyl": Fragments.fr_Al_OH,
    "methoxy": Fragments.fr_methoxy,
    "carboxylic acid": Fragments.fr_COO,
    "ketone": Fragments.fr_ketone,
    "phenol": Fragments.fr_phenol,
    "ester": Fragments.fr_ester,
    "furan ring": Fragments.fr_furan,
    "sulfonamide": Fragments.fr_sulfonamd,      # not fr_amide
    "thiocarbonyl": Fragments.fr_C_S,
    "thioether": Fragments.fr_sulfide,          # not fr_ether
    "oxime": Fragments.fr_oxime,
    "thiazole ring": Fragments.fr_thiazole,     # not fr_azo
    "sulfone": Fragments.fr_sulfone,
    "urea": Fragments.fr_urea,
    "hydrazone": Fragments.fr_hdrzone,          # not fr_azo
    "thiophene ring": Fragments.fr_thiophene,
    "nitrile": Fragments.fr_nitrile,
    "hydroxylamine": Fragments.fr_N_O,
    "aldehyde": Fragments.fr_aldehyde,
    "amidine": Fragments.fr_amidine,
    "guanidine": Fragments.fr_guanido,
    "hydrazine": Fragments.fr_hdrzine,
    "oxazole ring": Fragments.fr_oxazole,       # not fr_azo
    "alkyl carbamate": Fragments.fr_alkyl_carbamate,
    "thiol": Fragments.fr_SH,
    "nitroso": Fragments.fr_nitroso,            # not fr_nitro
    "azo": Fragments.fr_azo,
    "beta lactam": Fragments.fr_lactam,
    "isothiocyanate": Fragments.fr_isothiocyan,
    "benzodiazepine": Fragments.fr_benzodiazepine,
    "thiocyanate": Fragments.fr_thiocyan,
    "barbiturate": Fragments.fr_barbitur,
    "diazo": Fragments.fr_diazo,                # not fr_azo
    "isocyanate": Fragments.fr_isocyan,
}


# --- the constraint set -----------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Bound:
    """One property constraint. ``min``/``max`` are inclusive and either may be absent."""

    name: str
    lo: float | None = None
    hi: float | None = None

    @property
    def computable(self) -> bool:
        return self.name in PROPERTY_FNS

    @property
    def contradictory(self) -> bool:
        """``min > max`` -- unsatisfiable by construction, and one of the ways the benchmark
        marks an instance infeasible."""
        return self.lo is not None and self.hi is not None and self.lo > self.hi

    def check(self, value: float) -> tuple[bool, float]:
        """``(satisfied, distance)``; distance is 0 when inside the bounds."""
        if self.lo is not None and value < self.lo:
            return False, float(self.lo - value)
        if self.hi is not None and value > self.hi:
            return False, float(value - self.hi)
        return True, 0.0


@dataclass(frozen=True, slots=True)
class FragmentRequirement:
    name: str
    count: int = 1
    preserve_only: bool = False
    """The group must survive the edit, not appear a specific number of times."""

    @property
    def computable(self) -> bool:
        return self.name in FRAGMENT_FNS


@dataclass(slots=True)
class Constraints:
    """One instance's requirements."""

    bounds: list[Bound] = field(default_factory=list)
    fragments: list[FragmentRequirement] = field(default_factory=list)
    instance_id: str = ""
    task_type: str = ""

    @classmethod
    def from_answer(
        cls,
        answer: dict,
        *,
        instance_id: str = "",
        task_type: str = "",
        meta: dict | None = None,
    ) -> Constraints:
        """Build a constraint set from an instance.

        ``meta`` matters for optimization tasks. There, ``answer.fragments`` carries the
        *seed molecule's* group counts while ``optimization_goal.preserve_fgs`` names the
        groups that must survive the edit -- and surviving means still being present, not
        being present the same number of times. Demanding the seed's exact count marked 185
        of the benchmark's own reference answers as failures; honouring the distinction
        takes that to near zero.
        """
        bounds = [
            Bound(name=c["property"], lo=c.get("min"), hi=c.get("max"))
            for c in (answer.get("properties") or [])
            if c.get("property")
        ]

        preserve = set()
        if meta:
            goal = meta.get("optimization_goal") or {}
            preserve = {str(g) for g in (goal.get("preserve_fgs") or [])}

        frags: list[FragmentRequirement] = []
        for entry in answer.get("fragments") or []:
            if not isinstance(entry, dict):
                continue
            for k, v in entry.items():
                count = 1 if k in preserve else int(v)
                frags.append(FragmentRequirement(name=k, count=count, preserve_only=k in preserve))
        return cls(bounds=bounds, fragments=frags, instance_id=instance_id, task_type=task_type)

    @property
    def total(self) -> int:
        return len(self.bounds) + len(self.fragments)

    @property
    def n_uncomputable(self) -> int:
        return sum(1 for b in self.bounds if not b.computable) + sum(
            1 for f in self.fragments if not f.computable
        )

    @property
    def fully_computable(self) -> bool:
        return self.n_uncomputable == 0

    def contradictions(self) -> list[str]:
        """Why this instance is impossible, if it is.

        Only *structural* impossibility is claimed -- bounds that cannot both hold. A
        constraint set that is merely very hard is not reported here, because guessing at
        that is how an agent talks itself out of a solvable problem.
        """
        out = [f"{b.name}: min {b.lo} > max {b.hi}" for b in self.bounds if b.contradictory]
        # The same property constrained twice with disjoint windows.
        by_name: dict[str, list[Bound]] = {}
        for b in self.bounds:
            by_name.setdefault(b.name, []).append(b)
        for name, bs in by_name.items():
            if len(bs) < 2:
                continue
            lo = max((b.lo for b in bs if b.lo is not None), default=None)
            hi = min((b.hi for b in bs if b.hi is not None), default=None)
            if lo is not None and hi is not None and lo > hi:
                out.append(f"{name}: combined window [{lo}, {hi}] is empty")
        return out

    @property
    def needs_predictor(self) -> bool:
        """Whether any constraint here requires an ADMET prediction.

        Checked before a batch is sent, so an instance needing nothing from ADMET-AI never
        wakes the worker. This is broader than :data:`NEEDS_PREDICTOR`: six of these are
        descriptors we *could* compute, but computing them here would use a different
        RDKit than the answer key — see :data:`drugorch.admet.BENCH_COLUMNS`.
        """
        return any(b.name in BENCH_COLUMNS for b in self.bounds)

    @property
    def infeasible(self) -> bool:
        return bool(self.contradictions())


# --- evaluation ---------------------------------------------------------------------
@dataclass(slots=True)
class Evaluation:
    """How one molecule measures against one constraint set."""

    smiles: str
    satisfied: int = 0
    evaluated: int = 0
    skipped: int = 0
    total_distance: float = 0.0
    details: list[dict] = field(default_factory=list)
    valid: bool = True

    @property
    def csr(self) -> float:
        """Constraint satisfaction rate over the constraints we could actually check.

        Denominator is ``evaluated``, not ``total``: scoring an unavailable ADMET constraint
        as failed would punish the molecule for our missing predictor, and scoring it as
        passed would reward it for the same. Neither is a measurement. How many were skipped
        travels alongside so the number is never read without it.
        """
        return self.satisfied / self.evaluated if self.evaluated else 0.0

    @property
    def mcd(self) -> float:
        """Mean constraint distance over the ones that were checked."""
        return self.total_distance / self.evaluated if self.evaluated else 0.0

    @property
    def all_satisfied(self) -> bool:
        return self.evaluated > 0 and self.satisfied == self.evaluated

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "csr": round(self.csr, 4),
            "mcd": round(self.mcd, 4),
            "satisfied": self.satisfied,
            "evaluated": self.evaluated,
            "skipped": self.skipped,
            "all_satisfied": self.all_satisfied,
            "valid": self.valid,
        }


def evaluate(
    smiles: str,
    constraints: Constraints,
    *,
    mol: Any = None,
    predictions: dict | None = None,
) -> Evaluation:
    """Measure one molecule against one constraint set.

    ``predictions`` is this molecule's ADMET panel, as the ``admet`` subsystem produces it.
    Supplying it turns the four predicted properties from skipped into evaluated; omitting it
    leaves them skipped, which is the honest state when no predictor is reachable. Nothing
    here estimates them.
    """
    m = mol if mol is not None else to_mol(smiles)
    if m is None:
        return Evaluation(smiles=smiles, valid=False, skipped=constraints.total)

    ev = Evaluation(smiles=smiles)

    for b in constraints.bounds:
        value, source = None, "local"
        supplied = (predictions or {}).get(BENCH_COLUMNS.get(b.name, ""))
        if isinstance(supplied, (int, float)):
            value, source = float(supplied), "benchmark"
        elif b.name in NEEDS_PREDICTOR or not b.computable:
            ev.skipped += 1
            ev.details.append({"constraint": b.name, "status": UNAVAILABLE})
            continue
        else:
            try:
                value = float(PROPERTY_FNS[b.name](m))
            except Exception:  # noqa: BLE001 - a failed descriptor is not a failed molecule
                ev.skipped += 1
                ev.details.append({"constraint": b.name, "status": "error"})
                continue

        # Rounded to two decimals before the comparison because that is what the benchmark
        # compares. Their analysis tool rounds every property it reports, and one reference
        # answer sits exactly on a boundary at logP 2.0001 against `max: 2.0` -- correct at
        # their precision, a failure at full precision, and the difference is the answer key
        # rather than the chemistry.
        value = round(value, 2)
        ok, dist = b.check(value)
        ev.evaluated += 1
        ev.satisfied += int(ok)
        ev.total_distance += dist
        ev.details.append(
            {
                "constraint": b.name,
                "value": value,
                "min": b.lo,
                "max": b.hi,
                "ok": ok,
                "distance": round(dist, 3),
                "source": source,
            }
        )

    for f in constraints.fragments:
        if not f.computable:
            ev.skipped += 1
            ev.details.append({"constraint": f.name, "status": UNAVAILABLE})
            continue
        try:
            found = int(FRAGMENT_FNS[f.name](m))
        except Exception:  # noqa: BLE001
            ev.skipped += 1
            ev.details.append({"constraint": f.name, "status": "error"})
            continue
        ok = found >= f.count
        ev.evaluated += 1
        ev.satisfied += int(ok)
        ev.total_distance += 0.0 if ok else float(f.count - found)
        ev.details.append(
            {"constraint": f.name, "found": found, "required": f.count, "ok": ok}
        )

    return ev


def evaluate_many(
    smiles: list[str],
    constraints: Constraints,
    *,
    predictor: Any = None,
) -> list[Evaluation]:
    """Evaluate a population in one pass.

    The predictor is asked once for the whole batch rather than once per molecule: the ADMET
    worker costs ~5s to wake and ~50ms per molecule after that, so per-molecule calls would
    spend almost all of their time on overhead.

    A predictor that cannot answer degrades to skipping the four predicted properties, and
    the Evaluation records that as ``skipped``. It never guesses.
    """
    preds: dict[str, dict] = {}
    if predictor is not None and constraints.needs_predictor:
        try:
            preds = predictor(smiles) or {}
        except Exception:  # noqa: BLE001 - an unreachable predictor is a skip, not a crash
            preds = {}
    return [evaluate(s, constraints, predictions=preds.get(s)) for s in smiles]


def admet_predictor(venv: Any = None):
    """A predictor backed by the isolated ADMET worker.

    Returned as a callable rather than wired in directly so tests and the gate can substitute
    a recorded table, and so importing this module never starts a subprocess.
    """
    from ...admet import BENCH_COLUMNS as _COLS
    from ...admet import PANEL, get_worker

    def predict(smiles: list[str]) -> dict[str, dict]:
        wanted = sorted({*_COLS.values(), *(c for c, _u in PANEL.values())})
        return get_worker(venv).predict(list(smiles), columns=wanted)

    return predict


def unsatisfied(ev: Evaluation) -> list[dict]:
    """Just the failures, for a generator or a prompt to act on."""
    return [d for d in ev.details if d.get("ok") is False]


__all__ = [
    "BENCH_COLUMNS",
    "FRAGMENT_FNS",
    "NEEDS_PREDICTOR",
    "PROPERTY_FNS",
    "UNAVAILABLE",
    "Bound",
    "Constraints",
    "Evaluation",
    "FragmentRequirement",
    "admet_predictor",
    "evaluate",
    "evaluate_many",
    "unsatisfied",
]
