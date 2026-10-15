"""Molecule generation by graph genetic algorithm (GraphGA).

Why a GA rather than a neural generator, for v1: it is pure RDKit, so there is no GPU
dependency, no wheel to match against a Blackwell card, and no pretrained checkpoint whose
training set might already contain the answers. It is deterministic under a seed, which the
reproducibility claim depends on. And it produces valid molecules essentially always, where a
character-level model emits a tail of unparseable strings that would quietly consume budget.

The approach is Jensen's: treat molecules as graphs, cut parents at acyclic single bonds and
recombine the pieces (crossover), then perturb single atoms and bonds (mutation). Fitness is
supplied by the orchestrator from scores already observed -- **this subsystem never calls
docking or properties itself.** That restriction is deliberate: if the generator could
commission its own scoring, it would be spending budget the routing policy did not authorize,
and the policy would no longer be the only thing deciding where compute goes.

Seeds come from retrieval's ChEMBL actives when they exist. Generating from nothing is
possible but much weaker, and the gap between the two is part of what makes the ordering
decision ("fetch actives before generating?") worth studying.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from ..chem import canonical, murcko_scaffold, to_mol
from ..versions import GEN_V
from .base import BaseSubsystem, Capability, Cost, ParamSpec, ProgressFn, SubsystemResult

if TYPE_CHECKING:  # pragma: no cover
    from ..orchestrator.state import RunState

# Rejoin two cut pieces at their dummy atoms.
_JOIN = AllChem.ReactionFromSmarts("[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]")

# Single-step perturbations. Kept small and chemically bland on purpose: a mutation operator
# that builds exotic motifs produces molecules no one would synthesize, which inflates the
# apparent novelty of a run without improving it.
_MUTATIONS: tuple[str, ...] = (
    "[*;!H0:1]>>[*:1]C",  # grow a methyl
    "[*;!H0:1]>>[*:1]O",  # add a hydroxyl
    "[*;!H0:1]>>[*:1]N",  # add an amine
    "[*;!H0:1]>>[*:1]F",  # fluorinate
    "[*;!H0:1]>>[*:1]Cl",
    "[C;!H0:1]>>[N:1]",  # carbon to nitrogen
    "[N;!H0:1]>>[C:1]",
    "[O;D2:1]>>[N;H1:1]",  # ether to secondary amine
    "[C:1][CH3]>>[C:1]",  # trim a methyl
    "[c:1][OH]>>[c:1]OC",  # cap a phenol
)
_MUTATION_RXNS = tuple(AllChem.ReactionFromSmarts(s) for s in _MUTATIONS)

# Keep offspring inside the range where docking scores mean anything and chemistry is
# plausible. Molecules outside this are not "bold", they are noise.
MIN_HEAVY_ATOMS = 12
MAX_HEAVY_ATOMS = 55
MAX_MW = 650.0


def _sanitize(mol: Chem.Mol | None) -> str | None:
    """Canonical SMILES for a generated molecule, or ``None`` if it is unusable."""
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    smi = Chem.MolToSmiles(mol)
    clean = canonical(smi)
    if clean is None:
        return None
    m = to_mol(clean)
    if m is None:
        return None
    n = m.GetNumHeavyAtoms()
    if n < MIN_HEAVY_ATOMS or n > MAX_HEAVY_ATOMS:
        return None
    if Descriptors.MolWt(m) > MAX_MW:
        return None
    return clean


def _cut(mol: Chem.Mol, rng: np.random.Generator) -> tuple[Chem.Mol, Chem.Mol] | None:
    """Split at a random acyclic single bond, leaving a dummy atom on each piece.

    Ring bonds are excluded: cutting one opens the ring and the recombination almost always
    produces a strained or invalid structure, so it wastes a draw.
    """
    bonds = [
        b.GetIdx()
        for b in mol.GetBonds()
        if b.GetBondType() == Chem.BondType.SINGLE
        and not b.IsInRing()
        and b.GetBeginAtom().GetDegree() > 1
        and b.GetEndAtom().GetDegree() > 1
    ]
    if not bonds:
        return None
    bidx = int(rng.choice(bonds))
    try:
        frag = Chem.FragmentOnBonds(mol, [bidx], addDummies=True, dummyLabels=[(1, 1)])
        pieces = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=True)
    except Exception:
        return None
    if len(pieces) != 2:
        return None
    return pieces[0], pieces[1]


@dataclass(frozen=True, slots=True)
class Operation:
    """A child plus the operation that made it.

    The operators used to return a bare SMILES, which meant that *which* of the ten SMARTS
    fired was a loop variable that died at the return statement. A researcher looking at a
    generation round could not tell a productive mutation operator from a dead one, so the
    operator set could never be tuned on evidence.
    """

    smiles: str
    operator: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {"smiles": self.smiles, "operator": self.operator, "detail": self.detail}


def crossover(a: str, b: str, rng: np.random.Generator, tries: int = 6) -> Operation | None:
    """Recombine two parents into one child, or ``None`` if no valid child appeared."""
    ma, mb = to_mol(a), to_mol(b)
    if ma is None or mb is None:
        return None
    for _ in range(tries):
        ca, cb = _cut(ma, rng), _cut(mb, rng)
        if ca is None or cb is None:
            continue
        left = ca[int(rng.integers(2))]
        right = cb[int(rng.integers(2))]
        try:
            products = _JOIN.RunReactants((left, right))
        except Exception:
            continue
        for prod in products:
            smi = _sanitize(prod[0])
            if smi is not None:
                return Operation(smiles=smi, operator="crossover")
    return None


def mutate(smiles: str, rng: np.random.Generator, tries: int = 8) -> Operation | None:
    """Apply one random single-step perturbation, reporting which one fired."""
    mol = to_mol(smiles)
    if mol is None:
        return None
    for _ in range(tries):
        idx = int(rng.integers(len(_MUTATION_RXNS)))
        rxn = _MUTATION_RXNS[idx]
        try:
            products = rxn.RunReactants((mol,))
        except Exception:
            continue
        if not products:
            continue
        prod = products[int(rng.integers(len(products)))][0]
        smi = _sanitize(prod)
        if smi is not None and smi != smiles:
            return Operation(smiles=smi, operator="mutation", detail=_MUTATIONS[idx])
    return None


CAPABILITY = Capability(
    key="generation",
    title="Molecule generation and lead optimization (GraphGA)",
    summary=(
        "Produce new molecules by recombining and perturbing the best molecules seen so far. "
        "Works far better when seeded with known actives from retrieval than from nothing. "
        "Generated molecules have no properties and no binding estimate until the relevant "
        "subsystem is run on them, so generating without scoring afterwards adds candidates "
        "but no knowledge."
    ),
    requires=("molecules",),
    produces=("new_molecules",),
    params=(
        ParamSpec(
            name="n_offspring",
            type="int",
            default=20,
            lo=4,
            hi=100,
            doc="How many new molecules to produce this round.",
        ),
        ParamSpec(
            name="n_parents",
            type="int",
            default=12,
            lo=2,
            hi=60,
            doc="Size of the parent pool, drawn from the best-scoring molecules available.",
        ),
        ParamSpec(
            name="mutation_rate",
            type="float",
            default=0.35,
            lo=0.0,
            hi=1.0,
            doc="Chance that a child is additionally mutated after crossover.",
        ),
        ParamSpec(
            name="novel_scaffolds_only",
            type="bool",
            default=False,
            doc="Keep only offspring whose Murcko scaffold is new to the run. Explores "
            "harder but yields fewer molecules per call.",
        ),
    ),
    unit_cost=2.0,
    per="call",
    typical_wall_s=3.0,
    version=GEN_V,
)


class GenerationSubsystem(BaseSubsystem):
    """One GA generation per call, seeded from the run's best molecules."""

    capability = CAPABILITY

    def preflight(self, params: dict, state: RunState) -> tuple[bool, str]:
        if not state.molecules:
            return False, "no molecules to breed from; retrieve known actives first"
        return True, ""

    def _parents(self, state: RunState, n: int) -> tuple[list[tuple[str, float | None]], str]:
        """The best molecules available, and **on what basis** they were judged best.

        Three fallbacks with materially different scientific meaning: the objective score,
        then measured potency, then whatever happens to exist in insertion order. The last
        one is not selection at all -- it breeds on nothing. Returning the basis alongside
        the parents is what lets a reader tell those apart; previously the branch taken was
        unrecorded and all three looked identical from outside.
        """
        scored = state.scored()
        if scored:
            return [(s, v) for s, v in scored[:n]], state.objective_key

        by_pchembl = [
            (s, rec.scores["pchembl"])
            for s, rec in state.molecules.items()
            if rec.scores.get("pchembl") is not None
        ]
        if by_pchembl:
            by_pchembl.sort(key=lambda kv: kv[1], reverse=True)
            return by_pchembl[:n], "pchembl"

        return [(s, None) for s in list(state.molecules)[:n]], "insertion_order"

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
        n_offspring = int(params.get("n_offspring", 20))
        n_parents = int(params.get("n_parents", 12))
        mutation_rate = float(params.get("mutation_rate", 0.35))
        novel_only = bool(params.get("novel_scaffolds_only", False))

        # Seeded from the run seed and the step, so one run is reproducible end to end while
        # successive generations still explore differently.
        rng = np.random.default_rng(state.seed * 1_000_003 + state.step)

        ranked, basis = self._parents(state, n_parents)
        if len(ranked) < 2:
            return SubsystemResult.failure(
                self.capability.key, "need at least two parent molecules to breed"
            )
        parents = [s for s, _ in ranked]

        known = set(state.molecules)
        known_scaffolds = {
            murcko_scaffold(s) for s in known if murcko_scaffold(s) is not None
        }

        # A real GA generation index. `state.step` is the orchestrator's step counter, so a
        # policy calling generation at steps 3, 7 and 11 produced "generations" 3, 7 and 11 --
        # which is not a generation number and cannot be plotted against anything.
        generation = state.generation + 1

        children: dict[str, tuple[tuple[str, str], Operation]] = {}
        n_attempts = 0
        max_attempts = n_offspring * 12

        # Four distinct ways a candidate dies, counted separately. The ratio between them is
        # the clearest signal a GA gives about its own health: mostly `duplicate` means the
        # parent pool is exhausted, mostly `invalid` means the operators are fighting the
        # chemistry, mostly `scaffold` means the novelty filter is doing the work.
        rejected = {"invalid": 0, "duplicate": 0, "scaffold": 0, "same_parent": 0}
        mutations_used: dict[str, int] = {}
        n_mutated = 0

        while len(children) < n_offspring and n_attempts < max_attempts:
            if cancel.is_set():
                break
            n_attempts += 1
            i, j = rng.integers(len(parents)), rng.integers(len(parents))
            if i == j:
                rejected["same_parent"] += 1
                continue
            pa, pb = parents[int(i)], parents[int(j)]

            op = crossover(pa, pb, rng)
            if op is None:
                rejected["invalid"] += 1
                continue

            if rng.random() < mutation_rate:
                mut = mutate(op.smiles, rng)
                if mut is not None:
                    op = Operation(
                        smiles=mut.smiles, operator="crossover+mutation", detail=mut.detail
                    )
                    mutations_used[mut.detail] = mutations_used.get(mut.detail, 0) + 1
                    n_mutated += 1

            child = op.smiles
            if child in known or child in children:
                rejected["duplicate"] += 1
                continue
            if novel_only:
                sc = murcko_scaffold(child)
                if sc is None or sc in known_scaffolds:
                    rejected["scaffold"] += 1
                    continue
                known_scaffolds.add(sc)

            children[child] = ((pa, pb), op)
            if len(children) % 5 == 0:
                progress(
                    f"{len(children)}/{n_offspring} offspring", len(children) / n_offspring
                )

        wall = time.perf_counter() - t0
        if children:
            state.generation = generation

        records = [
            {
                "smiles": smi,
                "parents": list(ps),
                "props": {
                    "generation_step": state.step,
                    "generation": generation,
                    "operator": op.operator,
                    "operator_detail": op.detail,
                    "backend": "graph_ga",
                },
            }
            for smi, (ps, op) in children.items()
        ]

        note = (
            f"{len(children)} new molecules from {len(parents)} parents "
            f"({n_attempts} attempts)"
        )
        if novel_only:
            note += "; novel scaffolds only"
        if not records:
            note = f"no valid novel offspring in {n_attempts} attempts"

        n_rejected = sum(rejected.values())
        top_rejects = ", ".join(
            f"{n} {why}" for why, n in sorted(rejected.items(), key=lambda kv: -kv[1]) if n
        )
        explain = {
            "backend": "graph_ga",
            "generation": generation,
            "parent_basis": basis,
            "parents": [
                {"smiles": smi, "fitness": round(v, 4) if v is not None else None, "rank": k}
                for k, (smi, v) in enumerate(ranked, start=1)
            ],
            "operators": {
                "crossover": len(children) - n_mutated,
                "crossover+mutation": n_mutated,
            },
            "mutations_used": mutations_used,
            "rejected": rejected,
            "attempts": n_attempts,
            "max_attempts": max_attempts,
            "accepted": [
                {"smiles": smi, "parents": list(ps), "operator": op.operator}
                for smi, (ps, op) in children.items()
            ],
            "rationale": (
                f"Generation {generation}: bred {len(parents)} parents ranked by {basis} "
                f"into {len(children)} offspring over {n_attempts} attempts"
                + (f"; {n_rejected} rejected ({top_rejects})" if n_rejected else "")
                + "."
            ),
        }

        return SubsystemResult(
            subsystem=self.capability.key,
            ok=bool(records),
            records=records,
            explain=explain,
            cost=Cost(
                budget_units=0.0,
                wall_s=round(wall, 3),
                n_unique_smiles=len(records),
            ),
            note=note,
            error=None if records else "generation produced no valid novel molecules",
        )

    def describe(self) -> dict:
        return {
            "key": self.capability.key,
            "version": self.capability.version,
            "algorithm": "GraphGA (acyclic-bond crossover + single-step SMARTS mutation)",
            "n_mutation_operators": len(_MUTATIONS),
            "heavy_atom_range": [MIN_HEAVY_ATOMS, MAX_HEAVY_ATOMS],
            "max_mw": MAX_MW,
        }
