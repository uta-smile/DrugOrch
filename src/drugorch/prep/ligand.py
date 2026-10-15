"""Turning a SMILES string into a 3D ligand Vina can dock.

A SMILES has no geometry, so a conformer has to be generated before docking. Vina then
explores torsions itself, which means the starting conformer matters less than the ring
geometry and stereochemistry being right -- those it will not fix.

Reproducibility is the reason this module exists separately rather than being four lines
inside the docking subsystem. Conformer generation is stochastic, and a docking score that
silently depends on an unrecorded random draw is not a measurement. The embedding seed is
therefore derived deterministically from the molecule and the run seed, so the same molecule
in the same run always starts from the same geometry, and two runs at the same seed are
comparable.
"""

from __future__ import annotations

import contextlib
import hashlib

from rdkit import Chem
from rdkit.Chem import AllChem

from ..chem import to_mol


class LigandPrepError(RuntimeError):
    """This molecule could not be turned into a dockable 3D structure."""


def conformer_seed(smiles: str, run_seed: int = 0) -> int:
    """A stable embedding seed for this molecule in this run.

    Derived from a hash rather than a counter so it does not depend on the order molecules
    happen to be docked in -- otherwise adding one molecule to a batch would change the
    geometry, and therefore the score, of every molecule after it.
    """
    h = hashlib.sha256(f"{smiles}|{run_seed}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def embed(smiles: str, run_seed: int = 0, *, max_attempts: int = 3) -> Chem.Mol:
    """Build a 3D, hydrogen-added, force-field-relaxed conformer."""
    mol = to_mol(smiles)
    if mol is None:
        raise LigandPrepError(f"unparseable SMILES: {smiles!r}")

    mol = Chem.AddHs(mol)
    seed = conformer_seed(smiles, run_seed)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    # Crystal-derived torsion preferences give better starting geometry on ring systems,
    # which is most of what a kinase inhibitor is made of.
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True

    cid = -1
    for attempt in range(max_attempts):
        params.randomSeed = seed + attempt
        cid = AllChem.EmbedMolecule(mol, params)
        if cid >= 0:
            break
    if cid < 0:
        # Random-coordinate embedding is slower and lower quality, but it rescues strained
        # or highly constrained molecules that distance-geometry alone cannot place.
        params.useRandomCoords = True
        params.randomSeed = seed
        cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        raise LigandPrepError("could not generate a 3D conformer")

    # An unrelaxed conformer still docks; Vina re-optimizes torsions anyway. Failing the
    # whole molecule because the force field lacked parameters would be worse.
    with contextlib.suppress(Exception):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=600)
    return mol


def to_pdbqt(smiles: str, run_seed: int = 0) -> str:
    """PDBQT text for one molecule, ready to pass to Vina."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    mol = embed(smiles, run_seed)
    prep = MoleculePreparation()
    try:
        setups = prep.prepare(mol)
    except Exception as exc:
        raise LigandPrepError(f"meeko preparation failed: {exc}") from exc
    if not setups:
        raise LigandPrepError("meeko produced no molecule setup")

    text, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok or not text:
        raise LigandPrepError(f"meeko pdbqt writer failed: {err}")
    return text
