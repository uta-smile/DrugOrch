"""Molecule standardization. The canonical SMILES produced here is the key for everything.

Two molecules that differ only in salt form, protonation or atom ordering are the same
molecule as far as a budget is concerned, and charging twice for them would inflate every
cost curve. So every SMILES entering the system is funnelled through :func:`canonical_smiles`
before it is counted, cached or compared, and :data:`drugorch.versions.STD_V` rides along in
the cache key so a change to this pipeline invalidates rather than blends.

Parse failures are not exceptions here. A generator will emit unparseable strings, and the
orchestrator needs to charge for the attempt and move on, not unwind.
"""

from __future__ import annotations

from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

from .versions import STD_V

# RDKit logs every sanitization failure to stderr. We handle failures by returning None,
# so the log is pure noise that would bury real output during a run.
RDLogger.DisableLog("rdApp.*")

_normalizer = rdMolStandardize.Normalizer()
_largest = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def to_mol(smiles: str) -> Chem.Mol | None:
    """Parse and standardize, or ``None`` if RDKit cannot make sense of the string.

    The order matters: normalize functional groups first so that fragment selection sees
    consistent representations, then take the largest fragment to drop counterions, then
    neutralize what remains.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _normalizer.normalize(mol)
        mol = _largest.choose(mol)
        mol = _uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol if mol.GetNumAtoms() > 0 else None


def canonical_smiles(smiles: str) -> str | None:
    """The standardized canonical SMILES, or ``None`` for an unparseable string."""
    mol = to_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


@lru_cache(maxsize=200_000)
def _canonical_cached(smiles: str, std_version: str) -> str | None:
    return canonical_smiles(smiles)


def canonical(smiles: str) -> str | None:
    """Memoised :func:`canonical_smiles`. Use this on hot paths.

    ``STD_V`` is part of the memo key so the in-process cache cannot outlive a recipe change
    within a single session either.
    """
    if not isinstance(smiles, str):
        return None
    return _canonical_cached(smiles, STD_V)


def canonical_many(smiles: list[str]) -> list[str | None]:
    """Canonicalize a batch, preserving order and position of failures."""
    return [canonical(s) for s in smiles]


def is_valid(smiles: str) -> bool:
    return canonical(smiles) is not None


def inchikey(smiles: str) -> str | None:
    """InChIKey of the standardized molecule -- a stable cross-database identifier."""
    mol = to_mol(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def murcko_scaffold(smiles: str) -> str | None:
    """Bemis-Murcko scaffold, used to tell novel chemotypes from analogues."""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = to_mol(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return None


def dedupe(smiles: list[str]) -> list[str]:
    """Unique canonical SMILES in first-seen order, dropping unparseable strings."""
    seen: set[str] = set()
    out: list[str] = []
    for s in smiles:
        c = canonical(s)
        if c is not None and c not in seen:
            seen.add(c)
            out.append(c)
    return out
