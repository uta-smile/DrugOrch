"""Morgan fingerprints and Tanimoto similarity.

The repo had no notion of molecular similarity at all before this -- the only structural
comparison anywhere was ``murcko_scaffold``. Rediscovery, novelty and diversity all need one,
so this is it.

Packed as a ``uint8`` bit matrix rather than RDKit bit vectors, because every consumer wants
*all-pairs* comparisons -- "how close is each of 200 generated molecules to each of 50 held-out
actives" -- and a numpy popcount over a packed matrix does that in one call instead of 10,000
Python-level ones. The fingerprint itself is cached per canonical SMILES; a benchmark hits the
same actives on every arm and every seed, so the same molecule is fingerprinted once per
process rather than once per run.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .chem import canonical
from .versions import FP_V

N_BITS = 2048
RADIUS = 2
"""ECFP4 in the usual naming -- radius 2 counts two bonds out, so a diameter of four."""

# popcount for a uint8, looked up rather than computed. numpy has no native popcount on
# older versions and this is the hot path in an all-pairs comparison.
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def _generator():
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)


_GEN = None


@lru_cache(maxsize=200_000)
def _packed(smiles: str, version: str) -> bytes | None:
    """One molecule's packed fingerprint, keyed on the recipe version.

    ``version`` is in the cache key for the same reason every other cache in this project
    carries one: changing the radius or the bit count must invalidate rather than silently
    blend two different fingerprints into one comparison.
    """
    global _GEN  # noqa: PLW0603 - one generator per process
    from .chem import to_mol

    mol = to_mol(smiles)
    if mol is None:
        return None
    if _GEN is None:
        _GEN = _generator()
    bv = _GEN.GetFingerprintAsNumPy(mol).astype(np.uint8)
    return np.packbits(bv).tobytes()


def fingerprint(smiles: str) -> np.ndarray | None:
    """One packed fingerprint, or ``None`` if the molecule will not parse."""
    smi = canonical(smiles)
    if smi is None:
        return None
    raw = _packed(smi, FP_V)
    return None if raw is None else np.frombuffer(raw, dtype=np.uint8)


def matrix(smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    """Packed fingerprints for a list, plus the molecules that actually parsed.

    Returns the surviving SMILES alongside the matrix so a caller can keep rows and labels
    aligned -- silently dropping unparseable molecules while keeping the original list is how
    a metric ends up reported against the wrong molecule.
    """
    rows, kept = [], []
    for s in smiles:
        fp = fingerprint(s)
        if fp is not None:
            rows.append(fp)
            kept.append(s)
    if not rows:
        return np.zeros((0, N_BITS // 8), dtype=np.uint8), []
    return np.vstack(rows), kept


def tanimoto(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """All-pairs Tanimoto between two packed matrices, shape ``(len(a), len(b))``.

    ``|A & B| / (|A| + |B| - |A & B|)``, computed over packed bytes. An empty input gives an
    empty matrix rather than raising, because "no molecules to compare" is a normal state
    early in a run and not an error.
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    popc = _POPCOUNT[a].sum(axis=1).astype(np.float32)
    popc_b = _POPCOUNT[b].sum(axis=1).astype(np.float32)

    # The intersection, one byte-column at a time. Materializing the full
    # (len(a), len(b), n_bytes) AND would be gigabytes on a real benchmark.
    inter = np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    for col in range(a.shape[1]):
        inter += _POPCOUNT[a[:, col, None] & b[None, :, col]]

    union = popc[:, None] + popc_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out.astype(np.float32)


def max_similarity(query: list[str], reference: list[str]) -> dict[str, float]:
    """For each query molecule, its similarity to the nearest reference molecule.

    The primitive behind both rediscovery ("did we get close to a held-out active?") and
    novelty ("how far is this from anything we were given?").
    """
    qm, qk = matrix(query)
    rm, _ = matrix(reference)
    if qm.size == 0 or rm.size == 0:
        return dict.fromkeys(qk, 0.0)
    sims = tanimoto(qm, rm).max(axis=1)
    return {s: float(v) for s, v in zip(qk, sims, strict=True)}


def mean_pairwise_distance(smiles: list[str], cap: int = 400) -> float:
    """Mean ``1 - Tanimoto`` over distinct pairs: a diversity number, 0 to 1.

    Capped because this is quadratic and a diversity estimate does not improve meaningfully
    past a few hundred molecules.
    """
    m, kept = matrix(smiles[:cap])
    if len(kept) < 2:
        return 0.0
    sims = tanimoto(m, m)
    iu = np.triu_indices(len(kept), k=1)
    return float(1.0 - sims[iu].mean())


__all__ = [
    "N_BITS",
    "RADIUS",
    "fingerprint",
    "matrix",
    "max_similarity",
    "mean_pairwise_distance",
    "tanimoto",
]
