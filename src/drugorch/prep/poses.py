"""Where docked poses are kept, and why they are named the way they are.

A Vina run produces the only 3D evidence this system ever generates: the actual coordinates
of a ligand in a pocket. The score is a summary of that pose, and a summary is not a
substitute -- a favourable number computed from a physically implausible geometry is a
false positive that no amount of downstream filtering on the number itself can catch. So
the poses are kept, and kept addressable.

**Naming.** Poses are named by a SHA-256 of the canonical SMILES. The obvious alternative,
Python's builtin ``hash()``, is salted per process unless ``PYTHONHASHSEED`` is pinned, so
the same molecule lands under a different filename on every run and no pose on disk can be
traced back to the molecule it came from. :func:`drugorch.prep.ligand.conformer_seed`
already hashes with SHA-256 for exactly this reason; this module follows it.

**Location.** Poses live beside the value cache rather than inside the run directory,
because the *score* cache is shared across runs. A run whose scores all came from cache
would otherwise produce no poses at all, which would make a run's 3D evidence a function of
what some earlier run happened to evaluate rather than of the run itself. Keying the pose
store by the same string as the score cache keeps the two in step: if the score was cached,
the pose that produced it is on disk too.

The run directory keeps ``index.json`` -- the record of which poses a given run actually
used, in run-local terms, with the score beside each one. That file is what the API serves
and what the viewer reads.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

#: How many hex characters of the digest name a file. 16 is 64 bits -- far past collision
#: concern for a corpus of molecules, and short enough to read in a directory listing.
_NAME_CHARS = 16


def pose_name(smiles: str) -> str:
    """Stable filename stem for one molecule's pose.

    Deterministic across processes and machines, which ``hash()`` is not.
    """
    return hashlib.sha256(smiles.encode()).hexdigest()[:_NAME_CHARS]


def bucket(cache_key: str) -> str:
    """Directory name for one docking configuration.

    The docking cache key contains ``|`` and ``:``, neither of which is a legal Windows
    filename character, so the key is hashed rather than sanitized -- sanitizing would let
    two genuinely different configurations collapse onto one directory.
    """
    return hashlib.sha256(cache_key.encode()).hexdigest()[:_NAME_CHARS]


def pose_path(store_root: Path, cache_key: str, smiles: str) -> Path:
    """Absolute path where this molecule's pose for this configuration lives."""
    return store_root / bucket(cache_key) / f"{pose_name(smiles)}.pdbqt"


def put(store_root: Path, cache_key: str, smiles: str, pdbqt_text: str) -> Path:
    """Write one pose into the store and return its path."""
    path = pose_path(store_root, cache_key, smiles)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pdbqt_text, encoding="utf-8")
    return path


def write_index(
    run_poses_dir: Path, cache_key: str, entries: list[dict], receptor: dict | None = None
) -> Path:
    """Record which poses this run used, and what they were docked into.

    ``entries`` are ``{"smiles", "score", "file", "from_cache"}``. ``file`` is the pose's
    name within the bucket, not an absolute path, so the index survives the data directory
    being moved or a run directory being copied to another machine. The bucket is recorded
    once at the top rather than on every entry, since a single call docks into one
    configuration by construction.

    ``receptor`` is stored here because a pose without the structure it was docked into is
    a cloud of atoms in space. Neither ``run.json`` nor ``summary.json`` carries it and the
    ``call_finished`` event drops ``facts``, so for a finished run this file is the only
    record of which structure produced these coordinates.
    """
    run_poses_dir.mkdir(parents=True, exist_ok=True)
    index = run_poses_dir / "index.json"
    index.write_text(
        json.dumps(
            {"bucket": bucket(cache_key), "receptor": receptor, "poses": entries},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return index


class PoseConvertError(RuntimeError):
    """This pose could not be turned into SDF."""


def to_sdf(pdbqt_text: str) -> str:
    """Convert a multi-MODEL Vina PDBQT into a multi-record SDF.

    Browsers cannot be trusted with PDBQT. 3Dmol.js parses it without raising and returns
    the wrong molecule -- measured on a 9-MODEL, 27-atom-per-model file, it reported 27
    "frames" and 9 atoms of only C and N, having transposed models and atoms. A viewer that
    fails loudly is a bug; one that silently renders a plausible wrong structure defeats the
    entire point of looking at the pose, so the conversion happens here instead.

    Meeko is already the writer on the way in (:mod:`drugorch.prep.ligand`), and it stamps
    ``REMARK SMILES`` into the PDBQT it produces, so the bond orders that PDBQT itself does
    not carry are recoverable on the way back out. Poses written by some other tool will not
    have that remark and will fail here rather than be guessed at.
    """
    from meeko import PDBQTMolecule, RDKitMolCreate

    try:
        pmol = PDBQTMolecule(pdbqt_text, is_dlg=False, skip_typing=True)
        sdf, failures = RDKitMolCreate.write_sd_string(pmol)
    except Exception as exc:  # noqa: BLE001 -- meeko raises a wide range
        raise PoseConvertError(f"meeko could not read the pose: {exc}") from exc
    if not sdf:
        raise PoseConvertError(f"meeko produced no SDF (failed poses: {failures})")
    return sdf


def read_index(run_poses_dir: Path) -> dict:
    """What :func:`write_index` wrote, or an empty index if there is none.

    Always returns the same shape, so callers never have to distinguish "no poses yet"
    from "unreadable index" unless they want to.
    """
    empty: dict = {"bucket": None, "receptor": None, "poses": []}
    index = run_poses_dir / "index.json"
    if not index.is_file():
        return empty
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return empty
    return {
        "bucket": data.get("bucket"),
        "receptor": data.get("receptor"),
        "poses": data.get("poses", []),
    }
