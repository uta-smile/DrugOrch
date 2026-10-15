"""Turning a PDB entry into a receptor Vina can dock into.

This is the most failure-prone step in the system, because crystal structures are messy in
ways that have nothing to do with chemistry: alternate conformations, missing side chains,
four copies of the protein in the asymmetric unit, a dozen buffer molecules. The pipeline is
therefore written to fail *informatively* and to be retried on a different structure, which
is why the retrieval subsystem keeps a shortlist rather than a single pick.

Steps:

1. Read the mmCIF with gemmi.
2. Choose one polymer chain -- the longest -- and discard the rest. Docking into a dimer
   when the site is in one monomer doubles the search space for nothing.
3. Find the native ligand by its component id and take its coordinates.
4. **Compute the docking box ourselves** from those coordinates plus padding, rather than
   delegating it. The box is the single most important input to a docking run: too small and
   the true pose will not fit, too large and the search dilutes until scores are noise. It is
   worth owning outright and worth recording in the provenance.
5. Strip waters, ions, buffers and the ligand itself, and write the receptor as PDB.
6. Convert to PDBQT with Meeko.

The result is cached on disk keyed by structure, chain and recipe version, because preparing
a receptor takes seconds and a demo should not repeat it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import gemmi

from ..versions import RECPREP_V

# Everything that is in a crystal because of how it was grown, not because it binds.
NON_RECEPTOR = frozenset(
    """
    HOH DOD SO4 PO4 GOL EDO PEG PG4 PGE 1PE 2PE P6G MPD MES EPE TRS BTB BIS CIT ACT ACY
    FMT OXL TAR MLI SCN AZI NO3 CO3 BCT NH4 IMD DMS DMF ACN EOH IPA MOH TFA
    NA CL K MG CA ZN MN FE FE2 CU CU1 NI CO CD HG AU PT PB BA SR CS RB LI BR IOD IUM F
    """.split()
)

MIN_BOX = 18.0
"""Floor on each box dimension, in angstroms.

A small or flat native ligand produces a box too thin for a larger candidate to adopt any
pose at all, and the resulting scores look like weak binding when they really mean 'did not
fit'. Clamping up front is cheaper than discovering it from the numbers.
"""
MAX_BOX = 34.0
"""Ceiling on each dimension. Past roughly this size Vina's search is spread too thin at
normal exhaustiveness and scores stop being comparable between ligands."""


class ReceptorPrepError(RuntimeError):
    """Preparation failed for this structure. Try the next one on the shortlist."""


@dataclass(frozen=True, slots=True)
class Box:
    """A docking box, in the form Vina's config file wants."""

    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float

    def to_config(self) -> str:
        return "\n".join(f"{k} = {v:.3f}" for k, v in asdict(self).items()) + "\n"

    def to_dict(self) -> dict:
        return {k: round(v, 3) for k, v in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class Receptor:
    """A prepared receptor, ready to hand to Vina."""

    pdb_id: str
    chain: str
    ligand_code: str
    pdbqt_path: Path
    box_path: Path
    box: Box
    n_residues: int
    directory: Path

    def to_dict(self) -> dict:
        return {
            "pdb_id": self.pdb_id,
            "chain": self.chain,
            "ligand_code": self.ligand_code,
            "pdbqt": str(self.pdbqt_path),
            "box_config": str(self.box_path),
            "box": self.box.to_dict(),
            "n_residues": self.n_residues,
            "recprep_version": RECPREP_V,
        }


def box_from_points(points: list[tuple[float, float, float]], padding: float = 5.0) -> Box:
    """A box enclosing these atoms plus padding, clamped to a workable size."""
    if not points:
        raise ReceptorPrepError("no ligand atoms to build a box from")
    xs, ys, zs = zip(*points, strict=True)
    cx, cy, cz = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, (max(zs) + min(zs)) / 2
    sx = min(MAX_BOX, max(MIN_BOX, (max(xs) - min(xs)) + 2 * padding))
    sy = min(MAX_BOX, max(MIN_BOX, (max(ys) - min(ys)) + 2 * padding))
    sz = min(MAX_BOX, max(MIN_BOX, (max(zs) - min(zs)) + 2 * padding))
    return Box(cx, cy, cz, sx, sy, sz)


def _load(cif_text: str) -> gemmi.Structure:
    st = gemmi.read_structure_string(cif_text, format=gemmi.CoorFormat.Mmcif)
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    return st


def _pick_chain(model: gemmi.Model, ligand_code: str) -> tuple[str, int]:
    """The chain to dock into: the longest polymer that carries the ligand if any does.

    Preferring the chain the ligand actually sits in matters when the asymmetric unit holds
    several copies and only one has the site occupied -- docking into an empty copy would
    put the box in open solvent.
    """
    best: tuple[str, int] | None = None
    with_ligand: tuple[str, int] | None = None
    for chain in model:
        poly = chain.get_polymer()
        n = len(poly)
        if n < 30:
            continue
        has_lig = ligand_code and any(r.name == ligand_code for r in chain)
        if best is None or n > best[1]:
            best = (chain.name, n)
        if has_lig and (with_ligand is None or n > with_ligand[1]):
            with_ligand = (chain.name, n)
    chosen = with_ligand or best
    if chosen is None:
        raise ReceptorPrepError("no polymer chain of at least 30 residues found")
    return chosen


def _ligand_points(
    model: gemmi.Model, ligand_code: str, chain_name: str
) -> list[tuple[float, float, float]]:
    """Coordinates of the native ligand -- preferring the copy in the chosen chain."""
    same_chain: list[tuple[float, float, float]] = []
    anywhere: list[tuple[float, float, float]] = []
    for chain in model:
        for res in chain:
            if res.name != ligand_code:
                continue
            pts = [(a.pos.x, a.pos.y, a.pos.z) for a in res]
            if chain.name == chain_name and not same_chain:
                same_chain = pts
            elif not anywhere:
                anywhere = pts
    return same_chain or anywhere


def _write_receptor_pdb(st: gemmi.Structure, chain_name: str, out: Path) -> int:
    """Write the chosen polymer chain alone, with every heteroatom removed."""
    keep = gemmi.Structure()
    keep.spacegroup_hm = st.spacegroup_hm
    keep.cell = st.cell
    model = gemmi.Model("1")
    new_chain = gemmi.Chain(chain_name)
    n = 0
    for res in st[0][chain_name]:
        info = gemmi.find_tabulated_residue(res.name)
        is_aa = info is not None and info.is_amino_acid()
        if not is_aa or res.name in NON_RECEPTOR:
            continue
        new_chain.add_residue(res)
        n += 1
    if n == 0:
        raise ReceptorPrepError(f"chain {chain_name} has no standard amino-acid residues")
    model.add_chain(new_chain)
    keep.add_model(model)
    keep.setup_entities()
    out.parent.mkdir(parents=True, exist_ok=True)
    keep.write_pdb(str(out))
    return n


def _meeko_exe() -> str | None:
    """Locate ``mk_prepare_receptor``, preferring the one in this interpreter's venv."""
    local = Path(sys.executable).parent / (
        "mk_prepare_receptor.exe" if sys.platform == "win32" else "mk_prepare_receptor"
    )
    if local.is_file():
        return str(local)
    return shutil.which("mk_prepare_receptor")


def _to_pdbqt(pdb: Path, out_basename: Path, timeout: float = 300.0) -> Path:
    """Convert a receptor PDB to PDBQT with Meeko.

    ``-x`` lets Meeko carry on past residues it cannot template (non-standard residues,
    truncated side chains). Those are common in real structures and almost never near the
    site; refusing the whole structure over one of them would throw away most of the PDB.
    """
    exe = _meeko_exe()
    if exe is None:
        raise ReceptorPrepError(
            "mk_prepare_receptor not found; install meeko to enable docking"
        )
    pdbqt = out_basename.with_suffix(".pdbqt")
    cmd = [exe, "--read_pdb", str(pdb), "-o", str(out_basename), "-p", "-x"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if not pdbqt.is_file():
        tail = ((proc.stderr or "") + (proc.stdout or "")).strip().splitlines()
        detail = " | ".join(tail[-4:]) if tail else f"exit {proc.returncode}"
        raise ReceptorPrepError(f"meeko did not produce a pdbqt: {detail}")
    return pdbqt


def prepare(
    cif_text: str,
    pdb_id: str,
    ligand_code: str,
    out_dir: Path,
    *,
    padding: float = 5.0,
    force: bool = False,
) -> Receptor:
    """Full pipeline: mmCIF text in, prepared receptor and box out.

    Cached on ``out_dir``; pass ``force`` to rebuild. Raises :class:`ReceptorPrepError` with
    a human-readable reason, which the docking subsystem surfaces as a routing event.
    """
    out_dir = Path(out_dir)
    meta_path = out_dir / "receptor.json"
    pdbqt_path = out_dir / "receptor.pdbqt"
    box_path = out_dir / "box.txt"

    if not force and meta_path.is_file() and pdbqt_path.is_file() and box_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("recprep_version") == RECPREP_V:
            b = meta["box"]
            return Receptor(
                pdb_id=meta["pdb_id"],
                chain=meta["chain"],
                ligand_code=meta["ligand_code"],
                pdbqt_path=pdbqt_path,
                box_path=box_path,
                box=Box(**b),
                n_residues=meta["n_residues"],
                directory=out_dir,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    st = _load(cif_text)
    if not len(st):
        raise ReceptorPrepError(f"{pdb_id}: structure contains no models")

    chain_name, _ = _pick_chain(st[0], ligand_code)
    points = _ligand_points(st[0], ligand_code, chain_name)
    if not points:
        raise ReceptorPrepError(
            f"{pdb_id}: ligand {ligand_code!r} not found in the structure, so the "
            "docking box cannot be placed"
        )
    box = box_from_points(points, padding=padding)

    receptor_pdb = out_dir / "receptor.pdb"
    n_res = _write_receptor_pdb(st, chain_name, receptor_pdb)
    _to_pdbqt(receptor_pdb, out_dir / "receptor")
    box_path.write_text(box.to_config(), encoding="utf-8")

    rec = Receptor(
        pdb_id=pdb_id.upper(),
        chain=chain_name,
        ligand_code=ligand_code,
        pdbqt_path=pdbqt_path,
        box_path=box_path,
        box=box,
        n_residues=n_res,
        directory=out_dir,
    )
    meta_path.write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")
    return rec
