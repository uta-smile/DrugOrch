"""Pocket-conditioned generation with DiffSBDD, callable rather than command-line.

DiffSBDD ships ``generate_ligands.py``, a script that parses argv, loads a checkpoint, writes
an SDF and exits. That is the wrong shape for anything that wants to generate into several
pockets, or hold the model in memory across calls, or get molecules back as objects -- which
is everything DrugOrch would want. So this wraps the same model with a function.

Three things have to be dealt with before the model will run at all on this machine, and all
three are environmental rather than scientific:

**The checkpoint is a Lightning 1.8 pickle and torch is 2.11.** Since torch 2.6 ``torch.load``
defaults to ``weights_only=True``, and this file carries an ``argparse.Namespace``. The fix
used here is to allowlist that one class. The alternative -- ``weights_only=False`` -- would
also work and would permit arbitrary code execution while unpickling; a published checkpoint
from Zenodo is probably fine, but "probably fine" is not a reason to take the broader option
when the narrow one works.

**torch-scatter is not installed and cannot be.** It ships CUDA kernels needing a compile
against this exact torch build, and no wheel exists for torch 2.11 + cu128 on Windows. See
``external/torch_scatter_shim`` -- DiffSBDD uses two of its functions, both at ``dim=0``,
which torch does natively.

**The repo is not a package.** Its modules import each other by bare name, so the clone has
to be on ``sys.path`` and the working directory does not matter.

The model is 1.01 M parameters, which is the surprise: this is far smaller than the docking
and Boltz-2 work it would sit alongside, and the VRAM measurement below reflects that.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent / "DiffSBDD"


@dataclass
class GenerationResult:
    """What one pocket-conditioned sampling call produced."""

    smiles: list[str] = field(default_factory=list)
    n_requested: int = 0
    n_returned: int = 0
    wall_s: float = 0.0
    peak_vram_gb: float | None = None
    device: str = ""
    sdf_path: Path | None = None

    @property
    def validity(self) -> float:
        """Fraction of requested molecules that came back as parseable structures.

        Reported because a generative model's headline number is usually quoted after
        filtering, and the gap between asked-for and usable is the thing a budget has to be
        planned against.
        """
        return self.n_returned / self.n_requested if self.n_requested else 0.0


def _patch_openbabel_tempfile() -> None:
    """Fix an upstream bug that makes the molecule builder fail on Windows.

    ``analysis.molecule_builder.make_mol_openbabel`` opens a ``NamedTemporaryFile`` and then
    hands its *name* to three other functions that open it again. On POSIX that is fine; on
    Windows the handle is exclusive, so the second open raises ``PermissionError`` and every
    generated molecule is lost -- after the diffusion has already run, which is the
    expensive part.

    This is a portability bug, not a scientific one: the fix is a temporary *directory* with
    ordinary files inside it, and the chemistry is untouched. Patching here rather than
    editing the clone keeps the checkout pristine, so `git pull` does not conflict and
    nobody has to wonder whether the model was modified.

    Two further departures from upstream, both deliberate:
    * separate paths for the xyz and the sdf, rather than overwriting one -- the original
      relies on OpenBabel having fully read the file before it writes over it;
    * a ``None`` return when RDKit cannot read the sdf back, instead of an ``AttributeError``
      on the next line. The caller already drops ``None``, and one unreadable atom cloud
      should not lose the whole batch.
    """
    from analysis import molecule_builder as mb  # noqa: PLC0415 -- needs sys.path first

    if getattr(mb, "_drugorch_patched", False):
        return

    from openbabel import openbabel
    from rdkit import Chem

    def make_mol_openbabel(positions, atom_types, atom_decoder):
        names = [atom_decoder[x] for x in atom_types]

        # No temporary files at all. OpenBabel's ReadFile/WriteFile keep their streams open
        # until the OBConversion is collected, so even a temporary *directory* could not be
        # torn down on Windows -- the handle outlives the `with` block and cleanup raises
        # WinError 32. ReadString/WriteString do the identical conversion in memory, which
        # removes the whole class of problem rather than working around it, and is faster
        # besides: this runs once per generated molecule.
        xyz = f"{len(positions)}\n\n" + "".join(
            f"{names[i]} {positions[i, 0]:.3f} {positions[i, 1]:.3f} {positions[i, 2]:.3f}\n"
            for i in range(len(positions))
        )

        conv = openbabel.OBConversion()
        conv.SetInAndOutFormats("xyz", "sdf")
        ob_mol = openbabel.OBMol()
        if not conv.ReadString(ob_mol, xyz):
            return None
        sdf = conv.WriteString(ob_mol)
        if not sdf:
            return None

        # `removeHs` defaults to True in both this and the SDMolSupplier upstream used, so
        # the molecule that comes back is the same one.
        tmp_mol = Chem.MolFromMolBlock(sdf, sanitize=False)
        if tmp_mol is None or tmp_mol.GetNumConformers() == 0:
            return None

        # Upstream's own workaround to strip radicals, kept verbatim.
        mol = Chem.RWMol()
        for atom in tmp_mol.GetAtoms():
            mol.AddAtom(Chem.Atom(atom.GetSymbol()))
        mol.AddConformer(tmp_mol.GetConformer(0))
        for bond in tmp_mol.GetBonds():
            mol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
        return mol

    mb.make_mol_openbabel = make_mol_openbabel
    mb._drugorch_patched = True


def _prepare_imports() -> None:
    """Put the clone on the path and make the checkpoint loadable. Idempotent."""
    import torch

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    torch.serialization.add_safe_globals([argparse.Namespace])
    _patch_openbabel_tempfile()


def load_model(checkpoint: Path, device: str | None = None):
    """Load the DDPM once, so several pockets can be sampled without reloading it."""
    import torch

    _prepare_imports()
    from lightning_modules import LigandPocketDDPM  # noqa: PLC0415 -- needs sys.path first

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = LigandPocketDDPM.load_from_checkpoint(str(checkpoint), map_location=dev)
    model = model.to(dev)
    model.eval()
    return model, dev


def generate(
    model,
    device: str,
    pdbfile: Path,
    ref_ligand: str,
    n_samples: int = 10,
    outfile: Path | None = None,
    timesteps: int | None = None,
    sanitize: bool = True,
) -> GenerationResult:
    """Sample ``n_samples`` molecules into the pocket ``ref_ligand`` marks in ``pdbfile``.

    ``ref_ligand`` is either ``"CHAIN:RESI"`` (a residue in the PDB file) or a path to an
    SDF holding the reference ligand. It does not condition the chemistry -- it says where
    the pocket is, by naming something already bound there, exactly as the docking box in
    this project's own receptor preparation does.

    ``sanitize`` drops molecules RDKit cannot read. Left on, because an unsanitizable output
    is not a molecule and counting it as one is how generative-model validity numbers get
    quoted higher than they are.
    """
    import torch
    from rdkit import Chem

    _prepare_imports()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    # DiffSBDD prints progress to stdout from inside the sampling loop; captured so a caller
    # gets a return value rather than a log.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        molecules = model.generate_ligands(
            str(pdbfile),
            n_samples,
            ref_ligand=ref_ligand,
            sanitize=sanitize,
            largest_frag=False,
            relax_iter=0,
            timesteps=timesteps,
        )

    wall = time.perf_counter() - t0
    peak = (
        torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None
    )

    smiles: list[str] = []
    for mol in molecules:
        if mol is None:
            continue
        try:
            smi = Chem.MolToSmiles(mol)
        except Exception:  # noqa: BLE001 -- a molecule that will not serialise is a miss
            continue
        if smi:
            smiles.append(smi)

    written = None
    if outfile is not None:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        with Chem.SDWriter(str(outfile)) as w:
            for mol in molecules:
                if mol is not None:
                    w.write(mol)
        written = outfile

    return GenerationResult(
        smiles=smiles,
        n_requested=n_samples,
        n_returned=len(smiles),
        wall_s=round(wall, 2),
        peak_vram_gb=round(peak, 3) if peak is not None else None,
        device=device,
        sdf_path=written,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO / "checkpoints" / "crossdocked_fullatom_cond.ckpt")
    ap.add_argument("--pdbfile", type=Path, default=REPO / "example" / "3rfm.pdb")
    ap.add_argument("--ref-ligand", default="A:330")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--outfile", type=Path, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model, dev = load_model(args.checkpoint, args.device)
    res = generate(
        model, dev, args.pdbfile, args.ref_ligand,
        n_samples=args.n_samples, outfile=args.outfile, timesteps=args.timesteps,
    )

    print(f"device         {res.device}")
    print(f"requested      {res.n_requested}")
    print(f"returned       {res.n_returned}  (validity {res.validity:.0%})")
    print(f"wall           {res.wall_s}s" +
          (f"  ({res.wall_s / res.n_returned:.2f}s per molecule)" if res.n_returned else ""))
    if res.peak_vram_gb is not None:
        print(f"peak VRAM      {res.peak_vram_gb} GB")
    if res.sdf_path:
        print(f"wrote          {res.sdf_path}")
    for smi in res.smiles:
        print("  ", smi)
    return 0 if res.n_returned else 1


if __name__ == "__main__":
    raise SystemExit(main())
