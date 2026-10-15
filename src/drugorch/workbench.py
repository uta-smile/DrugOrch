"""Generation as a tool, not as a step in an experiment.

Everything else in this project is built to answer one research question -- *does an LLM
route a discovery campaign better than a rule?* -- and that question imposes a shape on the
code that is wrong for ordinary use. A run has a budget, because two arms can only be
compared at equal spend. The generator may not call docking
(:mod:`drugorch.subsystems.generation`), because a subsystem that commissions its own scoring
takes the spending decision away from the policy under test. Every call goes through a
routing policy, because the routing *is* the experiment.

A scientist who wants to see a molecule in 3D, or elaborate a scaffold, or ask which of
forty analogues fits a pocket, wants none of that. They want the operation. Making them
start a budgeted, routed, policy-driven campaign to get one conformer is an experimental
protocol leaking into a tool.

So this module exposes the same primitives directly:

* :func:`embed_3d` -- SMILES to a real 3D conformer. This direction is exact and cheap:
  distance geometry plus a force field, no model, no training set, deterministic under a
  seed.
* :func:`elaborate` -- a seed molecule to analogues, by the same GraphGA operators the
  generation subsystem uses, with no run state and no budget.
* :func:`dock_batch` -- molecules into a prepared pocket, the structure-conditioned
  direction.

**On "generate a molecule from a 3D structure".** That is the genuinely hard direction and
it is worth being precise about what this does and does not do. A pocket-conditioned
*generative prior* -- Pocket2Mol, TargetDiff, DiffSBDD and their relatives -- proposes atoms
directly into a binding site. None of those is installed here and none has been verified to
run on this machine, so nothing in this module pretends to be one. What :func:`dock_batch`
offers instead is structure-conditioned *selection*: propose molecules by chemistry, build
each one's geometry, score every one against the real pocket, and keep what fits. That is a
weaker method than a learned prior and a completely honest one -- the geometry is real, the
score is real, and the ranking means what it says.

Nothing here is charged, cached against a run, or recorded in a routing trace. These are
operations on molecules, and they belong to whoever asked for them.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .chem import canonical
from .config import Settings, get_settings
from .prep import ligand as ligprep
from .subsystems.generation import crossover, mutate
from .subsystems.properties import compute_panel

MAX_BATCH = 64
"""Ceiling on molecules per workbench call.

Not a licence restriction -- a browser one. These endpoints are synchronous apart from
docking, and a request that embeds five hundred conformers holds a worker for a minute and
returns a payload no page can usefully draw.
"""


class WorkbenchError(RuntimeError):
    """A workbench operation could not be carried out, with a reason worth showing."""


# --- SMILES -> 3D ---------------------------------------------------------------


def embed_3d(smiles: str, seed: int = 42) -> dict:
    """One molecule's 3D structure, as SDF, with the numbers that describe it.

    The exact direction. A SMILES carries connectivity and stereochemistry but no geometry,
    and :func:`drugorch.prep.ligand.embed` supplies the geometry with ETKDGv3 -- distance
    geometry under crystal-derived torsion preferences -- followed by an MMFF relaxation.
    There is no model and no training data involved, so there is nothing here that can be
    confidently wrong in the way a generated structure can be.

    It is also reproducible: the embedding seed is derived from the molecule and ``seed``
    (see :func:`drugorch.prep.ligand.conformer_seed`), so the same request returns the same
    coordinates on any machine. That matters more than it sounds -- a conformer that changed
    between two requests would make every number computed from it unquotable.

    The returned conformer is *a* low-energy structure, not *the* bound conformation. A
    ligand adopts a different geometry in a pocket than it does in isolation, which is
    precisely why :func:`dock_batch` exists and why its poses are the ones to reason about
    when a target is in play.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    smi = canonical(smiles)
    if not smi:
        raise WorkbenchError(f"not a readable SMILES: {smiles!r}")

    try:
        mol = ligprep.embed(smi, seed)
    except ligprep.LigandPrepError as exc:
        raise WorkbenchError(str(exc)) from exc

    # The MMFF energy of the relaxed conformer. Reported because it is the one number that
    # says whether the relaxation actually found something sensible, and withheld rather
    # than faked when MMFF has no parameters for the molecule.
    energy: float | None = None
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props)
            if ff is not None:
                energy = round(float(ff.CalcEnergy()), 3)
    except Exception:  # noqa: BLE001 -- an energy is a nicety, never a reason to fail
        energy = None

    mol.SetProp("_Name", smi)
    sdf = Chem.MolToMolBlock(mol)

    heavy = mol.GetNumHeavyAtoms()
    return {
        "smiles": smi,
        "sdf": sdf,
        "n_atoms": mol.GetNumAtoms(),
        "n_heavy_atoms": heavy,
        "n_rotatable_bonds": int(Descriptors.NumRotatableBonds(mol)),
        "mmff_energy": energy,
        "seed": ligprep.conformer_seed(smi, seed),
        "method": "ETKDGv3 (small-ring and macrocycle torsions) + MMFF94 relaxation",
        "panel": compute_panel(smi),
    }


def embed_many(smiles: list[str], seed: int = 42) -> tuple[list[dict], list[dict]]:
    """:func:`embed_3d` over a list. Returns ``(structures, failures)``.

    Failures are returned rather than raised because a batch is usually a generated set, and
    one molecule that will not embed is a fact about that molecule, not a reason to discard
    the other thirty-nine.
    """
    out: list[dict] = []
    failed: list[dict] = []
    for s in smiles[:MAX_BATCH]:
        try:
            out.append(embed_3d(s, seed))
        except WorkbenchError as exc:
            failed.append({"smiles": s, "error": str(exc)})
    return out, failed


# --- seed -> analogues ----------------------------------------------------------


def elaborate(
    seeds: list[str],
    *,
    n: int = 20,
    seed: int = 42,
    mutation_rate: float = 0.5,
    max_attempts_per_child: int = 12,
    with_properties: bool = True,
) -> dict:
    """Analogues of one or more seed molecules, by GraphGA crossover and mutation.

    The same operators :mod:`drugorch.subsystems.generation` uses, called directly. The
    subsystem wraps them in parent selection by observed fitness, budget accounting and a
    run-state diff; none of that means anything outside a campaign, and with a single seed
    there is no fitness to select on anyway.

    With one seed, every child is a mutation of it -- crossover needs two distinct parents.
    With several, the two operators alternate by ``mutation_rate``, which is what makes a
    hand-picked pair of actives a more interesting input than one lead.

    Deterministic under ``seed``: the same seeds and the same seed give the same children.
    """
    cleaned = [c for c in (canonical(s) for s in seeds) if c]
    if not cleaned:
        raise WorkbenchError("none of the seed molecules could be read")

    n = max(1, min(MAX_BATCH, int(n)))
    rng = np.random.default_rng(seed)
    single = len(cleaned) < 2

    children: list[dict] = []
    seen: set[str] = set(cleaned)
    rejected: dict[str, int] = {}
    operators: dict[str, int] = {}
    attempts = 0

    while len(children) < n and attempts < n * max_attempts_per_child:
        attempts += 1
        # One seed cannot be crossed with itself -- the operator needs two distinct parents
        # and would otherwise burn every attempt producing the parent back.
        use_mutation = single or rng.random() < mutation_rate
        # The operators report which SMARTS or cut fired but not what they were applied to,
        # so parentage is tracked here. Without it an analogue is a molecule with no account
        # of where it came from, which is the whole value of elaborating a known seed.
        if use_mutation:
            parents = [cleaned[int(rng.integers(len(cleaned)))]]
            op = mutate(parents[0], rng)
        else:
            i, j = rng.choice(len(cleaned), size=2, replace=False)
            parents = [cleaned[int(i)], cleaned[int(j)]]
            op = crossover(parents[0], parents[1], rng)

        # Rejections are counted per operator, because the two fail for different reasons
        # and the difference is the answer to "why did I get no crossovers?". Crossover cuts
        # at acyclic single bonds, so a pair of small rigid molecules offers almost nowhere
        # to cut and it fails on nearly every attempt -- mutation then quietly fills the
        # quota and the result looks like a choice rather than a limitation.
        kind = "mutation" if use_mutation else "crossover"
        if op is None:
            key = f"{kind}_failed"
            rejected[key] = rejected.get(key, 0) + 1
            continue
        if op.smiles in seen:
            rejected["duplicate"] = rejected.get("duplicate", 0) + 1
            continue

        seen.add(op.smiles)
        operators[op.operator] = operators.get(op.operator, 0) + 1
        row: dict[str, Any] = {
            "smiles": op.smiles,
            "operator": op.operator,
            "detail": op.detail,
            "parents": parents,
        }
        if with_properties:
            row["panel"] = compute_panel(op.smiles)
        children.append(row)

    return {
        "seeds": cleaned,
        "children": children,
        "attempts": attempts,
        "operators": operators,
        "rejected": rejected,
        # The yield is the honest health reading of a GA: a run that tried 240 times for 20
        # children is nearly exhausted, whatever the count at the top says.
        "yield": round(len(children) / attempts, 3) if attempts else None,
        "note": (
            f"{len(children)} analogues of {len(cleaned)} seed"
            f"{'' if len(cleaned) == 1 else 's'} in {attempts} attempts"
            + ("; crossover needs two seeds, so these are all mutations" if single else "")
            + (
                "; crossover found nowhere to cut these seeds -- it needs acyclic single "
                "bonds, and small rigid molecules have few"
                if not single
                and not operators.get("crossover")
                and rejected.get("crossover_failed")
                else ""
            )
        ),
    }


# --- molecules -> a pocket ------------------------------------------------------


@dataclass
class Pocket:
    """A prepared receptor plus how it was chosen, for the workbench's own use."""

    pdb_id: str
    ligand_code: str
    chain: str
    receptor_pdbqt: Path
    box_config: Path
    box: dict
    n_residues: int
    receptor_pdb: Path
    why: str = ""

    def to_dict(self) -> dict:
        return {
            "pdb_id": self.pdb_id,
            "ligand_code": self.ligand_code,
            "chain": self.chain,
            "box": self.box,
            "n_residues": self.n_residues,
            "why": self.why,
        }


def prepare_pocket(
    pdb_id: str, ligand_code: str = "", settings: Settings | None = None
) -> Pocket:
    """Download a PDB entry and prepare its ligand-marked pocket for docking.

    ``ligand_code`` names which bound ligand marks the site. Left empty, the entry's own
    ligands are ranked the same way the retrieval subsystem ranks them and the best is used
    -- a structure usually has several heteroatoms bound and most of them are buffer, so
    picking the largest is wrong often enough to be worth doing properly.
    """
    from .clients import rcsb
    from .clients.http import BankClient
    from .prep import receptor as recprep
    from .versions import RECPREP_V

    st = settings or get_settings()
    st.ensure_dirs()
    client = BankClient(st.http_cache_dir, offline=st.offline)
    pid = pdb_id.strip().upper()
    if not pid:
        raise WorkbenchError("no PDB id given")

    why = f"ligand {ligand_code.upper()} as given" if ligand_code else ""
    lig = ligand_code.strip().upper()
    if not lig:
        described = rcsb.describe_entry(client, pid)
        if described is None or not described.ligand_code:
            raise WorkbenchError(
                f"{pid} has no bound ligand to mark a pocket. Docking needs a site; pick a "
                "structure solved with a ligand in it, or name the ligand code explicitly."
            )
        lig = described.ligand_code
        why = described.reason or f"largest drug-like ligand in {pid}"

    cif = rcsb.fetch_cif(client, pid)
    if not cif:
        raise WorkbenchError(f"could not download {pid} from RCSB")

    try:
        rec = recprep.prepare(
            cif, pid, lig, st.receptor_cache_dir / f"{pid}_{lig}_{RECPREP_V}"
        )
    except Exception as exc:  # noqa: BLE001 -- recprep raises several unrelated types
        raise WorkbenchError(f"could not prepare {pid}/{lig}: {exc}") from exc

    return Pocket(
        pdb_id=rec.pdb_id,
        ligand_code=rec.ligand_code,
        chain=rec.chain,
        receptor_pdbqt=rec.pdbqt_path,
        box_config=rec.box_path,
        box=rec.box.to_dict(),
        n_residues=rec.n_residues,
        receptor_pdb=rec.directory / "receptor.pdb",
        why=why,
    )


def dock_batch(
    smiles: list[str],
    pocket: Pocket,
    *,
    seed: int = 42,
    exhaustiveness: int = 8,
    settings: Settings | None = None,
    cancel: threading.Event | None = None,
    on_progress=None,
) -> list[dict]:
    """Dock each molecule into ``pocket``. Returns one row per molecule, best first.

    The structure-conditioned direction, done by selection: the pocket does not propose
    molecules, it judges them. Each row carries the pose, so a score can be looked at rather
    than taken on trust -- a favourable number computed from a ligand threaded through a
    backbone is a false positive no filter on the number itself will catch.
    """
    from .subsystems.docking import dock_one

    st = settings or get_settings()
    vina = st.resolved_vina()
    if vina is None:
        raise WorkbenchError(
            "no Vina executable found. Put one at tools/vina.exe or on PATH; without it a "
            "pocket can be prepared but nothing can be scored against it."
        )

    # Unreadable input is reported, not dropped. Silently returning two rows for three
    # molecules leaves the reader to work out which one went missing and why, and the most
    # likely explanation -- a typo in a pasted SMILES -- is exactly the one worth surfacing.
    targets: list[str] = []
    unreadable: list[str] = []
    for raw in smiles[:MAX_BATCH]:
        smi = canonical(raw)
        if smi:
            targets.append(smi)
        else:
            unreadable.append(raw)
    if not targets:
        raise WorkbenchError("none of the molecules could be read")

    rows: list[dict] = []
    done = 0
    lock = threading.Lock()

    def one(smi: str) -> dict:
        nonlocal done
        if cancel is not None and cancel.is_set():
            return {"smiles": smi, "score": None, "note": "cancelled", "pose": ""}
        score, note, pose = dock_one(
            smi,
            pocket.receptor_pdbqt,
            pocket.box_config,
            vina,
            run_seed=seed,
            exhaustiveness=exhaustiveness,
            cpu=st.vina_cpu_per_worker,
            timeout=st.vina_timeout_s,
        )
        with lock:
            done += 1
            if on_progress is not None:
                on_progress(done, len(targets), smi)
        return {"smiles": smi, "score": score, "note": note, "pose": pose}

    with concurrent.futures.ThreadPoolExecutor(max_workers=st.vina_workers) as pool:
        rows = list(pool.map(one, targets))

    for row in rows:
        row["panel"] = compute_panel(row["smiles"])
    rows += [
        {"smiles": s, "score": None, "note": "not a readable SMILES", "pose": "", "panel": None}
        for s in unreadable
    ]
    # Every score in this project is a maximization -- docking's is negated at the source --
    # so one ordering works everywhere. Molecules that failed sort last rather than being
    # dropped: "this would not dock" is a result about the molecule.
    rows.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0.0)))
    return rows


# --- jobs -----------------------------------------------------------------------
#
# Docking is tens of seconds per molecule, so it cannot be an ordinary request. The two
# operations above return in milliseconds and stay synchronous; this is the one that needs
# somewhere to live while it runs.


@dataclass
class Job:
    """One long workbench operation, observable while it runs."""

    job_id: str
    kind: str
    status: str = "running"
    created_at: float = field(default_factory=time.time)
    done: int = 0
    total: int = 0
    message: str = ""
    result: Any = None
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    poses: list[str] = field(default_factory=list)
    """Pose text, parallel to the result rows, served one at a time rather than inline.

    A nine-model pose is several KB and only the selected one is ever drawn, so sixty-four
    of them in a polling response would be almost entirely waste.
    """

    def snapshot(self) -> dict:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "done": self.done,
            "total": self.total,
            "message": self.message,
            "error": self.error,
            "elapsed_s": round(time.time() - self.created_at, 1),
            "result": self.result if self.status == "done" else None,
        }


class JobBoard:
    """In-memory job registry for the workbench.

    Deliberately not persisted. A workbench operation is a question someone asked while
    looking at the screen, not an experiment whose provenance has to survive a restart --
    runs are the thing this project keeps, and conflating the two would put unreviewed
    scratch work in the record alongside measured campaigns.
    """

    def __init__(self, keep: int = 32):
        self._jobs: dict[str, Job] = {}
        self._keep = keep
        self._lock = threading.Lock()
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="workbench"
        )

    def submit(self, kind: str, fn) -> Job:
        """Run ``fn(job)`` on a worker thread and return the job immediately."""
        job = Job(job_id=f"j_{int(time.time() * 1000) % 10_000_000:07d}", kind=kind)
        with self._lock:
            self._jobs[job.job_id] = job
            self._evict()

        def runner() -> None:
            try:
                job.result = fn(job)
                job.status = "cancelled" if job.cancel.is_set() else "done"
            except Exception as exc:  # noqa: BLE001 -- a job must not kill the server
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = "failed"

        self._pool.submit(runner)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def _evict(self) -> None:
        if len(self._jobs) <= self._keep:
            return
        for jid, _ in sorted(self._jobs.items(), key=lambda kv: kv[1].created_at)[
            : len(self._jobs) - self._keep
        ]:
            self._jobs.pop(jid, None)


_BOARD: JobBoard | None = None


def board() -> JobBoard:
    global _BOARD  # noqa: PLW0603 -- one board per process, created on first use
    if _BOARD is None:
        _BOARD = JobBoard()
    return _BOARD


__all__ = [
    "MAX_BATCH",
    "Job",
    "JobBoard",
    "Pocket",
    "WorkbenchError",
    "board",
    "dock_batch",
    "elaborate",
    "embed_3d",
    "embed_many",
    "prepare_pocket",
]
