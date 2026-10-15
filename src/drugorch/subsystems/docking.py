"""Docking with AutoDock Vina: the expensive subsystem, and the one worth routing around.

Everything else in this system costs milliseconds. Docking costs ten to thirty seconds per
molecule, which makes it the only subsystem where *when* you call it genuinely matters -- and
therefore the one that gives the routing question its teeth. A policy that docks forty
molecules it could have filtered on properties first has wasted most of its budget, and that
mistake is only visible because docking is slow.

Scores are reported **negated**, so higher is better and the objective is a maximization like
every other score in the run. Vina reports binding free energy in kcal/mol where more
negative means tighter; carrying that convention through the whole system would mean every
comparison needed a sign check somewhere.

Two failure modes are handled differently on purpose:

* **Per-molecule failures** (a conformer that will not embed) are recorded and skipped. One
  awkward molecule should not end a round.
* **Receptor preparation failure** is structural -- it will not succeed on a retry with the
  same structure. The subsystem walks the shortlist, and only if every candidate fails does
  it mark docking unavailable, which the policy then sees and routes around.
"""

from __future__ import annotations

import concurrent.futures as cf
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ..cache import ValueCache
from ..chem import canonical
from ..clients import rcsb
from ..clients.http import BankClient
from ..config import Settings, get_settings
from ..prep import poses as posestore
from ..prep import receptor as recprep
from ..prep.ligand import LigandPrepError, to_pdbqt
from ..versions import DOCK_V, RECPREP_V
from .base import BaseSubsystem, Capability, Cost, ParamSpec, ProgressFn, SubsystemResult

if TYPE_CHECKING:  # pragma: no cover
    from ..orchestrator.state import RunState

SCORE_KEY = "vina_score"
_RESULT_RE = re.compile(r"^REMARK VINA RESULT:\s+([-\d.]+)", re.MULTILINE)


def parse_vina_score(pdbqt_text: str) -> float | None:
    """Best (first) pose affinity from a Vina output file, in kcal/mol."""
    m = _RESULT_RE.search(pdbqt_text or "")
    return float(m.group(1)) if m else None


def dock_one(
    smiles: str,
    receptor_pdbqt: Path,
    box_config: Path,
    vina_exe: Path,
    *,
    run_seed: int = 42,
    exhaustiveness: int = 8,
    cpu: int = 4,
    timeout: float = 300.0,
) -> tuple[float | None, str, str]:
    """Dock one molecule. Returns ``(score, note, pose_pdbqt)``.

    ``score`` is negated so higher is better; ``None`` means this molecule could not be
    docked, with the reason in ``note``.
    """
    try:
        ligand_text = to_pdbqt(smiles, run_seed)
    except LigandPrepError as exc:
        return None, f"ligand prep failed: {exc}", ""

    with tempfile.TemporaryDirectory(prefix="drugorch_dock_") as tmp:
        tmpd = Path(tmp)
        lig = tmpd / "ligand.pdbqt"
        out = tmpd / "out.pdbqt"
        lig.write_text(ligand_text, encoding="utf-8")

        cmd = [
            str(vina_exe),
            "--receptor", str(receptor_pdbqt),
            "--ligand", str(lig),
            "--config", str(box_config),
            "--out", str(out),
            "--exhaustiveness", str(exhaustiveness),
            "--cpu", str(cpu),
            "--seed", str(run_seed),
            "--verbosity", "0",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, f"vina timed out after {timeout:.0f}s", ""

        if not out.is_file():
            tail = ((proc.stderr or "") + (proc.stdout or "")).strip().splitlines()
            return None, f"vina produced no output: {' | '.join(tail[-3:]) or proc.returncode}", ""

        pose = out.read_text(encoding="utf-8")

    raw = parse_vina_score(pose)
    if raw is None:
        return None, "could not parse a score from the vina output", ""
    return -raw, f"vina {raw:.2f} kcal/mol", pose


CAPABILITY = Capability(
    key="docking",
    title="Structure-based binding affinity (AutoDock Vina)",
    summary=(
        "Estimate how tightly molecules bind the target by docking them into a crystal "
        "structure. This is the only subsystem that measures target engagement, and it is "
        "by far the most expensive -- roughly 20x the cost of the property panel per "
        "molecule and tens of seconds each. Requires a structure from retrieval. Worth "
        "spending on molecules that have already survived cheaper filters."
    ),
    requires=("pdb_shortlist",),
    produces=(SCORE_KEY,),
    params=(
        ParamSpec(
            name="smiles",
            type="smiles_list",
            default=[],
            doc="Molecules to dock. Leave empty to dock the best unscored molecules.",
        ),
        ParamSpec(
            name="max_molecules",
            type="int",
            default=8,
            lo=1,
            hi=40,
            doc="Cap on molecules docked this call. Each one costs real wall-clock time.",
        ),
        ParamSpec(
            name="exhaustiveness",
            type="int",
            default=8,
            lo=1,
            hi=32,
            doc="Search effort per molecule. Higher is more reliable and proportionally "
            "slower; 8 is the Vina default.",
        ),
    ),
    unit_cost=1.0,
    per="molecule",
    typical_wall_s=20.0,
    version=f"{DOCK_V}+{RECPREP_V}",
)


class DockingSubsystem(BaseSubsystem):
    """Prepares a receptor once, then docks batches of molecules in parallel."""

    capability = CAPABILITY

    def __init__(
        self,
        settings: Settings | None = None,
        cache: ValueCache | None = None,
        client: BankClient | None = None,
    ):
        self.settings = settings or get_settings()
        self.cache = cache
        self._client = client
        self._receptor: recprep.Receptor | None = None
        self._receptor_failures: dict[str, str] = {}

    @property
    def client(self) -> BankClient:
        if self._client is None:
            self.settings.ensure_dirs()
            self._client = BankClient(self.settings.http_cache_dir, offline=self.settings.offline)
        return self._client

    def cache_key(self, rec: recprep.Receptor, exhaustiveness: int) -> str:
        """Cache namespace. Changing the receptor or the search effort changes the number,
        so both belong in the key -- otherwise two incomparable runs would share rows."""
        return f"docking|{self.capability.version}|{rec.pdb_id}:{rec.chain}|ex{exhaustiveness}"

    # --- receptor --------------------------------------------------------------
    def _ensure_receptor(self, state: RunState, progress: ProgressFn) -> recprep.Receptor:
        """Prepare a receptor, walking the shortlist until one works."""
        if self._receptor is not None:
            return self._receptor

        shortlist = state.facts.get("pdb_shortlist") or []
        candidates = [c for c in shortlist if c.get("ligand_code")]
        if not candidates:
            raise recprep.ReceptorPrepError(
                "no PDB candidate with a bound ligand; run retrieval with fetch='structures'"
            )

        errors: list[str] = []
        for cand in candidates[:4]:
            pid, lig = cand["pdb_id"], cand["ligand_code"]
            if pid in self._receptor_failures:
                errors.append(f"{pid}: {self._receptor_failures[pid]}")
                continue
            progress(f"preparing receptor {pid}", 0.05)
            try:
                cif = rcsb.fetch_cif(self.client, pid)
                if not cif:
                    raise recprep.ReceptorPrepError(f"{pid}: could not download the structure")
                rec = recprep.prepare(
                    cif,
                    pid,
                    lig,
                    self.settings.receptor_cache_dir / f"{pid}_{lig}_{RECPREP_V}",
                )
            except Exception as exc:
                why = str(exc)
                self._receptor_failures[pid] = why
                errors.append(f"{pid}: {why}")
                continue
            self._receptor = rec
            return rec

        raise recprep.ReceptorPrepError(
            "every candidate structure failed preparation -- " + "; ".join(errors[:4])
        )

    # --- selection -------------------------------------------------------------
    def _targets(self, params: dict, state: RunState) -> list[str]:
        """Which molecules to dock.

        An empty ``smiles`` list means "the most promising unscored molecules", ranked by
        QED when a property panel exists. That ordering matters: with a cap of eight per
        call, docking the eight most drug-like unscored molecules is a meaningfully better
        use of the budget than docking eight arbitrary ones.
        """
        cap = int(params.get("max_molecules", 8))
        given = [c for c in (canonical(s) for s in (params.get("smiles") or [])) if c]
        if given:
            return given[:cap]

        unscored = [
            (s, state.molecules[s].scores.get("qed") or 0.0) for s in state.unscored(SCORE_KEY)
        ]
        unscored.sort(key=lambda kv: kv[1], reverse=True)
        return [s for s, _ in unscored[:cap]]

    def preflight(self, params: dict, state: RunState) -> tuple[bool, str]:
        if self.settings.resolved_vina() is None:
            return False, "the AutoDock Vina executable was not found"
        if not (state.facts.get("pdb_shortlist") or []):
            return False, "no PDB structure available; run retrieval with fetch='structures'"
        if not self._targets(params, state):
            return False, "every molecule already has a binding estimate"
        return True, ""

    def estimate_cost(self, params: dict, state: RunState) -> Cost:
        targets = self._targets(params, state)
        novel = state.budget.novel(self.capability.key, targets)
        ex = int(params.get("exhaustiveness", 8))
        per = 2.0 + 1.6 * ex  # measured roughly: a few seconds plus search effort
        return Cost(
            budget_units=self.capability.unit_cost * len(novel),
            n_unique_smiles=len(novel),
            wall_s=per * len(novel) / max(1, self.settings.vina_workers),
        )

    # --- run -------------------------------------------------------------------
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
        vina = self.settings.resolved_vina()
        if vina is None:
            return SubsystemResult.failure(self.capability.key, "vina executable not found")

        try:
            rec = self._ensure_receptor(state, progress)
        except Exception as exc:
            # Structural, not transient: tell the orchestrator to stop offering docking.
            state.mark_unavailable(self.capability.key, str(exc))
            return SubsystemResult.failure(
                self.capability.key, f"receptor preparation failed: {exc}"
            )

        targets = self._targets(params, state)
        if not targets:
            return SubsystemResult.failure(self.capability.key, "no molecules to dock")

        ex = int(params.get("exhaustiveness", 8))
        key = self.cache_key(rec, ex)
        cached = self.cache.get_many(key, targets) if self.cache else {}

        records: list[dict] = []
        to_write: dict[str, tuple] = {}
        failures: list[str] = []
        poses_dir = self.settings.runs_dir / state.run_id / "poses"
        store = self.settings.pose_cache_dir
        # Keyed by SMILES so an entry can be filled in by either path below and the index
        # ends up in a stable order rather than in completion order.
        index: dict[str, dict] = {}

        pending = [s for s in targets if s not in cached]
        for smi in targets:
            if smi in cached:
                score, ok = cached[smi]
                if ok and score is not None:
                    records.append({"smiles": smi, "scores": {SCORE_KEY: score}})
                    # The score came from cache, so the pose that produced it is already in
                    # the store under the same key -- unless it was docked before this
                    # store existed, in which case there is nothing to point at and the
                    # molecule is listed without a file rather than silently omitted.
                    path = posestore.pose_path(store, key, smi)
                    index[smi] = {
                        "smiles": smi,
                        "score": score,
                        "file": path.name if path.is_file() else None,
                        "from_cache": True,
                    }

        done = len(targets) - len(pending)
        progress(
            f"{done} from cache, docking {len(pending)}",
            done / max(1, len(targets)),
        )

        if pending:
            workers = max(1, min(self.settings.vina_workers, len(pending)))
            with cf.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        dock_one,
                        smi,
                        rec.pdbqt_path,
                        rec.box_path,
                        vina,
                        run_seed=state.seed,
                        exhaustiveness=ex,
                        cpu=self.settings.vina_cpu_per_worker,
                        timeout=self.settings.vina_timeout_s,
                    ): smi
                    for smi in pending
                }
                for fut in cf.as_completed(futures):
                    smi = futures[fut]
                    done += 1
                    if cancel.is_set():
                        for f in futures:
                            f.cancel()
                        break
                    try:
                        score, note, pose = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        score, note, pose = None, f"{type(exc).__name__}: {exc}", ""

                    to_write[smi] = (score, score is not None, 0.0)
                    if score is None:
                        failures.append(f"{smi[:28]}: {note}")
                    else:
                        records.append({"smiles": smi, "scores": {SCORE_KEY: score}})
                        if pose:
                            path = posestore.put(store, key, smi, pose)
                            index[smi] = {
                                "smiles": smi,
                                "score": score,
                                "file": path.name,
                                "from_cache": False,
                            }
                    progress(f"docked {done}/{len(targets)}", done / max(1, len(targets)))

        if self.cache is not None and to_write:
            self.cache.put_many(key, to_write)

        # Ordered by score so the viewer opens on the best pose rather than on whichever
        # molecule happened to finish first.
        entries = sorted(index.values(), key=lambda e: e["score"], reverse=True)
        n_poses = sum(1 for e in entries if e["file"])
        if entries:
            posestore.write_index(poses_dir, key, entries, receptor=rec.to_dict())

        wall = time.perf_counter() - t0
        note = (
            f"{len(records)} scored against {rec.pdb_id} chain {rec.chain} "
            f"(exhaustiveness {ex})"
        )
        if failures:
            note += f"; {len(failures)} failed"
        if cancel.is_set():
            note += "; cancelled early"

        return SubsystemResult(
            subsystem=self.capability.key,
            ok=bool(records),
            records=records,
            facts={
                "receptor": rec.to_dict(),
                "docking_failures": failures[:10] if failures else [],
            },
            # A route, not a filesystem path: the browser cannot open the latter, and the
            # old `poses_dir` string was never consumed by anything for that reason.
            artifacts=(
                {
                    "poses_url": f"/api/runs/{state.run_id}/poses",
                    "n_poses": n_poses,
                    "bucket": posestore.bucket(key),
                }
                if n_poses
                else {}
            ),
            cost=Cost(
                budget_units=0.0,
                wall_s=round(wall, 3),
                n_unique_smiles=len(targets),
                cache_hits=len(cached),
            ),
            note=note,
            error=None if records else f"every molecule failed to dock: {'; '.join(failures[:3])}",
        )

    def describe(self) -> dict:
        return {
            "key": self.capability.key,
            "version": self.capability.version,
            "engine": "AutoDock Vina",
            "vina_exe": str(self.settings.resolved_vina() or ""),
            "exhaustiveness_default": self.settings.vina_exhaustiveness,
            "workers": self.settings.vina_workers,
            "receptor": self._receptor.to_dict() if self._receptor else None,
            "receptor_failures": self._receptor_failures,
            "score_key": SCORE_KEY,
            "score_convention": "negated vina affinity -- higher is better",
        }
