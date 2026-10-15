"""Retrieval: turning a name into a target a run can actually work on.

This is almost always the first subsystem a policy calls, because nearly everything else
depends on it. It produces three things, and which of them a call fetches is a routing choice
in its own right:

* **identity** -- the UniProt accession and sequence. Cheap, and a precondition for the rest.
* **structures** -- a ranked shortlist of dockable PDB entries. Needed before docking; useless
  if the run never docks.
* **actives** -- known potent molecules from ChEMBL. These seed generation, and they are also
  the only measured ground truth in the system.

Fetching all three when the budget only allows one docking round is a waste, and fetching
none before generating molecules means generating from nothing. That trade-off is exactly
what the routing policy exists to make, so the ``fetch`` parameter is left under its control
rather than being decided here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING

from ..clients import chembl, pubchem, rcsb, uniprot
from ..clients.http import BankClient, OfflineCacheMiss
from ..config import Settings, get_settings
from ..types import TargetSpec
from ..versions import HTTP_V
from .base import BaseSubsystem, Capability, Cost, ParamSpec, ProgressFn, SubsystemResult

if TYPE_CHECKING:  # pragma: no cover
    from ..orchestrator.state import RunState

FETCH_CHOICES = ("identity", "structures", "actives", "all")

CAPABILITY = Capability(
    key="retrieval",
    title="Data-bank retrieval (UniProt, RCSB PDB, ChEMBL, PubChem)",
    summary=(
        "Resolve the target and pull public data about it. 'identity' gives the UniProt "
        "accession and sequence; 'structures' gives a ranked shortlist of crystal structures "
        "suitable for docking; 'actives' gives known potent molecules from ChEMBL, which are "
        "the best available starting points for generation. Cheap relative to docking. "
        "Docking cannot run until structures have been fetched, and generation produces much "
        "better molecules when seeded with actives."
    ),
    requires=(),
    produces=("target_identity", "pdb_shortlist", "known_actives"),
    params=(
        ParamSpec(
            name="fetch",
            type="enum",
            default="all",
            choices=FETCH_CHOICES,
            doc="Which data to pull. 'all' does everything in one call.",
        ),
        ParamSpec(
            name="n_actives",
            type="int",
            default=100,
            lo=10,
            hi=500,
            doc="How many known actives to retrieve, strongest first.",
        ),
        ParamSpec(
            name="min_pchembl",
            type="float",
            default=6.0,
            lo=4.0,
            hi=10.0,
            doc="Potency floor for actives, as pChEMBL (6.0 = 1 micromolar).",
        ),
    ),
    unit_cost=1.0,
    per="call",
    typical_wall_s=8.0,
    version=HTTP_V,
)


class RetrievalSubsystem(BaseSubsystem):
    """Fans out to the four data banks and folds the answers into the run state."""

    capability = CAPABILITY

    def __init__(self, settings: Settings | None = None, client: BankClient | None = None):
        self.settings = settings or get_settings()
        self._client = client

    @property
    def client(self) -> BankClient:
        if self._client is None:
            self.settings.ensure_dirs()
            self._client = BankClient(
                self.settings.http_cache_dir, offline=self.settings.offline
            )
        return self._client

    def preflight(self, params: dict, state: RunState) -> tuple[bool, str]:
        """Refuse calls that could not add anything.

        A subsystem is the right place to catch a no-op, because every policy benefits: a
        confused LLM that keeps asking for data it already has is stopped here rather than
        being charged for nothing, and the refusal is visible in the trace as a routing
        mistake rather than hidden as a successful call that did nothing.
        """
        fetch = params.get("fetch", "all")
        have_identity = state.target.resolved and bool(state.target.sequence)
        have_structures = "pdb_shortlist" in state.facts
        have_actives = "known_actives" in state.facts

        if fetch in ("structures", "actives") and not state.target.resolved:
            return False, "target identity not resolved yet; fetch 'identity' first"
        if fetch == "identity" and have_identity:
            return False, "target identity already resolved"
        if fetch == "structures" and have_structures:
            return False, "PDB shortlist already retrieved"
        if fetch == "actives" and have_actives:
            return False, "known actives already retrieved"
        if fetch == "all" and have_identity and have_structures and have_actives:
            return False, "identity, structures and actives have all been retrieved already"
        return True, ""

    def estimate_cost(self, params: dict, state: RunState) -> Cost:
        fetch = params.get("fetch", "all")
        # 'all' makes substantially more requests than a single slice, and a policy choosing
        # between them should see that difference.
        units = {"identity": 1.0, "structures": 2.0, "actives": 2.0, "all": 4.0}[fetch]
        wall = {"identity": 2.0, "structures": 12.0, "actives": 8.0, "all": 20.0}[fetch]
        return Cost(budget_units=units, wall_s=wall, api_calls=int(units * 3))

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
        fetch = params.get("fetch", "all")
        want_identity = fetch in ("identity", "all") or not state.target.resolved
        want_structures = fetch in ("structures", "all")
        want_actives = fetch in ("actives", "all")

        facts: dict = {}
        records: list[dict] = []
        notes: list[str] = []
        target = state.target

        # What this call actually did, as opposed to what was asked of it. These diverge:
        # `want_identity` fires whenever the target is unresolved, so a request for
        # `fetch="structures"` can silently resolve identity too -- and the UI, showing the
        # request, was reporting something that had not happened.
        ran: list[str] = []
        explain: dict = {}
        req0, hit0 = self.client.n_requests, self.client.n_cache_hits

        try:
            # --- identity ------------------------------------------------------
            if want_identity and not cancel.is_set():
                progress("resolving target against UniProt", 0.1)
                hit = uniprot.resolve(self.client, target.query)
                if hit is None:
                    return SubsystemResult.failure(
                        self.capability.key,
                        f"UniProt has no entry matching {target.query!r}",
                        Cost(0.0, wall_s=time.perf_counter() - t0),
                    )
                seq = uniprot.sequence(self.client, hit.accession)
                progress(f"resolved {hit.gene or hit.accession}", 0.3)

                ct = chembl.target_for_uniprot(self.client, hit.accession)
                target = TargetSpec(
                    query=target.query,
                    uniprot=hit.accession,
                    name=hit.name,
                    gene=hit.gene,
                    organism=hit.organism,
                    sequence=seq,
                    chembl_id=ct.chembl_id if ct else None,
                    pdb_id=target.pdb_id,
                    pdb_ids=hit.pdb_ids,
                )
                facts["target_identity"] = {
                    "uniprot": hit.accession,
                    "name": hit.name,
                    "gene": hit.gene,
                    "organism": hit.organism,
                    "length": len(seq),
                    "reviewed": hit.reviewed,
                    "chembl_id": target.chembl_id,
                    "n_pdb_crossrefs": len(hit.pdb_ids),
                }
                facts["target_sequence"] = seq
                notes.append(f"resolved {hit.gene or hit.accession} ({hit.organism})")
                ran.append("identity")
                explain["identity"] = {
                    "query": target.query,
                    "resolved_to": hit.accession,
                    "gene": hit.gene,
                    "organism": hit.organism,
                    "reviewed": hit.reviewed,
                    "sequence_length": len(seq),
                    "chembl_target": target.chembl_id,
                    "n_pdb_crossrefs": len(hit.pdb_ids),
                    "why": "exact gene or accession match, reviewed, preferred organism"
                    if hit.reviewed
                    else "best available match; entry is unreviewed",
                }

            # --- structures ----------------------------------------------------
            if want_structures and target.uniprot and not cancel.is_set():
                progress("ranking PDB structures for dockability", 0.5)
                cands = rcsb.shortlist(
                    self.client, target.uniprot, fallback_ids=target.pdb_ids, keep=6
                )
                dockable = [c for c in cands if c.ligand_code]
                facts["pdb_shortlist"] = [c.to_dict() for c in cands]
                ran.append("structures")
                explain["structures"] = {
                    "selection": (
                        "PDB entries cross-referenced from the UniProt entry, ranked on "
                        "resolution and whether a bound ligand marks the pocket; the "
                        "best-ranked entry that has one is docked into."
                    ),
                    "examined": len(cands),
                    "dockable": len(dockable),
                    "chosen": (dockable or cands)[0].pdb_id if cands else None,
                    # The ranking, not just the winner: a reader disagreeing with the choice
                    # needs to see what it beat and on what score.
                    "ranking": [
                        {
                            "pdb_id": c.pdb_id,
                            "resolution": c.resolution,
                            "ligand": c.ligand_code or None,
                            "score": round(c.score, 2),
                            "reason": c.reason,
                        }
                        for c in cands
                    ],
                }
                if cands:
                    chosen = (dockable or cands)[0]
                    target = replace(target, pdb_id=chosen.pdb_id)
                    notes.append(
                        f"{len(cands)} structures ranked, {len(dockable)} with a bound ligand; "
                        f"chose {chosen.pdb_id} ({chosen.reason})"
                    )
                else:
                    notes.append("no X-ray structures with a bound ligand found")

            # --- actives -------------------------------------------------------
            if want_actives and not cancel.is_set():
                if not target.chembl_id:
                    notes.append("no ChEMBL target for this protein; no actives available")
                else:
                    progress("fetching known actives from ChEMBL", 0.75)
                    acts = chembl.actives(
                        self.client,
                        target.chembl_id,
                        limit=int(params.get("n_actives", 100)),
                        min_pchembl=float(params.get("min_pchembl", 6.0)),
                    )
                    # The holdout is enforced here, at the one place actives enter a run.
                    # Filtering later would be too late: a molecule the policy has already
                    # seen cannot be un-seen, and the rediscovery number would be fiction.
                    withheld = 0
                    if state.holdout:
                        kept = [a for a in acts if a.smiles not in state.holdout]
                        withheld = len(acts) - len(kept)
                        acts = kept
                    records = [a.to_record() for a in acts]
                    facts["known_actives"] = {
                        "n": len(acts),
                        "chembl_target": target.chembl_id,
                        "best_pchembl": round(acts[0].pchembl, 2) if acts else None,
                        "min_pchembl": float(params.get("min_pchembl", 6.0)),
                    }
                    notes.append(
                        f"{len(acts)} actives from {target.chembl_id}"
                        + (f", best pChEMBL {acts[0].pchembl:.2f}" if acts else "")
                    )
                    ran.append("actives")
                    by_assay: dict[str, int] = {}
                    for a in acts:
                        by_assay[a.standard_type] = by_assay.get(a.standard_type, 0) + 1
                    explain["actives"] = {
                        "chembl_target": target.chembl_id,
                        # The rule, in words, beside the numbers it produced. Without it a
                        # reader sees a hundred molecules appear and cannot say what was
                        # asked for -- which molecules were chosen, on what basis, or what
                        # was left behind. The policy picks the thresholds; this picks the
                        # molecules, and that division should not have to be inferred.
                        "selection": (
                            f"every molecule with a measured {'/'.join(chembl.POTENCY_TYPES)} "
                            f"against {target.chembl_id} at pChEMBL >= "
                            f"{float(params.get('min_pchembl', 6.0))}, strongest first, "
                            f"capped at {int(params.get('n_actives', 100))}. Where one "
                            "molecule has several measurements the strongest is kept. The "
                            "routing policy chose the cutoff and the cap; the ordering is "
                            "potency and is not the policy's to choose."
                        ),
                        "filter": f"pChEMBL >= {float(params.get('min_pchembl', 6.0))}",
                        "requested": int(params.get("n_actives", 100)),
                        "returned": len(acts),
                        "withheld_by_holdout": withheld,
                        "by_assay": by_assay,
                        "pchembl_range": [
                            round(acts[-1].pchembl, 2),
                            round(acts[0].pchembl, 2),
                        ]
                        if acts
                        else None,
                        "sample": [
                            {
                                "smiles": a.smiles,
                                "pchembl": round(a.pchembl, 2),
                                "assay": a.standard_type,
                                "chembl_id": a.chembl_id,
                            }
                            for a in acts[:12]
                        ],
                    }

        except OfflineCacheMiss as exc:
            return SubsystemResult.failure(
                self.capability.key,
                f"offline replay needs data that was never cached: {exc}",
                Cost(0.0, wall_s=time.perf_counter() - t0),
            )

        # The orchestrator applies this; retrieval is the one subsystem that can change the
        # target the whole run is about.
        if target is not state.target:
            facts["_target_update"] = target.to_dict()

        explain["ran"] = ran
        explain["banks"] = {
            "requests": self.client.n_requests - req0,
            "from_cache": self.client.n_cache_hits - hit0,
        }
        explain["rationale"] = "; ".join(notes) or "nothing to fetch"

        progress("done", 1.0)
        return SubsystemResult(
            subsystem=self.capability.key,
            ok=True,
            records=records,
            facts=facts,
            explain=explain,
            cost=Cost(
                budget_units=0.0,
                wall_s=round(time.perf_counter() - t0, 3),
                n_unique_smiles=len(records),
                api_calls=self.client.n_requests,
                cache_hits=self.client.n_cache_hits,
            ),
            note="; ".join(notes) or "nothing to fetch",
        )

    def lookup_compound(self, name: str) -> str | None:
        """Name to SMILES, for goals that mention a specific drug."""
        try:
            return pubchem.smiles_for_name(self.client, name)
        except Exception:
            return None

    def describe(self) -> dict:
        return {
            "key": self.capability.key,
            "version": self.capability.version,
            "banks": ["uniprot", "rcsb", "chembl", "pubchem"],
            "http": self._client.describe() if self._client else None,
            "offline": self.settings.offline,
        }
