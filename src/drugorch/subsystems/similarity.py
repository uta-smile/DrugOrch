"""Searching the banks *with a molecule* instead of with a protein.

:mod:`drugorch.subsystems.retrieval` asks a target-shaped question -- "what is known to bind
JAK2" -- and that is the only way molecules entered a run. It has a blind spot that matters
most for exactly the molecules this framework is meant to produce: a generated structure has
no activity record, so no target-shaped query can say anything about it. It is a string that
scored well, and nothing in the run can tell whether it resembles a marketed drug, a compound
withdrawn for toxicity, or nothing anyone has ever made.

This subsystem closes that loop. It takes molecules *out of the run* -- generated ones by
default -- and asks the banks what catalogued chemistry sits near them. That makes the
generate step checkable against reality rather than only against a docking score, and it is
the one path by which generation feeds back into retrieval.

Three things are deliberate:

**The similarity that is filtered on is computed here, not quoted.** ChEMBL returns its own
similarity figure and PubChem returns none at all. Ranking a mixed result set on "whatever
each bank said" would produce one column of numbers meaning two different things, and a
threshold would silently mean something different per source. So every hit is re-scored with
this project's own fingerprint (:mod:`drugorch.fingerprints`, ECFP4/2048) and the bank's own
figure is carried alongside as a cross-check, never as the filter.

**Nothing enters the run without passing a stated gate.** A molecule pulled off the open
internet on the strength of a shape match is not automatically a thing worth spending docking
budget on, and some of them are documented harms -- ChEMBL knows which compounds were
withdrawn and which carry a black-box warning, and that is recorded fact rather than
prediction. :data:`SAFETY_RULES` declares every gate, what each one does in each mode, and
every rejection is counted and reported. A filter you cannot see is a filter you cannot
disagree with.

**Which molecules were used as queries is reported, not inferred.** The complaint this
answers is that a retrieval step says "retrieved 100 molecules" and leaves a reader unable to
say what was asked for or why those came back. So ``explain`` names each query molecule, where
in the run it came from, what it found, and what the selection rule was.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..chem import canonical, to_mol
from ..clients import chembl, pubchem
from ..clients.http import BankClient, OfflineCacheMiss
from ..clients.neighbour import Neighbour
from ..config import Settings, get_settings
from ..fingerprints import max_similarity, mean_pairwise_distance
from ..versions import HTTP_V, SIM_V
from .base import BaseSubsystem, Capability, Cost, ParamSpec, ProgressFn, SubsystemResult
from .properties import compute_panel, lipinski_violations

if TYPE_CHECKING:  # pragma: no cover
    from ..orchestrator.state import RunState

QUERY_CHOICES = ("generated", "best", "actives", "unscored", "explicit")
SOURCE_CHOICES = ("chembl", "pubchem", "both")
SAFETY_CHOICES = ("strict", "standard", "off")

#: Elements a small molecule in this system may contain. Anything else -- a metal centre, a
#: radionuclide, a counter-ion the standardizer did not strip -- is out of scope for Vina and
#: for the property panel alike, so it is refused at the door rather than failing later in a
#: way that reads as a bug in docking.
ORGANIC = frozenset("H B C N O F Si P S Cl Se Br I".split())

MIN_HEAVY, MAX_HEAVY = 8, 70

SEARCHED_MARKER = "searched_banks"
"""Prop stamped on every molecule this subsystem has used as a query.

Without it a molecule whose search returned nothing stays indistinguishable from one that
was never searched, so it is searched again on the next call, and again -- a loop that only
the orchestrator's stuck-detector ended, four wasted calls later. A search that found
nothing is a result, and it has to be recorded as one. The marker carries the capability
version so that changing the banks or the gate makes an old search count as not done, for
the same reason the property panel's marker does.
"""


@dataclass(frozen=True, slots=True)
class SafetyRule:
    """One gate a candidate must pass, and what each mode does with a failure.

    ``action`` maps a mode name to ``"reject"``, ``"flag"`` or ``"allow"``. The table is data
    so that the same declaration drives the filtering, the count in ``explain`` and the
    description the UI shows -- a rule described in prose next to code that does something
    else is the failure this shape exists to prevent.
    """

    key: str
    title: str
    why: str
    action: dict[str, str]

    def verdict(self, mode: str) -> str:
        return self.action.get(mode, "allow")

    def to_dict(self) -> dict:
        return {"key": self.key, "title": self.title, "why": self.why, "action": self.action}


_ALWAYS = {"strict": "reject", "standard": "reject", "off": "reject"}
_RELAXABLE = {"strict": "reject", "standard": "flag", "off": "allow"}

SAFETY_RULES: tuple[SafetyRule, ...] = (
    SafetyRule(
        "parse",
        "Unparseable structure",
        "The bank returned something RDKit cannot read. It cannot be docked or profiled.",
        _ALWAYS,
    ),
    SafetyRule(
        "element",
        "Non-organic element",
        f"Contains an atom outside {{{', '.join(sorted(ORGANIC))}}} -- a metal complex or an "
        "unstripped counter-ion. Out of scope for both the force field and the property panel.",
        _ALWAYS,
    ),
    SafetyRule(
        "size",
        "Outside the small-molecule size window",
        f"Fewer than {MIN_HEAVY} or more than {MAX_HEAVY} heavy atoms: a fragment or a "
        "peptide-sized molecule, neither of which this pipeline scores meaningfully.",
        _ALWAYS,
    ),
    SafetyRule(
        "threshold",
        "Below the similarity floor on our own metric",
        "The bank called it a hit but the locally recomputed Tanimoto disagrees. The usual "
        "cause is a salt or co-crystal: ChEMBL matched the active component, and keeping "
        "the largest fragment left the counter-ion -- the aspirin-lysine salt reduces to "
        "lysine, which is not what matched. The number filtered on is the one computed here.",
        _ALWAYS,
    ),
    SafetyRule(
        "withdrawn",
        "Withdrawn from market",
        "ChEMBL records this compound as withdrawn. That is regulatory history, not a "
        "prediction, and it is the strongest negative signal either bank carries.",
        {"strict": "reject", "standard": "reject", "off": "flag"},
    ),
    SafetyRule(
        "black_box",
        "Carries a black-box warning",
        "Approved, but with the most serious warning a label can carry. Worth knowing before "
        "a generated analogue of it is pursued.",
        _RELAXABLE,
    ),
    SafetyRule(
        "pains",
        "PAINS substructure",
        "Matches a pan-assay interference pattern: a motif that produces apparent activity "
        "across unrelated assays. A flag on the experiment, not on the chemistry.",
        _RELAXABLE,
    ),
    SafetyRule(
        "lipinski",
        "Two or more Lipinski violations",
        "A developability heuristic, not a law -- which is why it only rejects in strict mode.",
        _RELAXABLE,
    ),
)

RULES_BY_KEY = {r.key: r for r in SAFETY_RULES}

CAPABILITY = Capability(
    key="similarity",
    title="Molecule-first bank search (ChEMBL, PubChem)",
    summary=(
        "Search the chemical databases USING a molecule from this run rather than using the "
        "target. Given generated molecules, it returns catalogued compounds that are "
        "structurally near them, each with a locally computed Tanimoto and -- from ChEMBL -- "
        "how far it got clinically and whether it was withdrawn. This is the only way to find "
        "out whether a generated molecule resembles real chemistry or nothing at all, so it "
        "is the natural follow-up to a generation round. Cheap relative to docking. Returns "
        "real molecules that can then be docked or profiled like any other."
    ),
    requires=(),
    produces=("neighbours", "catalogued_analogues"),
    params=(
        ParamSpec(
            name="query_from",
            type="enum",
            default="generated",
            choices=QUERY_CHOICES,
            doc="Which molecules in the run to search WITH. 'generated' uses the newest "
            "molecules the generator made, 'best' the top-scoring ones, 'actives' the "
            "retrieved known binders, 'unscored' those with no objective score yet, "
            "'explicit' whatever is passed in 'smiles'.",
        ),
        ParamSpec(
            name="smiles",
            type="smiles_list",
            default=[],
            doc="Query molecules, used only when query_from is 'explicit'.",
        ),
        ParamSpec(
            name="source",
            type="enum",
            default="chembl",
            choices=SOURCE_CHOICES,
            doc="Which bank to search. ChEMBL knows clinical phase and withdrawal; PubChem "
            "is far broader but knows nothing about a compound's fate.",
        ),
        ParamSpec(
            name="min_similarity",
            type="float",
            default=0.7,
            lo=0.4,
            hi=1.0,
            doc="Tanimoto floor, 0-1. Below 0.4 the banks refuse the query.",
        ),
        ParamSpec(
            name="max_queries",
            type="int",
            default=5,
            lo=1,
            hi=20,
            doc="How many molecules from the run to search with. Each one is a bank request.",
        ),
        ParamSpec(
            name="max_results",
            type="int",
            default=40,
            lo=1,
            hi=200,
            doc="Ceiling on molecules admitted to the run, after the safety gate.",
        ),
        ParamSpec(
            name="safety",
            type="enum",
            default="standard",
            choices=SAFETY_CHOICES,
            doc="How strictly to gate what enters the run. 'strict' rejects PAINS, "
            "black-box and Lipinski failures; 'standard' rejects withdrawn compounds and "
            "flags the rest; 'off' keeps only the structural-integrity checks, which are "
            "never skippable.",
        ),
    ),
    unit_cost=0.4,
    per="call",
    typical_wall_s=12.0,
    version=f"{SIM_V}+{HTTP_V}",
)


class SimilaritySubsystem(BaseSubsystem):
    """Takes molecules out of the run, asks the banks what sits near them, gates the answer."""

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

    # --- query selection -------------------------------------------------------
    def _queries(self, params: dict, state: RunState) -> tuple[list[str], str]:
        """The molecules to search with, and the rule that chose them -- in words.

        The rule is returned rather than left implicit because "retrieval ran and 100
        molecules appeared" is exactly the opacity this subsystem exists to not repeat. A
        reader should be able to see which molecules went out to the bank and on what basis
        without reading this function.
        """
        mode = params.get("query_from", "generated")
        n = int(params.get("max_queries", 5))

        if mode == "explicit":
            given = [c for c in (canonical(s) for s in (params.get("smiles") or [])) if c]
            return given[:n], f"the {len(given[:n])} molecules named in the call"

        def fresh(smiles: list[str]) -> list[str]:
            """Drop molecules already searched under this version. See SEARCHED_MARKER."""
            return [
                s
                for s in smiles
                if state.molecules[s].props.get(SEARCHED_MARKER) != self.capability.version
            ]

        # Everything below reads keys out of `state.molecules`, which are canonical by the
        # time a subsystem writes them. They are not re-canonicalised: a query that differed
        # from its own key would make the "already in the run" check miss and re-admit a
        # molecule the run already had.
        if mode == "best":
            picked = fresh([s for s, _ in state.scored()])[:n]
            return picked, (
                f"the {len(picked)} highest-scoring not-yet-searched molecules, by "
                f"{state.objective_key}"
            )

        if mode == "actives":
            picked = [s for s, r in state.molecules.items() if r.origin == "retrieval"]
            picked.sort(
                key=lambda s: state.molecules[s].scores.get("pchembl", 0.0), reverse=True
            )
            picked = fresh(picked)[:n]
            return picked, (
                f"the {len(picked)} most potent not-yet-searched retrieved actives, by "
                "measured pChEMBL"
            )

        if mode == "unscored":
            picked = fresh(
                [
                    s
                    for s, r in state.molecules.items()
                    if r.scores.get(state.objective_key) is None
                ]
            )[-n:]
            return picked, (
                f"the {len(picked)} most recently added molecules with no "
                f"{state.objective_key} yet, excluding any already searched"
            )

        # generated: newest first, because the point is to check what was just made.
        gen = fresh([s for s, r in state.molecules.items() if r.origin == "generation"])
        picked = gen[-n:][::-1]
        return picked, f"the {len(picked)} most recently generated, not-yet-searched molecules"

    def preflight(self, params: dict, state: RunState) -> tuple[bool, str]:
        queries, rule = self._queries(params, state)
        if queries:
            return True, ""

        # Distinguish "there is nothing of this kind" from "there is, and it has all been
        # searched already". Both block the call; only the first is a sequencing problem the
        # policy can fix by calling something else first, and a policy told the wrong one
        # will keep trying.
        mode = params.get("query_from", "generated")
        pools = {
            "generated": [s for s, r in state.molecules.items() if r.origin == "generation"],
            "actives": [s for s, r in state.molecules.items() if r.origin == "retrieval"],
            "best": [s for s, _ in state.scored()],
        }
        pool = pools.get(mode)
        if pool:
            return False, (
                f"all {len(pool)} candidate molecules have already been searched against the "
                "banks; generate or retrieve more first"
            )
        if mode == "generated":
            return False, "no generated molecules yet; run generation first"
        if mode == "actives":
            return False, "no retrieved actives to search with; fetch actives first"
        if mode == "best":
            return False, "nothing is scored yet, so there is no 'best' to search with"
        return False, f"no query molecules available ({rule})"

    def estimate_cost(self, params: dict, state: RunState) -> Cost:
        queries, _ = self._queries(params, state)
        banks = 2 if params.get("source", "chembl") == "both" else 1
        n_req = len(queries) * banks
        return Cost(
            budget_units=self.capability.unit_cost * max(1, len(queries)),
            wall_s=2.5 * n_req,
            api_calls=n_req,
        )

    # --- the call --------------------------------------------------------------
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
        queries, rule = self._queries(params, state)
        source = params.get("source", "chembl")
        floor = float(params.get("min_similarity", 0.7))
        mode = str(params.get("safety", "standard"))
        cap_results = int(params.get("max_results", 40))
        per_query = max(5, min(50, cap_results))

        req0, hit0 = self.client.n_requests, self.client.n_cache_hits
        banks = ("chembl", "pubchem") if source == "both" else (source,)

        # smiles -> (neighbour, best tanimoto, the query it was nearest to)
        found: dict[str, tuple[Neighbour, float, str]] = {}
        per_query_report: list[dict] = []

        try:
            for i, q in enumerate(queries):
                if cancel.is_set():
                    break
                progress(f"searching {source} near query {i + 1}/{len(queries)}",
                         (i + 1) / (len(queries) + 1))
                hits: list[Neighbour] = []
                for bank in banks:
                    fn = chembl.similar if bank == "chembl" else pubchem.similar
                    hits.extend(fn(self.client, q, threshold=floor, limit=per_query))

                # One metric for every hit from every bank, computed here. See the module
                # docstring: the banks do not agree on what similarity means and one of them
                # does not report it at all.
                sims = self._tanimoto_to(q, [h.smiles for h in hits])
                best = 0.0
                for h in hits:
                    sim = sims.get(h.smiles, 0.0)
                    best = max(best, sim)
                    prior = found.get(h.smiles)
                    if prior is None or sim > prior[1]:
                        found[h.smiles] = (h, sim, q)

                rec = state.molecules.get(q)
                per_query_report.append(
                    {
                        "smiles": q,
                        "origin": rec.origin if rec else "unknown",
                        "score": _objective_of(state, q),
                        "parents": list(rec.parents) if rec else [],
                        "n_hits": len(hits),
                        "best_tanimoto": round(best, 3),
                    }
                )
        except OfflineCacheMiss as exc:
            return SubsystemResult.failure(
                self.capability.key,
                f"offline replay needs a bank response that was never cached: {exc}",
                Cost(0.0, wall_s=time.perf_counter() - t0),
            )

        progress("applying the safety gate", 0.9)
        kept, ledger, examples = self._gate(found, state, floor, mode)
        kept.sort(key=lambda row: row["tanimoto"], reverse=True)
        # Kept out of the safety ledger deliberately. "Turned away by a rule" and "fell off
        # the end of max_results" are different facts, and summing them would make the
        # rejection counts impossible to reconcile against the hit count.
        over_cap = max(0, len(kept) - cap_results)
        kept = kept[:cap_results]

        # Every query molecule is stamped as searched, including the ones that found
        # nothing. A search that came back empty is a result about that molecule, and
        # leaving it unmarked is what made this subsystem search the same five molecules on
        # every call until the run's stuck-detector stopped it.
        records = [
            {
                "smiles": q,
                "props": {SEARCHED_MARKER: self.capability.version},
                "scores": {},
            }
            for q in queries
        ]
        records += [
            {
                "smiles": row["smiles"],
                "props": {
                    "bank": row["source"],
                    "bank_id": row["bank_id"],
                    "bank_name": row["name"],
                    "tanimoto_to_query": row["tanimoto"],
                    "nearest_query": row["nearest_query"],
                    "max_phase": row["max_phase"],
                    "withdrawn": row["withdrawn"],
                    "black_box": row["black_box"],
                    "safety_flags": row["flags"],
                },
                "scores": {},
                "parents": [row["nearest_query"]],
            }
            for row in kept
        ]

        metrics = self._metrics(kept, state)
        # Duplicates are excluded: a molecule the run already had was not "turned away by
        # the gate", and counting it as such makes the number irreconcilable with the rules
        # table beside it, where duplicate is explicitly not a safety rule.
        n_refused = sum(n for key, n in ledger.items() if key != "duplicate")
        note = (
            f"{len(kept)} catalogued molecules near {len(queries)} query molecule"
            f"{'' if len(queries) == 1 else 's'}"
            + (f"; {metrics['n_clinical']} reached a clinical phase" if metrics["n_clinical"] else "")
            + (f"; {n_refused} rejected by the safety gate" if n_refused else "")
            + (f"; {ledger['duplicate']} already in the run" if ledger.get("duplicate") else "")
        )

        explain = {
            "rationale": note,
            "queries": {
                "mode": params.get("query_from", "generated"),
                "rule": rule,
                "available": _pool_sizes(state),
                "used": per_query_report,
            },
            "search": {
                "banks": list(banks),
                "min_similarity": floor,
                "metric": "Tanimoto on ECFP4/2048, recomputed locally for every hit",
                "requested_per_query": per_query,
                "raw_hits": len(found),
                "max_results": cap_results,
                "dropped_over_cap": over_cap,
            },
            "safety": {
                "mode": mode,
                "rejected": {k: v for k, v in ledger.items() if v},
                "rules": [
                    {**r.to_dict(), "verdict": r.verdict(mode), "rejected": ledger.get(r.key, 0)}
                    for r in SAFETY_RULES
                ],
                "examples": examples,
            },
            "kept": kept,
            "metrics": metrics,
            "banks": {
                "requests": self.client.n_requests - req0,
                "from_cache": self.client.n_cache_hits - hit0,
            },
        }

        return SubsystemResult(
            subsystem=self.capability.key,
            ok=True,
            records=records,
            facts={
                "neighbours": {
                    "n": len(kept),
                    "source": source,
                    "min_similarity": floor,
                    "n_clinical": metrics["n_clinical"],
                }
            }
            if kept
            else {},
            explain=explain,
            cost=Cost(
                budget_units=0.0,
                # The molecules admitted, not `len(records)` -- that also counts the query
                # molecules, which were already in the run and are only being stamped.
                wall_s=round(time.perf_counter() - t0, 3),
                n_unique_smiles=len(kept),
                api_calls=self.client.n_requests - req0,
                cache_hits=self.client.n_cache_hits - hit0,
            ),
            note=note,
        )

    # --- helpers ---------------------------------------------------------------
    @staticmethod
    def _tanimoto_to(query: str, candidates: list[str]) -> dict[str, float]:
        """Tanimoto from one query to many candidates, on this project's fingerprint."""
        if not candidates:
            return {}
        return max_similarity(candidates, [query])

    def _gate(
        self,
        found: dict[str, tuple[Neighbour, float, str]],
        state: RunState,
        floor: float,
        mode: str,
    ) -> tuple[list[dict], dict[str, int], list[dict]]:
        """Apply :data:`SAFETY_RULES`. Returns what survived, what was rejected, and why.

        A rejection is counted under the *first* rule it trips, so the counts sum to the
        number of molecules turned away rather than to the number of rule violations -- the
        first is the number a reader is actually trying to reconcile against the hit count.
        """
        ledger: dict[str, int] = {}
        examples: list[dict] = []
        kept: list[dict] = []

        def refuse(key: str, smi: str, detail: str) -> None:
            ledger[key] = ledger.get(key, 0) + 1
            if len(examples) < 8:
                examples.append({"smiles": smi, "rule": key, "detail": detail})

        # Structural integrity is checked before the similarity floor, because a molecule
        # that cannot be read cannot be fingerprinted either -- it scores Tanimoto 0 and
        # would be filed under "below threshold", which names the wrong problem. The
        # rejection counts are a diagnosis, so each one has to be the real reason.
        for smi, (hit, sim, query) in found.items():
            if smi in state.molecules:
                ledger["duplicate"] = ledger.get("duplicate", 0) + 1
                continue

            mol = to_mol(smi)
            if mol is None:
                refuse("parse", smi, "RDKit could not read the structure")
                continue

            bad = sorted(
                {a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() not in ORGANIC}
            )
            if bad:
                refuse("element", smi, f"contains {', '.join(bad)}")
                continue

            heavy = mol.GetNumHeavyAtoms()
            if heavy < MIN_HEAVY or heavy > MAX_HEAVY:
                refuse("size", smi, f"{heavy} heavy atoms")
                continue

            if sim < floor:
                refuse("threshold", smi, f"recomputed Tanimoto {sim:.2f} < {floor:.2f}")
                continue

            panel = compute_panel(smi)
            if panel is None:
                refuse("parse", smi, "the property panel could not be computed")
                continue

            flags: list[str] = []
            rejected = False
            for key, tripped, detail in (
                ("withdrawn", hit.withdrawn, "ChEMBL records this compound as withdrawn"),
                ("black_box", hit.black_box, "carries a black-box warning"),
                (
                    "pains",
                    bool(panel["flags"]["pains_alert"]),
                    str(panel["flags"]["pains_alert"]),
                ),
                (
                    "lipinski",
                    lipinski_violations(panel["descriptors"]) >= 2,
                    f"{lipinski_violations(panel['descriptors'])} rule-of-five violations",
                ),
            ):
                if not tripped:
                    continue
                verdict = RULES_BY_KEY[key].verdict(mode)
                if verdict == "reject":
                    refuse(key, smi, detail)
                    rejected = True
                    break
                if verdict == "flag":
                    flags.append(key)
            if rejected:
                continue

            kept.append(
                {
                    "smiles": smi,
                    "source": hit.source,
                    "bank_id": hit.bank_id,
                    "name": hit.name,
                    "tanimoto": round(float(sim), 3),
                    "bank_similarity": (
                        round(hit.bank_similarity, 3)
                        if hit.bank_similarity is not None
                        else None
                    ),
                    "nearest_query": query,
                    "max_phase": hit.max_phase,
                    "withdrawn": hit.withdrawn,
                    "black_box": hit.black_box,
                    "first_approval": hit.first_approval,
                    "qed": panel["scores"]["qed"],
                    "sa_score": panel["scores"]["sa_score"],
                    "mw": panel["descriptors"]["mw"],
                    "flags": flags,
                }
            )

        return kept, ledger, examples

    @staticmethod
    def _metrics(kept: list[dict], state: RunState) -> dict:
        """Numbers that say whether this search found anything worth having.

        ``resemblance_to_actives`` is the one that carries information. Everything here is
        near a query molecule by construction, so "is it novel" is not a question this
        search can answer -- but "does the catalogued chemistry around our generated
        molecules overlap what is already known to bind this target" is, and a high value
        means the generator is circling known actives rather than finding new ground.
        """
        sims = [row["tanimoto"] for row in kept]
        actives = [s for s, r in state.molecules.items() if r.origin == "retrieval"]
        smiles = [row["smiles"] for row in kept]
        resemblance = None
        if smiles and actives:
            near = max_similarity(smiles, actives)
            if near:
                resemblance = round(statistics.fmean(near.values()), 3)
        return {
            "n_kept": len(kept),
            "n_clinical": sum(
                1 for r in kept if r["max_phase"] is not None and r["max_phase"] >= 1
            ),
            "n_approved": sum(
                1 for r in kept if r["max_phase"] is not None and r["max_phase"] >= 4
            ),
            "n_flagged": sum(1 for r in kept if r["flags"]),
            "tanimoto": {
                "min": round(min(sims), 3),
                "median": round(statistics.median(sims), 3),
                "max": round(max(sims), 3),
            }
            if sims
            else None,
            "diversity": round(mean_pairwise_distance(smiles), 3) if len(smiles) > 1 else None,
            "resemblance_to_actives": resemblance,
        }

    def urgency(self, state: RunState, facts: dict) -> tuple[float, str]:
        """Wants to run right after generation has produced molecules nothing knows about.

        Rule-based arms score by name and would otherwise never reach this subsystem at
        all -- see :meth:`drugorch.subsystems.base.Subsystem.urgency`.
        """
        gen = [s for s, r in state.molecules.items() if r.origin == "generation"]
        if not gen:
            return 0.05, "nothing has been generated yet, so there is nothing to look up"
        searched = [
            s
            for s in gen
            if state.molecules[s].props.get(SEARCHED_MARKER) == self.capability.version
        ]
        unknown = [s for s in gen if s not in set(searched)]
        if not unknown:
            return 0.1, "every generated molecule has already been searched against the banks"

        # Searching has to be allowed to stop being worth doing. A generator that has moved
        # off catalogued chemistry will keep returning nothing however many more molecules
        # are thrown at the banks, and the evidence for that is already in the run: a dozen
        # molecules searched with no neighbour admitted is an answer, not a reason to ask
        # again. Without this the only thing that ended the sequence was the orchestrator's
        # stuck-detector, three wasted calls later, and the trace said "stuck" where it
        # should have said "looked, and there is nothing there".
        found = sum(1 for r in state.molecules.values() if r.origin == self.capability.key)
        if len(searched) >= 10 and found == 0:
            return 0.12, (
                f"{len(searched)} generated molecules have been searched and none had a "
                "catalogued neighbour; this generator has moved off known chemistry and "
                "searching more of it is unlikely to pay"
            )
        return 0.55, (
            f"{len(unknown)} generated molecules have no catalogued analogue on record; "
            "a bank search would say whether they resemble real chemistry"
        )

    def describe(self) -> dict:
        return {
            "key": self.capability.key,
            "version": self.capability.version,
            "banks": ["chembl", "pubchem"],
            "metric": "ECFP4/2048 Tanimoto, computed locally",
            "safety_rules": [r.to_dict() for r in SAFETY_RULES],
            "http": self._client.describe() if self._client else None,
            "offline": self.settings.offline,
        }


def _objective_of(state: RunState, smiles: str) -> float | None:
    rec = state.molecules.get(smiles)
    if rec is None:
        return None
    v = rec.scores.get(state.objective_key)
    return round(float(v), 4) if v is not None else None


def _pool_sizes(state: RunState) -> dict:
    """How many molecules each query mode had to choose from, so the rule can be checked."""
    by_origin: dict[str, int] = {}
    for rec in state.molecules.values():
        by_origin[rec.origin] = by_origin.get(rec.origin, 0) + 1
    return {
        "generated": by_origin.get("generation", 0),
        "actives": by_origin.get("retrieval", 0),
        "scored": len(state.scored()),
        "total": len(state.molecules),
    }


__all__ = [
    "CAPABILITY",
    "ORGANIC",
    "SAFETY_RULES",
    "SafetyRule",
    "SimilaritySubsystem",
]
