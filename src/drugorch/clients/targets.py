"""One target search across the four banks, so the user never has to visit them.

The thing this replaces is a scientist opening UniProt in one tab to find an accession, RCSB
in another to check a structure exists, and ChEMBL in a third to see whether anyone has ever
measured a compound against it -- then typing the accession back here. All three questions
are answerable from the free APIs, so they are answered here.

Two separate calls, because they have very different costs:

``suggest()``
    Runs on every keystroke. One UniProt request, no fan-out. It returns only what UniProt
    already hands back for free -- gene, name, organism, length, PDB cross-references -- so
    the list stays fast enough to type through.

``readiness()``
    Runs once, when a target is picked. This is the expensive fan-out: it asks RCSB whether
    any structure is actually dockable and ChEMBL how many potent molecules are on record.
    Those two answers decide whether a run can dock at all and whether the generator has
    anything to breed from, which is exactly what a user needs to know *before* spending a
    budget rather than after.

Queries are also classified rather than blindly full-text searched. Someone who types
``7LL4`` means that structure, and someone who types ``CHEMBL2971`` means that target; both
used to return nothing useful from a name search.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import re
from dataclasses import dataclass, field

from . import chembl, rcsb, uniprot
from .http import BankClient

# UniProt accessions have a fixed shape; matching it avoids a useless name search.
ACCESSION_RE = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$", re.I)
PDB_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
CHEMBL_RE = re.compile(r"^CHEMBL[0-9]+$", re.I)


def classify(query: str) -> str:
    """What the user most likely typed: ``accession``, ``pdb``, ``chembl`` or ``text``."""
    q = query.strip()
    if ACCESSION_RE.match(q):
        return "accession"
    if PDB_RE.match(q):
        return "pdb"
    if CHEMBL_RE.match(q):
        return "chembl"
    return "text"


# --- starter targets -----------------------------------------------------------
#
# Shown before anything is typed. A blank autocomplete is a dead end for anyone who does not
# already have a gene name in mind, and "which targets does this thing work on?" is the first
# question a new user actually has. These are well-studied, structurally characterised and
# richly measured, so every one of them gives a run that can dock and generate.

SUGGESTED: tuple[dict, ...] = (
    {"gene": "JAK2", "uniprot": "O60674", "name": "Tyrosine-protein kinase JAK2",
     "why": "kinase; many co-crystals and thousands of measured inhibitors"},
    {"gene": "EGFR", "uniprot": "P00533", "name": "Epidermal growth factor receptor",
     "why": "the canonical oncology kinase target"},
    {"gene": "ABL1", "uniprot": "P00519", "name": "Tyrosine-protein kinase ABL1",
     "why": "imatinib's target; textbook structure-based success"},
    {"gene": "CDK2", "uniprot": "P24941", "name": "Cyclin-dependent kinase 2",
     "why": "small, rigid ATP site that docks reliably"},
    {"gene": "ACE2", "uniprot": "Q9BYF1", "name": "Angiotensin-converting enzyme 2",
     "why": "peptidase; a non-kinase contrast"},
    {"gene": "PTGS2", "uniprot": "P35354", "name": "Prostaglandin G/H synthase 2 (COX-2)",
     "why": "classic NSAID target with a deep, well-defined pocket"},
    {"gene": "BACE1", "uniprot": "P56817", "name": "Beta-secretase 1",
     "why": "aspartic protease; a hard, well-documented target"},
    {"gene": "MAPK14", "uniprot": "Q16539", "name": "Mitogen-activated protein kinase 14 (p38α)",
     "why": "heavily co-crystallised; good for comparing runs"},
)


@dataclass(slots=True)
class TargetHit:
    """One autocomplete row. Everything here is free from the UniProt search response."""

    key: str
    uniprot: str
    name: str
    gene: str
    organism: str
    reviewed: bool
    length: int
    pdb_ids: tuple[str, ...] = ()
    source: str = "uniprot"
    chembl_id: str | None = None
    matched_on: str = ""
    """Why this row is here when the query was not a plain name -- e.g. 'PDB 7LL4'."""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "key": self.key,
            "uniprot": self.uniprot,
            "name": self.name,
            "gene": self.gene,
            "organism": self.organism,
            "reviewed": self.reviewed,
            "length": self.length,
            "pdb_ids": list(self.pdb_ids),
            "n_pdb": len(self.pdb_ids),
            "chembl_id": self.chembl_id,
            "matched_on": self.matched_on,
            "label": f"{self.gene or self.uniprot} — {self.name}" if self.name else (self.gene or self.key),
            "sublabel": _sublabel(self),
        }


def _sublabel(hit: TargetHit) -> str:
    bits = []
    if hit.uniprot:
        bits.append(f"UniProt {hit.uniprot}")
    if hit.organism:
        bits.append(hit.organism)
    if hit.length:
        bits.append(f"{hit.length} aa")
    return " · ".join(bits)


def _from_uniprot(h: uniprot.UniProtHit, *, matched_on: str = "") -> TargetHit:
    return TargetHit(
        key=h.accession,
        uniprot=h.accession,
        name=h.name,
        gene=h.gene,
        organism=h.organism,
        reviewed=h.reviewed,
        length=h.length,
        pdb_ids=h.pdb_ids,
        matched_on=matched_on,
    )


# --- the fast path -------------------------------------------------------------
def suggest(client: BankClient, query: str, limit: int = 8) -> list[TargetHit]:
    """Autocomplete rows for what the user has typed so far.

    One request in the common case. The identifier branches cost a second request only
    because resolving an ID to a protein is the whole point of typing one.
    """
    q = query.strip()
    if not q:
        return []
    kind = classify(q)

    if kind == "pdb":
        hit = _from_pdb(client, q)
        return [hit] if hit else _text_search(client, q, limit)

    if kind == "chembl":
        hit = _from_chembl_id(client, q)
        return [hit] if hit else _text_search(client, q, limit)

    # An accession search and a text search are the same UniProt call; ranking already
    # puts an exact accession match first, so there is nothing to branch on.
    return _text_search(client, q, limit)


def _text_search(client: BankClient, q: str, limit: int) -> list[TargetHit]:
    out: list[TargetHit] = []
    with contextlib.suppress(Exception):
        out = [_from_uniprot(h) for h in uniprot.search(client, q, limit=limit)]
    if len(out) >= 3:
        return out[:limit]

    # ChEMBL knows some target names UniProt's text index does not match well, so it is a
    # fallback rather than a parallel source -- running both every time would double the
    # request rate for results that are almost always worse.
    seen = {h.uniprot for h in out}
    with contextlib.suppress(Exception):
        for t in chembl.search_targets(client, q, limit=3):
            if t.chembl_id in seen:
                continue
            out.append(
                TargetHit(
                    key=t.chembl_id,
                    uniprot="",
                    name=t.name,
                    gene="",
                    organism=t.organism,
                    reviewed=False,
                    length=0,
                    source="chembl",
                    chembl_id=t.chembl_id,
                    matched_on=f"ChEMBL {t.target_type.lower()}" if t.target_type else "ChEMBL",
                )
            )
    return out[:limit]


def _from_pdb(client: BankClient, pdb_id: str) -> TargetHit | None:
    """Turn a structure ID into the protein it contains."""
    acc = rcsb.uniprot_for_pdb(client, pdb_id)
    if not acc:
        return None
    hit = uniprot.resolve(client, acc)
    if hit is None:
        return None
    return _from_uniprot(hit, matched_on=f"structure {pdb_id.upper()}")


def _from_chembl_id(client: BankClient, chembl_id: str) -> TargetHit | None:
    acc = chembl.accession_for_target(client, chembl_id)
    if not acc:
        return None
    hit = uniprot.resolve(client, acc)
    if hit is None:
        return None
    out = _from_uniprot(hit, matched_on=f"ChEMBL {chembl_id.upper()}")
    out.chembl_id = chembl_id.upper()
    return out


# --- the readiness check -------------------------------------------------------
@dataclass(slots=True)
class Readiness:
    """Whether a run on this target can actually do anything, and on what evidence."""

    uniprot: str
    gene: str = ""
    name: str = ""
    organism: str = ""
    length: int = 0
    sequence_ok: bool = False
    chembl_id: str | None = None
    n_actives: int = 0
    structures: list[dict] = field(default_factory=list)
    """The best dockable candidates, already scored -- not every structure that exists."""

    n_pdb_known: int = 0
    """How many PDB entries UniProt cross-references, dockable or not."""

    errors: dict = field(default_factory=dict)

    @property
    def dockable(self) -> list[dict]:
        """Structures that can actually define a docking box.

        An apo structure cannot: the box is computed from the coordinates of the native
        ligand, so with nothing bound there is nothing to centre on. Counting those as
        dockable promised a run that the receptor prep would then refuse.
        """
        return [s for s in self.structures if s.get("ligand_code")]

    @property
    def can_dock(self) -> bool:
        return bool(self.dockable)

    @property
    def can_generate(self) -> bool:
        """GraphGA needs real molecules to cross over; a handful is not a seed set."""
        return self.n_actives >= 10

    def to_dict(self) -> dict:
        return {
            "uniprot": self.uniprot,
            "gene": self.gene,
            "name": self.name,
            "organism": self.organism,
            "length": self.length,
            "chembl_id": self.chembl_id,
            "n_actives": self.n_actives,
            "structures": self.structures,
            "n_pdb_known": self.n_pdb_known,
            "best_structure": self.dockable[0] if self.dockable else None,
            "n_dockable": len(self.dockable),
            "can_dock": self.can_dock,
            "can_generate": self.can_generate,
            "sequence_ok": self.sequence_ok,
            "errors": self.errors,
            "verdict": self.verdict(),
        }

    def verdict(self) -> dict:
        """One sentence a user can act on, plus the severity behind it."""
        if not self.uniprot:
            return {"level": "error", "text": "This target could not be resolved to a protein."}
        best = self.dockable[0] if self.dockable else None
        if best:
            res = f", {best['resolution']:.2f} Å" if best.get("resolution") else ""
            where = f"{best['pdb_id']}{res}, ligand {best['ligand_code']}"
        else:
            where = ""

        if self.can_dock and self.can_generate:
            return {
                "level": "ok",
                "text": (
                    f"Ready. Docking into {where}; {self.n_actives:,} measured actives to "
                    "seed the generator. Every subsystem can run."
                ),
            }
        if self.can_dock and not self.can_generate:
            return {
                "level": "warn",
                "text": (
                    f"Docking is available ({where}), but only {self.n_actives} potent "
                    "molecules are on record — the generator has little to breed from, so a "
                    "run will lean on docking what it is given."
                ),
            }
        if self.can_generate and not self.can_dock:
            return {
                "level": "warn",
                "text": (
                    f"{self.n_actives:,} measured actives, but no structure with a bound ligand "
                    "was found — docking will be unavailable and the run will have no "
                    "objective to optimise."
                ),
            }
        return {
            "level": "warn",
            "text": (
                "No dockable structure and almost no measured activity. A run will resolve "
                "the target and then have little to do."
            ),
        }


def readiness(client: BankClient, query: str, *, keep_structures: int = 3) -> Readiness:
    """Everything needed to judge a target, gathered in parallel.

    The three banks are independent, so they are asked at once: serially this is the slowest
    thing in the UI, and it sits between picking a target and being able to start.
    """
    kind = classify(query)
    hit = None
    if kind == "pdb":
        acc = rcsb.uniprot_for_pdb(client, query)
        if acc:
            hit = uniprot.resolve(client, acc)
    elif kind == "chembl":
        acc = chembl.accession_for_target(client, query)
        if acc:
            hit = uniprot.resolve(client, acc)
    if hit is None:
        with contextlib.suppress(Exception):
            hit = uniprot.resolve(client, query)
    if hit is None:
        return Readiness(uniprot="", errors={"uniprot": "no match"})

    out = Readiness(
        uniprot=hit.accession,
        gene=hit.gene,
        name=hit.name,
        organism=hit.organism,
        length=hit.length,
        n_pdb_known=len(hit.pdb_ids),
    )

    def _structures() -> list[dict]:
        cands = rcsb.shortlist(
            client, hit.accession, fallback_ids=hit.pdb_ids, max_examined=6, keep=keep_structures
        )
        return [c.to_dict() for c in cands]

    def _bioactivity() -> tuple[str | None, int]:
        t = chembl.target_for_uniprot(client, hit.accession)
        if t is None:
            return None, 0
        return t.chembl_id, chembl.count_actives(client, t.chembl_id)

    def _sequence() -> bool:
        return bool(uniprot.sequence(client, hit.accession))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        f_struct = pool.submit(_structures)
        f_bio = pool.submit(_bioactivity)
        f_seq = pool.submit(_sequence)

        # Each bank fails independently: one being down should degrade the verdict, not
        # erase the other two answers.
        try:
            out.structures = f_struct.result()
        except Exception as exc:  # noqa: BLE001
            out.errors["rcsb"] = f"{type(exc).__name__}: {exc}"
        try:
            out.chembl_id, out.n_actives = f_bio.result()
        except Exception as exc:  # noqa: BLE001
            out.errors["chembl"] = f"{type(exc).__name__}: {exc}"
        try:
            out.sequence_ok = f_seq.result()
        except Exception as exc:  # noqa: BLE001
            out.errors["uniprot"] = f"{type(exc).__name__}: {exc}"

    return out


__all__ = [
    "SUGGESTED",
    "Readiness",
    "TargetHit",
    "classify",
    "readiness",
    "suggest",
]
