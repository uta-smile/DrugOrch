"""RCSB PDB: finding a structure that is actually dockable.

Most structures for a given protein are useless for docking. The job here is to rank the
candidates by whether they will survive receptor preparation and produce a meaningful box:

* **X-ray, good resolution.** NMR models have no single conformation and cryo-EM structures
  of a whole complex rarely resolve a small-molecule site well.
* **A bound drug-like ligand.** This matters more than resolution. A co-crystallized ligand
  tells us where the pocket is -- the docking box is derived from it -- and it tells us the
  site was in a ligand-binding conformation when it was solved. An apo structure leaves us
  guessing at both.
* **Not a buffer component.** Crystallization additives, ions and sugars are everywhere in
  the PDB and none of them mark a drug site, so they are filtered by identity and by size.

The whole shortlist is kept, not just the winner, because receptor preparation fails often
enough that "try the next structure" is a routing decision the orchestrator genuinely makes.
"""

from __future__ import annotations

from ..types import PdbCandidate
from .http import BankClient

SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{}"
NONPOLY = "https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{}/{}"
CIF_FILE = "https://files.rcsb.org/download/{}.cif"
PDB_FILE = "https://files.rcsb.org/download/{}.pdb"

# Crystallization additives, cryoprotectants, ions and common sugars. A hit on any of these
# is not a drug site, however well resolved it is.
BUFFER_CODES = frozenset(
    ["HOH", "DOD", "SO4", "PO4", "GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "2PE", "P6G", "MPD", "MES", "EPE", "TRS", "BTB", "BIS", "CIT", "ACT", "ACY", "FMT", "OXL", "TAR", "MLI", "SCN", "AZI", "NO3", "CO3", "BCT", "NH4", "IMD", "DMS", "DMF", "ACN", "EOH", "IPA", "MOH", "TFA", "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "FE2", "CU", "CU1", "NI", "CO", "CD", "HG", "AU", "PT", "PB", "BA", "SR", "CS", "RB", "LI", "BR", "IOD", "IUM", "F", "NAG", "NDG", "BMA", "MAN", "GAL", "GLC", "FUC", "XYS", "SIA", "BGC", "A2G", "ATP", "ADP", "AMP", "GTP", "GDP", "GMP", "NAD", "NAP", "NDP", "FAD", "FMN", "SAM", "SAH", "COA", "MYR", "PLM", "OLA", "STE", "LDA", "LMT", "C8E", "OCT", "HEZ", "BOG", "DDQ", "UNX", "UNL"]
)

MIN_LIGAND_HEAVY_ATOMS = 10
"""Below this, a 'ligand' is an ion or a fragment, not something that defines a pocket."""


def _search_payload(accession: str, limit: int) -> dict:
    """Structures of this exact protein, solved by X-ray, carrying a non-polymer entity."""
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers."
                        "reference_sequence_identifiers.database_accession",
                        "operator": "exact_match",
                        "value": accession.upper(),
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers."
                        "reference_sequence_identifiers.database_name",
                        "operator": "exact_match",
                        "value": "UniProt",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                        "operator": "greater",
                        "value": 0,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": limit},
            "sort": [
                {"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}
            ],
        },
    }


def search_by_uniprot(client: BankClient, accession: str, limit: int = 25) -> list[str]:
    """PDB IDs for this accession, best resolution first. Empty list if none."""
    try:
        data = client.post(SEARCH, body=_search_payload(accession, limit))
    except Exception:
        # The search service is the most failure-prone of the four banks; a caller can
        # always fall back to the PDB cross-references UniProt already returned.
        return []
    return [r["identifier"] for r in (data or {}).get("result_set", []) if r.get("identifier")]


def entry_info(client: BankClient, pdb_id: str) -> dict | None:
    return client.get(ENTRY.format(pdb_id.strip().upper()))


def _heavy_atoms_from_weight(formula_weight: float | None) -> int:
    """Estimate heavy-atom count from molecular weight.

    RCSB reports ``formula_weight`` on a non-polymer entity in **kilodaltons** (a 578 Da
    ligand comes back as ``0.578``), which is easy to miss and silently filters out every
    real ligand if you assume daltons. Anything under 10 is therefore read as kDa. The
    divisor is the average mass per heavy atom in a typical organic ligand.
    """
    if not formula_weight:
        return 0
    da = formula_weight * 1000.0 if formula_weight < 10.0 else formula_weight
    return int(da / 13.0)


def _ligands(client: BankClient, pdb_id: str, entry: dict) -> list[tuple[str, int, bool]]:
    """``(comp_id, heavy_atoms, is_subject_of_investigation)`` per non-buffer entity.

    RCSB annotates the ligand a structure was actually solved to study with
    ``SUBJECT_OF_INVESTIGATION``. That is a curator's judgement about which molecule matters,
    which beats any heuristic we could apply over size alone -- so it is carried through and
    weighted heavily in the ranking.
    """
    ids = (entry.get("rcsb_entry_container_identifiers") or {}).get("non_polymer_entity_ids") or []
    out: list[tuple[str, int, bool]] = []
    for eid in ids[:12]:
        try:
            ent = client.get(NONPOLY.format(pdb_id.upper(), eid))
        except Exception:
            continue
        if not ent:
            continue
        comp = (ent.get("pdbx_entity_nonpoly") or {}).get("comp_id") or (
            ent.get("rcsb_nonpolymer_entity_container_identifiers") or {}
        ).get("nonpolymer_comp_id", "")
        if not comp or comp.upper() in BUFFER_CODES:
            continue
        weight = (ent.get("rcsb_nonpolymer_entity") or {}).get("formula_weight")
        subject = any(
            a.get("type") == "SUBJECT_OF_INVESTIGATION"
            for a in (ent.get("rcsb_nonpolymer_entity_annotation") or [])
        )
        out.append((comp, _heavy_atoms_from_weight(weight), subject))
    return out


def describe_entry(client: BankClient, pdb_id: str) -> PdbCandidate | None:
    """Metadata and ligand content for one structure, as a scored candidate."""
    entry = entry_info(client, pdb_id)
    if not entry:
        return None

    info = entry.get("rcsb_entry_info") or {}
    res_list = info.get("resolution_combined") or []
    resolution = float(res_list[0]) if res_list else None
    methods = [m.get("method", "") for m in (entry.get("exptl") or [])]
    method = methods[0] if methods else ""
    title = (entry.get("struct") or {}).get("title", "")

    ligs = [(c, n, s) for c, n, s in _ligands(client, pdb_id, entry) if n >= MIN_LIGAND_HEAVY_ATOMS]
    # Prefer the curated subject of investigation; fall back to the largest ligand.
    ligs.sort(key=lambda t: (0 if t[2] else 1, -t[1]))
    lig_code, lig_heavy, lig_subject = ligs[0] if ligs else ("", 0, False)

    return PdbCandidate(
        pdb_id=pdb_id.upper(),
        method=method,
        resolution=resolution,
        title=title[:180],
        ligand_code=lig_code,
        ligand_heavy_atoms=lig_heavy,
        score=_score(method, resolution, lig_heavy, lig_subject),
        reason=_reason(method, resolution, lig_code, lig_heavy, lig_subject),
    )


def _score(method: str, resolution: float | None, lig_heavy: int, lig_subject: bool) -> float:
    """Higher is better. A bound ligand outweighs half an angstrom of resolution."""
    s = 0.0
    if "X-RAY" in method.upper():
        s += 2.0
    if resolution is not None:
        s += max(0.0, 3.5 - resolution) * 1.5
    if lig_heavy >= MIN_LIGAND_HEAVY_ATOMS:
        s += 4.0 + min(lig_heavy, 45) / 15.0
    if lig_subject:
        s += 2.0
    return round(s, 3)


def _reason(
    method: str, resolution: float | None, lig: str, lig_heavy: int, lig_subject: bool
) -> str:
    bits = []
    if resolution is not None:
        bits.append(f"{resolution:.2f} A")
    if method:
        bits.append(method.title().replace("X-Ray Diffraction", "X-ray"))
    if lig:
        tag = "studied ligand" if lig_subject else "ligand"
        bits.append(f"{tag} {lig} (~{lig_heavy} heavy atoms)")
    else:
        bits.append("no drug-like ligand -- docking box would be a guess")
    return ", ".join(bits)


def shortlist(
    client: BankClient,
    accession: str,
    *,
    fallback_ids: tuple[str, ...] = (),
    max_examined: int = 12,
    keep: int = 6,
) -> list[PdbCandidate]:
    """Ranked, dockable candidates for a target.

    ``fallback_ids`` are the PDB cross-references UniProt already returned; they are used
    when the RCSB search service is unavailable, which keeps a run going on a degraded but
    still correct path rather than losing docking entirely.
    """
    ids = search_by_uniprot(client, accession, limit=max(max_examined, 24)) or list(
        fallback_ids[:max_examined]
    )
    cands: list[PdbCandidate] = []
    with_ligand = 0
    for pid in ids:
        c = describe_entry(client, pid)
        if c is None:
            continue
        cands.append(c)
        if c.ligand_code:
            with_ligand += 1
        # Stop once there are enough *dockable* candidates, not merely enough candidates.
        # A target can have a run of very high-resolution apo structures at the top of the
        # search -- ABL1 does -- and stopping at a fixed count returns a shortlist whose
        # own reason field says the box would be a guess.
        if with_ligand >= keep and len(cands) >= max_examined:
            break
        if len(cands) >= max(max_examined, 24):
            break

    # A structure with no bound ligand cannot define a docking box, so it ranks below every
    # one that can regardless of resolution.
    cands.sort(key=lambda c: (1 if c.ligand_code else 0, c.score), reverse=True)
    return cands[:keep]


def fetch_cif(client: BankClient, pdb_id: str) -> str | None:
    return client.get_text(CIF_FILE.format(pdb_id.strip().upper()))


def fetch_pdb(client: BankClient, pdb_id: str) -> str | None:
    return client.get_text(PDB_FILE.format(pdb_id.strip().upper()))


POLYMER_ENTITY = "https://data.rcsb.org/rest/v1/core/polymer_entity/{}/{}"


def uniprot_for_pdb(client: BankClient, pdb_id: str) -> str | None:
    """The UniProt accession of a structure's largest protein chain.

    Lets a user paste a PDB ID they already have -- from a paper, a collaborator, a previous
    run -- instead of looking up which protein it is. Structures often contain several
    entities (a kinase plus a cyclin, a protein plus a nanobody), so the first polymer entity
    carrying a UniProt reference wins; RCSB orders them by size, so that is the subject of
    the structure rather than a crystallisation partner.
    """
    pid = pdb_id.strip().upper()
    entry = entry_info(client, pid)
    if not entry:
        return None
    ids = (entry.get("rcsb_entry_container_identifiers") or {}).get("polymer_entity_ids") or []
    for ent_id in ids:
        data = client.get(POLYMER_ENTITY.format(pid, ent_id))
        if not data:
            continue
        ident = data.get("rcsb_polymer_entity_container_identifiers") or {}
        accs = ident.get("uniprot_ids") or []
        if accs:
            return str(accs[0]).upper()
    return None
