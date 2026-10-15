"""UniProt: the canonical identity of a protein target.

UniProt is the hub the other three banks hang off. The accession it returns is what resolves
a ChEMBL target and what filters an RCSB structure search, so this is almost always the first
network call a run makes -- and the reason the policy's opening move is so often "retrieval".

Search results are ranked rather than taken in API order: a user typing "JAK2" means the
reviewed human entry, not an unreviewed fragment from a fungus.
"""

from __future__ import annotations

from dataclasses import dataclass

from .http import BankClient

SEARCH = "https://rest.uniprot.org/uniprotkb/search"
ENTRY = "https://rest.uniprot.org/uniprotkb/{}.json"

SEARCH_FIELDS = "accession,id,protein_name,gene_names,organism_name,reviewed,length,xref_pdb"

# Organisms a drug-discovery query almost always means, in order of preference.
PREFERRED_ORGANISMS = ("Homo sapiens", "Mus musculus", "Rattus norvegicus")


@dataclass(frozen=True, slots=True)
class UniProtHit:
    accession: str
    name: str
    gene: str
    organism: str
    reviewed: bool
    length: int
    pdb_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "source": "uniprot",
            "key": self.accession,
            "uniprot": self.accession,
            "name": self.name,
            "gene": self.gene,
            "organism": self.organism,
            "reviewed": self.reviewed,
            "length": self.length,
            "pdb_ids": list(self.pdb_ids),
        }


def _protein_name(entry: dict) -> str:
    desc = entry.get("proteinDescription") or {}
    for path in ("recommendedName", "submissionNames"):
        node = desc.get(path)
        if isinstance(node, list):
            node = node[0] if node else None
        if node:
            full = (node.get("fullName") or {}).get("value")
            if full:
                return full
    return entry.get("uniProtkbId") or entry.get("primaryAccession", "")


def _gene(entry: dict) -> str:
    genes = entry.get("genes") or []
    if genes:
        return (genes[0].get("geneName") or {}).get("value", "")
    return ""


def _pdb_ids(entry: dict) -> tuple[str, ...]:
    return tuple(
        x["id"]
        for x in (entry.get("uniProtKBCrossReferences") or [])
        if x.get("database") == "PDB" and x.get("id")
    )


def _parse_hit(entry: dict) -> UniProtHit:
    return UniProtHit(
        accession=entry.get("primaryAccession", ""),
        name=_protein_name(entry),
        gene=_gene(entry),
        organism=(entry.get("organism") or {}).get("scientificName", ""),
        reviewed=entry.get("entryType", "").startswith("UniProtKB reviewed"),
        length=(entry.get("sequence") or {}).get("length", 0),
        pdb_ids=_pdb_ids(entry),
    )


def _rank(hit: UniProtHit, query: str) -> tuple:
    """Sort key -- lower is better. Reviewed human entries with structures come first."""
    q = query.strip().lower()
    exact_gene = hit.gene.lower() == q
    exact_acc = hit.accession.lower() == q
    try:
        organism_rank = PREFERRED_ORGANISMS.index(hit.organism)
    except ValueError:
        organism_rank = len(PREFERRED_ORGANISMS)
    return (
        0 if (exact_acc or exact_gene) else 1,
        0 if hit.reviewed else 1,
        organism_rank,
        0 if hit.pdb_ids else 1,
        -hit.length,
    )


def search(client: BankClient, query: str, limit: int = 10) -> list[UniProtHit]:
    """Free-text search, ranked for what a drug-discovery user probably meant."""
    q = query.strip()
    if not q:
        return []
    data = client.get(
        SEARCH,
        params={"query": q, "fields": SEARCH_FIELDS, "size": max(limit, 10), "format": "json"},
    )
    hits = [_parse_hit(e) for e in (data or {}).get("results", [])]
    hits = [h for h in hits if h.accession]
    hits.sort(key=lambda h: _rank(h, q))
    return hits[:limit]


def entry(client: BankClient, accession: str) -> dict | None:
    """The full entry, including the sequence."""
    return client.get(ENTRY.format(accession.strip().upper()))


def sequence(client: BankClient, accession: str) -> str:
    data = entry(client, accession)
    if not data:
        return ""
    return (data.get("sequence") or {}).get("value", "")


def resolve(client: BankClient, query: str) -> UniProtHit | None:
    """Best single interpretation of what the user typed, or ``None``."""
    hits = search(client, query, limit=5)
    return hits[0] if hits else None
