"""PubChem: turning a drug name into a structure.

The narrowest of the four clients, and deliberately so. ChEMBL supplies the actives a run
works from; PubChem is here for the case where a goal names a specific compound -- "something
like imatinib but more selective" -- and we need the structure behind the name.

PubChem documents a hard courtesy limit of 5 requests per second and 400 per minute, which is
the strictest of the four banks; :mod:`drugorch.clients.http` enforces it per host.
"""

from __future__ import annotations

import urllib.parse

from ..chem import canonical
from .http import BankClient
from .neighbour import Neighbour

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# PubChem has renamed its SMILES properties more than once -- a request for
# `IsomericSMILES` currently comes back keyed `SMILES` -- so the response is read by trying
# the names in preference order rather than by trusting the one that was asked for.
_SMILES_KEYS = ("IsomericSMILES", "SMILES", "ConnectivitySMILES", "CanonicalSMILES")


def smiles_for_name(client: BankClient, name: str) -> str | None:
    """Canonical SMILES for a compound name, or ``None`` if PubChem does not know it."""
    q = name.strip()
    if not q:
        return None
    data = client.get(
        f"{PUG}/compound/name/{q}/property/CanonicalSMILES,IsomericSMILES/JSON"
    )
    props = ((data or {}).get("PropertyTable") or {}).get("Properties") or []
    if not props:
        return None
    row = props[0]
    # Isomeric SMILES carries stereochemistry, which matters for binding; prefer it.
    raw = row.get("IsomericSMILES") or row.get("CanonicalSMILES") or ""
    return canonical(raw) if raw else None


def synonyms(client: BankClient, name: str, limit: int = 8) -> list[str]:
    data = client.get(f"{PUG}/compound/name/{name.strip()}/synonyms/JSON")
    info = ((data or {}).get("InformationList") or {}).get("Information") or []
    if not info:
        return []
    return list(info[0].get("Synonym", []))[:limit]


def similar(
    client: BankClient,
    smiles: str,
    *,
    threshold: float = 0.7,
    limit: int = 25,
) -> list[Neighbour]:
    """Catalogued molecules near ``smiles``, by PubChem's own 2D similarity search.

    This is the breadth option. PubChem indexes an order of magnitude more structures than
    ChEMBL and knows essentially nothing about what became of them -- no clinical phase, no
    withdrawal, no approval year -- so a hit here says "this exists and someone has
    deposited it", not "this is a drug". :class:`~drugorch.clients.neighbour.Neighbour`
    leaves those fields unset rather than defaulting them to something that would read as a
    clean record.

    PubChem also reports no similarity score, which is why nothing downstream relies on the
    bank's own number: the Tanimoto that gets filtered and ranked on is recomputed locally
    for every hit from both banks with one fingerprint recipe.
    """
    pct = max(40, min(100, int(round(threshold * 100))))
    quoted = urllib.parse.quote(smiles, safe="")
    props = "Title,MolecularWeight,SMILES,XLogP,TPSA"
    try:
        data = client.get(
            f"{PUG}/compound/fastsimilarity_2d/smiles/{quoted}/property/{props}/JSON",
            params={"Threshold": pct, "MaxRecords": int(limit)},
        )
    except Exception:  # noqa: BLE001 -- an unparseable query is one lost query, not a failure
        return []

    out: list[Neighbour] = []
    for row in ((data or {}).get("PropertyTable") or {}).get("Properties") or []:
        raw = next((row[k] for k in _SMILES_KEYS if row.get(k)), "")
        smi = canonical(raw) if raw else None
        if not smi:
            continue
        cid = row.get("CID")
        out.append(
            Neighbour(
                smiles=smi,
                source="pubchem",
                bank_id=f"CID{cid}" if cid else "",
                name=row.get("Title") or "",
            )
        )
    return out


def resolve_many(client: BankClient, names: list[str]) -> dict[str, str]:
    """``{name: smiles}`` for the names PubChem could resolve; misses are simply absent."""
    out: dict[str, str] = {}
    for n in names:
        try:
            smi = smiles_for_name(client, n)
        except Exception:
            continue
        if smi:
            out[n] = smi
    return out
