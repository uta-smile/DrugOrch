"""ChEMBL: measured bioactivity, and the only ground truth in the system.

The actives fetched here do double duty, and the two uses must not be confused:

* as **seeds** for the generation subsystem -- real chemistry that is known to bind, which is
  a far better starting population than random molecules;
* as the **answer key** for ``OraclePolicy``, the ceiling arm in the ablation ladder.

Only the second is cheating, and only ``OraclePolicy`` is allowed to do it. Everything else
sees these as ordinary retrieved molecules.

Potency is read from ``pchembl_value`` -- a negative log of the measured concentration, so
higher is more potent, and values from IC50, Ki and Kd assays are comparable on one scale.
Rows without one are dropped rather than converted, because a conversion would silently mix
assay types that are not interchangeable.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass

from ..chem import canonical
from .http import BankClient
from .neighbour import Neighbour

BASE = "https://www.ebi.ac.uk/chembl/api/data"
TARGET = f"{BASE}/target.json"
ACTIVITY = f"{BASE}/activity.json"
MOLECULE = f"{BASE}/molecule.json"
SIMILARITY = f"{BASE}/similarity"

# Assay types that measure direct binding or functional potency on one comparable scale.
POTENCY_TYPES = ("IC50", "Ki", "Kd", "EC50")


@dataclass(frozen=True, slots=True)
class ChemblTarget:
    chembl_id: str
    name: str
    organism: str
    target_type: str

    def to_dict(self) -> dict:
        return {
            "source": "chembl",
            "key": self.chembl_id,
            "chembl_id": self.chembl_id,
            "name": self.name,
            "organism": self.organism,
            "target_type": self.target_type,
        }


@dataclass(frozen=True, slots=True)
class Active:
    """One measured molecule-target activity."""

    smiles: str
    pchembl: float
    standard_type: str
    chembl_id: str = ""

    def to_record(self) -> dict:
        # `standard_type` was parsed and then dropped here, so assay provenance was lost
        # before the molecule entered the run. A pChEMBL from a Ki and one from an IC50 are
        # not the same measurement, and a benchmark that pools them should at least be able
        # to say so.
        return {
            "smiles": self.smiles,
            "props": {"chembl_id": self.chembl_id, "assay_type": self.standard_type},
            "scores": {"pchembl": self.pchembl},
        }


def target_for_uniprot(client: BankClient, accession: str) -> ChemblTarget | None:
    """The single-protein ChEMBL target for a UniProt accession.

    A protein can map to several ChEMBL targets (a complex, a cell line, the protein alone).
    We want ``SINGLE PROTEIN`` -- activity against a complex is not activity against this
    protein, and mixing them would put noise straight into the seed set.
    """
    data = client.get(
        TARGET,
        params={"target_components__accession": accession.strip().upper(), "limit": 20},
    )
    rows = (data or {}).get("targets", [])
    if not rows:
        return None
    rows.sort(key=lambda r: 0 if r.get("target_type") == "SINGLE PROTEIN" else 1)
    r = rows[0]
    return ChemblTarget(
        chembl_id=r.get("target_chembl_id", ""),
        name=r.get("pref_name", ""),
        organism=r.get("organism", ""),
        target_type=r.get("target_type", ""),
    )


def search_targets(client: BankClient, query: str, limit: int = 5) -> list[ChemblTarget]:
    """Free-text target search, for the autocomplete fan-out."""
    data = client.get(
        TARGET, params={"pref_name__icontains": query.strip(), "limit": limit}
    )
    out = []
    for r in (data or {}).get("targets", []):
        if r.get("target_chembl_id"):
            out.append(
                ChemblTarget(
                    chembl_id=r["target_chembl_id"],
                    name=r.get("pref_name", ""),
                    organism=r.get("organism", ""),
                    target_type=r.get("target_type", ""),
                )
            )
    return out


def actives(
    client: BankClient,
    target_chembl_id: str,
    *,
    limit: int = 300,
    min_pchembl: float = 5.0,
    page_size: int = 1000,
) -> list[Active]:
    """Potent, standardized molecules measured against this target, best first.

    ``min_pchembl=5.0`` is 10 micromolar -- the conventional line below which a compound is
    not considered a hit worth following up.
    """
    collected: dict[str, Active] = {}
    offset = 0
    # Each page is one cached request; stop as soon as we have enough or the bank runs out.
    while len(collected) < limit and offset < 10_000:
        data = client.get(
            ACTIVITY,
            params={
                "target_chembl_id": target_chembl_id,
                "standard_type__in": ",".join(POTENCY_TYPES),
                "pchembl_value__gte": min_pchembl,
                "limit": page_size,
                "offset": offset,
            },
        )
        rows = (data or {}).get("activities", [])
        if not rows:
            break
        for r in rows:
            raw = r.get("canonical_smiles")
            pch = r.get("pchembl_value")
            if not raw or pch is None:
                continue
            smi = canonical(raw)
            if smi is None:
                continue
            try:
                value = float(pch)
            except (TypeError, ValueError):
                continue
            prev = collected.get(smi)
            # Several assays may measure the same molecule; keep the strongest.
            if prev is None or value > prev.pchembl:
                collected[smi] = Active(
                    smiles=smi,
                    pchembl=value,
                    standard_type=r.get("standard_type", ""),
                    chembl_id=r.get("molecule_chembl_id", "") or "",
                )
        page_meta = (data or {}).get("page_meta", {})
        if not page_meta.get("next"):
            break
        offset += page_size

    out = sorted(collected.values(), key=lambda a: a.pchembl, reverse=True)
    return out[:limit]


def _flag(value: object) -> bool:
    """ChEMBL writes its booleans as 0/1 ints, and as ``None`` when it does not know."""
    return bool(value) if value is not None else False


def similar(
    client: BankClient,
    smiles: str,
    *,
    threshold: float = 0.7,
    limit: int = 25,
) -> list[Neighbour]:
    """Catalogued molecules near ``smiles``, with what ChEMBL knows about their fate.

    ``threshold`` is a Tanimoto fraction here and a percentage in the URL, which is the one
    place the two conventions meet; everything above this layer stays in 0-1 so a cutoff
    means the same thing whichever bank answered.

    ChEMBL will not accept a query below 40% similarity and returns 400 for an unparseable
    SMILES, so this raises nothing and returns an empty list on refusal -- a query molecule
    the bank dislikes is one query lost, not a failed call.

    The SMILES goes in the *path*, so every character that means something to a URL has to
    be escaped: ``/``, ``#`` and ``+`` all occur in ordinary SMILES and all change the
    request if left alone.
    """
    pct = max(40, min(100, int(round(threshold * 100))))
    quoted = urllib.parse.quote(smiles, safe="")
    try:
        data = client.get(f"{SIMILARITY}/{quoted}/{pct}.json", params={"limit": int(limit)})
    except Exception:  # noqa: BLE001 -- a 400 here means "bad query", not "broken bank"
        return []

    out: list[Neighbour] = []
    for row in (data or {}).get("molecules", []):
        raw = (row.get("molecule_structures") or {}).get("canonical_smiles") or ""
        smi = canonical(raw)
        if not smi:
            continue
        try:
            sim = float(row.get("similarity")) / 100.0
        except (TypeError, ValueError):
            sim = None
        try:
            phase = float(row["max_phase"]) if row.get("max_phase") is not None else None
        except (TypeError, ValueError):
            phase = None
        try:
            approved = int(row["first_approval"]) if row.get("first_approval") else None
        except (TypeError, ValueError):
            approved = None
        out.append(
            Neighbour(
                smiles=smi,
                source="chembl",
                bank_id=row.get("molecule_chembl_id", "") or "",
                name=row.get("pref_name") or "",
                bank_similarity=sim,
                max_phase=phase,
                withdrawn=_flag(row.get("withdrawn_flag")),
                black_box=_flag(row.get("black_box_warning")),
                oral=bool(row["oral"]) if row.get("oral") is not None else None,
                first_approval=approved,
            )
        )
    return out


def smiles_for_chembl_id(client: BankClient, molecule_chembl_id: str) -> str | None:
    data = client.get(MOLECULE, params={"molecule_chembl_id": molecule_chembl_id, "limit": 1})
    rows = (data or {}).get("molecules", [])
    if not rows:
        return None
    struct = rows[0].get("molecule_structures") or {}
    return canonical(struct.get("canonical_smiles", "") or "")


def accession_for_target(client: BankClient, target_chembl_id: str) -> str | None:
    """The UniProt accession behind a ChEMBL target id, so a pasted CHEMBL… resolves."""
    data = client.get(f"{BASE}/target/{target_chembl_id.strip().upper()}.json")
    for comp in (data or {}).get("target_components") or []:
        acc = comp.get("accession")
        if acc:
            return str(acc).upper()
    return None


def count_actives(client: BankClient, target_chembl_id: str, min_pchembl: float = 5.0) -> int:
    """How many potent measurements exist, without downloading them.

    ChEMBL reports the match count in ``page_meta.total_count``, so asking for a single row
    answers "is there a seed set here?" in one request. Downloading the actives to count
    them would be several hundred kilobytes for a number shown next to a search result.
    """
    data = client.get(
        ACTIVITY,
        params={
            "target_chembl_id": target_chembl_id.strip().upper(),
            "pchembl_value__gte": min_pchembl,
            "limit": 1,
        },
    )
    return int(((data or {}).get("page_meta") or {}).get("total_count") or 0)
