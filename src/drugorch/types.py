"""Shared value types that cross module boundaries.

Kept deliberately small and dependency-free so that :mod:`drugorch.policy` and
:mod:`drugorch.subsystems` can both import them without importing each other.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class PdbCandidate:
    """One structure the retrieval subsystem considered for docking.

    The shortlist is kept, not just the winner, because receptor preparation fails often
    enough that "try the next structure" is a real routing path -- and because which
    structure was chosen, and why, belongs in the run's provenance.
    """

    pdb_id: str
    method: str = ""
    resolution: float | None = None
    title: str = ""
    ligand_code: str = ""
    ligand_heavy_atoms: int = 0
    chain: str = ""
    score: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """The protein a run is working on, as resolved against the data banks.

    A run is created with whatever the user typed; retrieval fills the rest in. Fields stay
    optional because a run must be able to start before the banks have been queried -- the
    orchestrator's first routing decision is usually "go find out".
    """

    query: str
    uniprot: str | None = None
    name: str = ""
    gene: str = ""
    organism: str = ""
    sequence: str = ""
    chembl_id: str | None = None
    pdb_id: str | None = None
    pdb_ids: tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        """Whether we know enough to do structure-free work on this target."""
        return self.uniprot is not None

    @property
    def label(self) -> str:
        if self.gene and self.uniprot:
            return f"{self.gene} · {self.uniprot}"
        return self.uniprot or self.query

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pdb_ids"] = list(self.pdb_ids)
        d["resolved"] = self.resolved
        d["label"] = self.label
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TargetSpec:
        known = {f for f in cls.__slots__}
        payload = {k: v for k, v in d.items() if k in known}
        if "pdb_ids" in payload and payload["pdb_ids"] is not None:
            payload["pdb_ids"] = tuple(payload["pdb_ids"])
        return cls(**payload)


@dataclass(slots=True)
class MoleculeRecord:
    """Everything the run knows about one molecule, accumulated across subsystems.

    ``origin`` and ``parents`` are what let the trace draw a dependency edge from the
    generation block that produced a molecule to the docking block that scored it.
    """

    smiles: str
    origin: str = "unknown"
    parents: tuple[str, ...] = ()
    props: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    step_first_seen: int = 0

    def merge(self, *, props: dict | None = None, scores: dict | None = None) -> None:
        if props:
            self.props.update(props)
        if scores:
            self.scores.update(scores)

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "origin": self.origin,
            "parents": list(self.parents),
            "props": self.props,
            "scores": self.scores,
            "step_first_seen": self.step_first_seen,
        }


PANEL_MARKER = "panel_version"
"""Key marking a molecule as carrying a computed property panel.

Provenance fields (a ChEMBL id, an origin tag) also live in ``props``, so "has any props" is
not the same question as "has been profiled" -- conflating them silently prevents the
properties subsystem from ever running on retrieved molecules. The value is the panel's
version string, so bumping :data:`drugorch.versions.PANEL_V` re-profiles rather than leaving
two recipes mixed in one column.
"""
