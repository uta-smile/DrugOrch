"""One molecule found by searching a bank *with another molecule*.

Retrieval's other query is a protein: "what is known to bind JAK2". This is the other
direction -- "what real, catalogued molecule looks like this one" -- and it is what lets a
generated structure be checked against chemistry that actually exists instead of being
judged only by a score.

The two banks answer that question with different amounts of context, and the difference
matters enough to keep in the record rather than flatten:

* **ChEMBL** knows how far a molecule got. ``max_phase``, ``withdrawn``, ``black_box`` and
  ``first_approval`` are regulatory history, not predictions -- a neighbour that reached
  phase 4 and one that was withdrawn are very different news about the same region of
  chemical space.
* **PubChem** knows far more molecules and almost nothing about their fate. It is breadth.

``bank_similarity`` is whatever the bank itself reported, on a 0-1 scale, and is ``None``
for a bank that reports nothing. It is deliberately *not* the number anything is filtered or
ranked on: ChEMBL's similarity and a locally computed one are different recipes, and mixing
them would make one column of numbers that means two things. The subsystem recomputes
Tanimoto for every hit with this project's own fingerprint and keeps the bank's figure
beside it as a cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Neighbour:
    """A catalogued molecule structurally near a query molecule."""

    smiles: str
    source: str
    bank_id: str
    name: str = ""
    bank_similarity: float | None = None
    max_phase: float | None = None
    withdrawn: bool = False
    black_box: bool = False
    oral: bool | None = None
    first_approval: int | None = None

    @property
    def clinical(self) -> bool:
        """Did this molecule reach a clinical trial? ``max_phase`` of 1 or more."""
        return self.max_phase is not None and self.max_phase >= 1.0

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "source": self.source,
            "bank_id": self.bank_id,
            "name": self.name,
            "bank_similarity": self.bank_similarity,
            "max_phase": self.max_phase,
            "withdrawn": self.withdrawn,
            "black_box": self.black_box,
            "oral": self.oral,
            "first_approval": self.first_approval,
        }
