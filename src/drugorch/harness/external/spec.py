"""The constraint set an arm believes it was asked for.

A MolDesignBench question states some constraints outright ("keep QED in the 0.36-0.56
range") and implies others through the scenario ("must cross a tightly packed lipid matrix",
which the answer key encodes as windows on MW, logP and TPSA). Half the benchmark's
difficulty is reading the second kind, so the first thing an arm does is write down what it
thinks the constraints are.

That written-down set is a :class:`DesignSpec`. It is the arm's own belief and nothing more:
it is never compared against the answer key, never corrected, and the answer key is not in
the process that produces it. What it is used for is fitness. A generator needs a direction
to move in, and "this molecule is 0.4 normalised units outside the TPSA window" is a
direction where "wrong" is not.

Distance is normalised the way the benchmark normalises it, by the p5-p95 range of each
property over ten million molecules. Optimising raw distance instead would put every unit of
effort into molecular weight, whose numbers are two orders of magnitude larger than QED's,
and none into the constraint that was actually missed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .panel import FRAGMENT_RANGES, PERCENTILE_RANGES, Measurement

# The vocabulary the benchmark's own measurement tool answers in. An arm that invents a
# property name gets it dropped here rather than silently scored as satisfied.
MEASURED_PROPERTIES = (
    "MW", "logP", "logD", "logS", "HBD", "HBA", "TPSA", "rotB", "rings_total",
    "rings_aromatic", "QED", "BBBP", "Mutag", "MR", "heavy_atoms", "formal_charge",
    "Fsp3", "hetero", "topo_diameter", "HIA",
)


@dataclass(frozen=True, slots=True)
class PropTarget:
    """One property window. Either bound may be absent, and both are inclusive."""

    name: str
    lo: float | None = None
    hi: float | None = None

    def distance(self, value: float) -> float:
        """Normalised distance outside the window; 0.0 inside it."""
        denom = PERCENTILE_RANGES.get(self.name) or 1.0
        if self.lo is not None and value < self.lo:
            return (self.lo - value) / denom
        if self.hi is not None and value > self.hi:
            return (value - self.hi) / denom
        return 0.0

    def describe(self) -> str:
        if self.lo is not None and self.hi is not None:
            return f"{self.name} in [{self.lo:g}, {self.hi:g}]"
        if self.lo is not None:
            return f"{self.name} >= {self.lo:g}"
        if self.hi is not None:
            return f"{self.name} <= {self.hi:g}"
        return self.name


@dataclass(frozen=True, slots=True)
class FragTarget:
    """One functional-group requirement.

    ``at_least`` follows the benchmark's own split: an optimization instance asks for a
    group to survive the edit, so more of it is fine, while a generation instance asks for
    an exact count.
    """

    name: str
    count: int = 1
    at_least: bool = True

    def distance(self, found: int) -> float:
        denom = max(1.0, FRAGMENT_RANGES.get(self.name, 1.0))
        if self.at_least:
            return max(0, self.count - found) / denom
        return abs(found - self.count) / denom

    def satisfied(self, found: int) -> bool:
        return found >= self.count if self.at_least else found == self.count

    def describe(self) -> str:
        word = "at least" if self.at_least else "exactly"
        return f"{word} {self.count} {self.name}"


@dataclass(slots=True)
class Score:
    """How one molecule stands against one spec."""

    satisfied: int = 0
    total: int = 0
    distance: float = 0.0
    misses: list[str] = field(default_factory=list)
    valid: bool = True

    @property
    def csr(self) -> float:
        return self.satisfied / self.total if self.total else 0.0

    @property
    def solved(self) -> bool:
        return self.valid and self.total > 0 and self.satisfied == self.total

    @property
    def fitness(self) -> float:
        """Higher is better, and bounded, so a wildly wrong molecule cannot dominate.

        Satisfaction count leads and distance breaks ties within it. A fitness that was only
        distance would happily trade a satisfied constraint for a small improvement on an
        unsatisfied one, which is the opposite of what the benchmark scores: success is all
        constraints at once, not a low average miss.
        """
        if not self.valid:
            return -1e6
        return self.satisfied - min(self.distance, 10.0) / 100.0


@dataclass(slots=True)
class DesignSpec:
    """What an arm believes the question asked for."""

    properties: list[PropTarget] = field(default_factory=list)
    fragments: list[FragTarget] = field(default_factory=list)
    infeasible: bool = False
    reason: str = ""

    @classmethod
    def from_llm(cls, payload: dict, *, at_least: bool = True) -> DesignSpec:
        """Build a spec from the planner's JSON, dropping what cannot be measured.

        A property the measurement tool does not answer in is dropped rather than kept as a
        constraint nothing can check. Keeping it would make every molecule fail it forever,
        and the generator would spend its whole budget chasing a name.
        """
        props = []
        for item in payload.get("properties") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("property") or item.get("name") or "")
            if name not in MEASURED_PROPERTIES:
                continue
            lo, hi = item.get("min"), item.get("max")
            lo = float(lo) if isinstance(lo, (int, float)) else None
            hi = float(hi) if isinstance(hi, (int, float)) else None
            if lo is None and hi is None:
                # A target with neither bound is not a constraint. The schema requires both
                # keys and allows both to be null, so the model can answer
                # {"property": "MW", "min": null, "max": null} and mean "molecular weight
                # matters here" -- which `distance` then satisfies for every molecule. Kept,
                # it would count toward `satisfied` *and* `total`, so a spec of two such
                # targets reports 2/2 and `solved`, and the arm answers the first molecule
                # it measured. Measured on a 200-instance run: 359 of these across 98 of
                # 202 attempts, inflating both the spec-recall figure and the confidence
                # precision on the public board.
                continue
            props.append(PropTarget(name=name, lo=lo, hi=hi))

        frags = []
        for item in payload.get("fragments") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            if not name:
                continue
            try:
                count = max(1, int(item.get("count", 1)))
            except (TypeError, ValueError):
                count = 1
            frags.append(FragTarget(name=name, count=count, at_least=at_least))

        return cls(
            properties=props,
            fragments=frags,
            infeasible=bool(payload.get("infeasible", False)),
            reason=str(payload.get("reason") or "")[:500],
        )

    @property
    def total(self) -> int:
        return len(self.properties) + len(self.fragments)

    def contradictions(self) -> list[str]:
        """Windows that cannot all hold at once.

        Only structural impossibility is reported: a window whose minimum exceeds its
        maximum, or two windows on one property that do not overlap. A merely difficult
        constraint set is not reported here, because an arm that talks itself out of hard
        instances answers None everywhere and scores well on a tenth of the benchmark while
        failing the other nine.
        """
        out = []
        by_name: dict[str, list[PropTarget]] = {}
        for p in self.properties:
            if p.lo is not None and p.hi is not None and p.lo > p.hi:
                out.append(f"{p.name}: minimum {p.lo:g} exceeds maximum {p.hi:g}")
            by_name.setdefault(p.name, []).append(p)
        for name, group in by_name.items():
            if len(group) < 2:
                continue
            lo = max((g.lo for g in group if g.lo is not None), default=None)
            hi = min((g.hi for g in group if g.hi is not None), default=None)
            if lo is not None and hi is not None and lo > hi:
                out.append(f"{name}: the windows leave an empty range [{lo:g}, {hi:g}]")
        return out

    def score(self, m: Measurement | None) -> Score:
        """Measure one molecule against this spec."""
        if m is None or not m.valid:
            return Score(total=self.total, valid=False, misses=["not a valid molecule"])

        s = Score(total=self.total)
        for p in self.properties:
            value = m.properties.get(p.name)
            if value is None:
                # Unmeasurable counts as missed, exactly as the evaluator treats it. It is
                # not skipped: the evaluator charges a full unit of distance for a property
                # it could not read, so a fitness that ignored it would steer toward
                # molecules the scorer will punish.
                s.distance += 1.0
                s.misses.append(f"{p.name}: not measured")
                continue
            d = p.distance(float(value))
            s.distance += d
            if d == 0.0:
                s.satisfied += 1
            else:
                s.misses.append(f"{p.describe()}, measured {float(value):g}")

        for f in self.fragments:
            found = int(m.fragments.get(f.name, 0) or 0)
            if f.satisfied(found):
                s.satisfied += 1
            else:
                s.distance += f.distance(found)
                s.misses.append(f"{f.describe()}, found {found}")
        return s

    def describe(self) -> str:
        """The spec as a line an arm can read back in its next prompt."""
        parts = [p.describe() for p in self.properties] + [
            f.describe() for f in self.fragments
        ]
        return "; ".join(parts) if parts else "no measurable constraints extracted"

    def to_dict(self) -> dict:
        return {
            "properties": [
                {"property": p.name, "min": p.lo, "max": p.hi} for p in self.properties
            ],
            "fragments": [
                {"name": f.name, "count": f.count, "at_least": f.at_least}
                for f in self.fragments
            ],
            "infeasible": self.infeasible,
            "reason": self.reason,
        }


def merge(base: DesignSpec, addition: DesignSpec) -> DesignSpec:
    """``base`` widened by everything in ``addition``, never narrowed by omission.

    Used wherever a second reading of a question produces constraints: the verification
    pass, and the model's own mid-run revision. Both must add rather than replace. A
    revision that replaced would let a model which answers with the one constraint it just
    noticed silently discard the six it already had, and the arm would then oscillate
    between two incomplete readings of the same text.

    A property named twice is intersected rather than appended, because two windows on one
    property stored side by side are satisfied independently and together mean nothing.
    """
    props = {p.name: p for p in base.properties}
    for p in addition.properties:
        old = props.get(p.name)
        if old is None:
            props[p.name] = p
            continue
        lo = max([x for x in (old.lo, p.lo) if x is not None], default=None)
        hi = min([x for x in (old.hi, p.hi) if x is not None], default=None)
        props[p.name] = PropTarget(name=p.name, lo=lo, hi=hi)

    frags = {f.name: f for f in base.fragments}
    for f in addition.fragments:
        frags.setdefault(f.name, f)

    return DesignSpec(
        properties=list(props.values()),
        fragments=list(frags.values()),
        infeasible=base.infeasible,
        reason=base.reason,
    )


# The shape the planner must answer in. Property names are enumerated rather than left free
# so a hallucinated name is a schema violation at the API rather than a silently dropped
# constraint here.
#
# Every object lists all of its keys in `required` and forbids extras. That is not style: the
# ChatGPT-plan route rejects a schema whose `required` omits any key, with a 400 that the run
# loop turns into a fallback, so a schema written the laxer way runs on one provider and
# produces an empty board on another. Optional values are expressed as a nullable type
# instead, which both routes accept.
SPEC_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "infeasible": {
            "type": "boolean",
            "description": "True only when the constraints contradict each other.",
        },
        "reason": {"type": "string"},
        "properties": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "property": {"type": "string", "enum": list(MEASURED_PROPERTIES)},
                    "min": {"type": ["number", "null"]},
                    "max": {"type": ["number", "null"]},
                },
                "required": ["property", "min", "max"],
                "additionalProperties": False,
            },
        },
        "fragments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["name", "count"],
                "additionalProperties": False,
            },
        },
        "candidates": {
            "type": "array",
            "description": "Opening SMILES proposals, best first.",
            "items": {"type": "string"},
        },
    },
    "required": ["infeasible", "reason", "properties", "fragments", "candidates"],
    "additionalProperties": False,
}


__all__ = [
    "MEASURED_PROPERTIES",
    "merge",
    "SPEC_SCHEMA",
    "DesignSpec",
    "FragTarget",
    "PropTarget",
    "Score",
]
