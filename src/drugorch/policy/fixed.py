"""The fixed pipeline: the null arm, and the thing the paper is arguing against.

This is what nearly every published LLM-agent drug-discovery system actually does -- resolve
the target, pull data, generate, score, repeat -- in a hard-coded order that does not depend
on what the run has learned. It is the baseline the adaptive arms must beat, so it has to be
a *fair* baseline: competently ordered, not a straw man.

The order encodes real domain sense:

1. identity before anything, since everything else needs the accession;
2. structures and actives next, because docking and generation respectively need them;
3. properties on whatever exists, because it is nearly free;
4. then a generate/score loop until the budget runs out.

What it never does is react. If generation produces nothing novel, it generates again. If
docking has become unavailable, it still asks for it and is refused. Those are precisely the
situations where an adaptive policy should pull ahead, and leaving them in is what makes the
comparison informative.
"""

from __future__ import annotations

from ..orchestrator.state import Alternative, Decision, RunState
from ..subsystems.base import Capability
from .base import affordable, assess

# (subsystem, params). Later stages repeat; the cycle point is where the loop restarts.
PIPELINE: list[tuple[str, dict]] = [
    ("retrieval", {"fetch": "identity"}),
    ("retrieval", {"fetch": "structures"}),
    ("retrieval", {"fetch": "actives"}),
    ("properties", {}),
    ("generation", {}),
    ("properties", {}),
    ("docking", {}),
]
CYCLE_FROM = 4
"""Index the pipeline loops back to once it has run through: generate, profile, dock."""


class FixedPolicy:
    """Walks :data:`PIPELINE` in order, looping over its tail until the budget is gone.

    Deterministic given a seed and a subsystem set. ``tests/test_fixed_policy.py`` pins that
    two runs at the same seed produce identical decision sequences -- if that ever breaks,
    every ablation built on this baseline is invalid.
    """

    name = "fixed"

    def __init__(self, pipeline: list[tuple[str, dict]] | None = None, cycle_from: int | None = None):
        self.pipeline = list(pipeline or PIPELINE)
        self.cycle_from = CYCLE_FROM if cycle_from is None else cycle_from
        self._i = 0

    def reset(self, state: RunState) -> None:
        self._i = 0

    def decide(self, state: RunState, manifest: list[Capability]) -> Decision:
        facts = assess(state, manifest)
        n = len(self.pipeline)

        # Walk forward from the current position, skipping stages whose subsystem is gone or
        # unaffordable, until we find one we can run or exhaust the cycle.
        for _ in range(n):
            if self._i >= n:
                self._i = self.cycle_from
            key, params = self.pipeline[self._i]
            self._i += 1

            if key not in facts["available"]:
                continue
            if not affordable(state, manifest, key):
                continue

            considered = tuple(
                Alternative(
                    subsystem=c.key,
                    score=0.0,
                    reason="not considered -- this policy follows a fixed order",
                )
                for c in manifest
                if c.key != key
            )
            stage = (self._i - 1) % n
            return Decision(
                action="call",
                subsystem=key,
                params=dict(params),
                rationale=(
                    f"Fixed pipeline, stage {stage + 1}/{n}: run {key}. "
                    "This policy does not read the run state; the order is set in advance."
                ),
                considered=considered,
                expected_gain="n/a -- fixed order",
                policy=self.name,
                source="rule",
            )

        return Decision.stop(
            "fixed pipeline: no remaining stage is available and affordable",
            policy=self.name,
            source="rule",
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "pipeline": [{"subsystem": k, "params": p} for k, p in self.pipeline],
            "cycle_from": self.cycle_from,
        }
