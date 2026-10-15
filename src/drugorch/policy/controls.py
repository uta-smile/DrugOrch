"""Cheap controls: random and round-robin routing.

These exist to separate two things that are easy to confuse. If the LLM arm beats the fixed
pipeline, it could be because its decisions are *good*, or merely because *varying* the
subsystem order helps at all. Random answers the second question; round-robin answers the
narrower "does varying it unpredictably matter, or would any rotation do?".

Both still respect availability and affordability. A control that proposes impossible actions
would lose for the wrong reason and make the comparison look better than it is.

Both draw from ``numpy.random.default_rng(seed)`` seeded per run, so an arm is reproducible
and paired with the other arms on the same seed.
"""

from __future__ import annotations

import numpy as np

from ..orchestrator.state import Decision, RunState
from ..subsystems.base import Capability
from .base import affordable, alternatives_from_scores, assess


def _viable(state: RunState, manifest: list[Capability]) -> list[str]:
    f = assess(state, manifest)
    return [c.key for c in manifest if c.key in f["available"] and affordable(state, manifest, c.key)]


class RandomPolicy:
    """Picks uniformly at random among subsystems that could run."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, state: RunState) -> None:
        # Re-seed from the run's own seed so the arm is paired with the others, not merely
        # reproducible in isolation.
        self.seed = state.seed
        self._rng = np.random.default_rng(state.seed)

    def decide(self, state: RunState, manifest: list[Capability]) -> Decision:
        options = _viable(state, manifest)
        if not options:
            return Decision.stop(
                "no subsystem is available and affordable", policy=self.name, source="rule"
            )
        chosen = str(self._rng.choice(options))
        scores = dict.fromkeys(options, 1.0 / len(options))
        reasons = dict.fromkeys(options, "uniform random control -- no state was read")
        return Decision(
            action="call",
            subsystem=chosen,
            params={},
            rationale=(
                f"Random control: chose {chosen} uniformly from {len(options)} available "
                "subsystems. This policy reads no run state."
            ),
            considered=alternatives_from_scores(scores, reasons, chosen),
            expected_gain="n/a -- random",
            policy=self.name,
            source="rule",
        )

    def describe(self) -> dict:
        return {"name": self.name, "seed": self.seed}


class RoundRobinPolicy:
    """Cycles through the subsystems in a fixed rotation, skipping unusable ones."""

    name = "roundrobin"

    def __init__(self) -> None:
        self._i = 0

    def reset(self, state: RunState) -> None:
        self._i = 0

    def decide(self, state: RunState, manifest: list[Capability]) -> Decision:
        order = [c.key for c in manifest]
        options = set(_viable(state, manifest))
        if not options:
            return Decision.stop(
                "no subsystem is available and affordable", policy=self.name, source="rule"
            )
        for _ in range(len(order)):
            key = order[self._i % len(order)]
            self._i += 1
            if key in options:
                scores = {k: (1.0 if k == key else 0.0) for k in order}
                reasons = dict.fromkeys(
                    order, "round-robin control -- position in the rotation, not merit"
                )
                return Decision(
                    action="call",
                    subsystem=key,
                    params={},
                    rationale=(
                        f"Round-robin control: {key} is next in the rotation. "
                        "This policy reads no run state."
                    ),
                    considered=alternatives_from_scores(scores, reasons, key),
                    expected_gain="n/a -- rotation",
                    policy=self.name,
                    source="rule",
                )
        return Decision.stop("rotation exhausted", policy=self.name, source="rule")

    def describe(self) -> dict:
        return {"name": self.name, "position": self._i}
