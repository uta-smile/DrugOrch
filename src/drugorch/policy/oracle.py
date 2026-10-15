"""The ceiling arm: routing with access to the answer key.

This is not a method and must never be reported as one. It reads the measured ChEMBL
potencies of the molecules in the run -- information no honest policy has -- and uses them to
route. Its only job is to bound the comparison: if the LLM arm lands near this, the routing
problem was easy and the headroom is small; if every arm clusters far below it, routing is
where the remaining value is.

Concretely, it exploits one thing a real policy cannot know: which unscored molecules are
actually potent. It therefore docks the molecules that will turn out best, spends nothing on
generating from weak parents, and stops as soon as the genuinely good molecules have been
evaluated.
"""

from __future__ import annotations

from ..orchestrator.state import Decision, RunState
from ..subsystems.base import Capability
from .base import affordable, alternatives_from_scores, assess, score_unknown

ANSWER_KEY = "pchembl"
"""The measured potency retrieval attaches to known actives. The thing being peeked at."""


class OraclePolicy:
    """Routes using measured potency that an honest policy would not have."""

    name = "oracle"

    def __init__(self, top_fraction: float = 0.25):
        self.top_fraction = top_fraction

    def reset(self, state: RunState) -> None:
        return None

    def _known_good_unscored(self, state: RunState) -> list[str]:
        """Unscored molecules that the answer key says are worth the docking budget."""
        pool = [
            (s, rec.scores[ANSWER_KEY])
            for s, rec in state.molecules.items()
            if rec.scores.get(ANSWER_KEY) is not None
            and rec.scores.get(state.objective_key) is None
        ]
        if not pool:
            return []
        pool.sort(key=lambda kv: kv[1], reverse=True)
        keep = max(1, int(len(pool) * self.top_fraction))
        return [s for s, _ in pool[:keep]]

    def decide(self, state: RunState, manifest: list[Capability]) -> Decision:
        f = assess(state, manifest)

        def usable(k: str) -> bool:
            return k in f["available"] and affordable(state, manifest, k)

        scores: dict[str, float] = {}
        reasons: dict[str, str] = {}
        params: dict[str, dict] = {}

        if usable("retrieval") and not f["has_identity"]:
            scores["retrieval"] = 1.0
            reasons["retrieval"] = "target unresolved"
            params["retrieval"] = {"fetch": "identity"}
        elif usable("retrieval") and not f["has_actives"]:
            scores["retrieval"] = 0.95
            reasons["retrieval"] = "the answer key itself has not been fetched yet"
            params["retrieval"] = {"fetch": "actives"}
        elif usable("retrieval") and not f["has_structure"] and "docking" in f["available"]:
            scores["retrieval"] = 0.9
            reasons["retrieval"] = "a structure is needed before the known-good can be docked"
            params["retrieval"] = {"fetch": "structures"}

        good = self._known_good_unscored(state)
        if usable("docking") and good and f["has_structure"]:
            scores["docking"] = 0.99
            reasons["docking"] = (
                f"{len(good)} unscored molecules are known from the answer key to be the "
                "most potent available; dock exactly those"
            )
            params["docking"] = {"smiles": good[:8], "max_molecules": 8}

        if usable("properties") and f["n_unprofiled"] > 0:
            # Cheap, and the oracle still wants QED for reporting; never the priority.
            scores["properties"] = 0.2
            reasons["properties"] = f"{f['n_unprofiled']} molecules unprofiled; nearly free"
            params["properties"] = {}

        if usable("generation"):
            scores["generation"] = 0.01
            reasons["generation"] = (
                "the answer key already contains better molecules than generation would "
                "produce; spending on it would be strictly worse"
            )
            params["generation"] = {}

        # Anything these rules know nothing about -- a plugin -- is still scored and
        # reported. The oracle is a ceiling, not a method, but it must not be the one arm
        # where a subsystem silently disappears from the trace.
        score_unknown(state, manifest, f, scores, reasons, params)

        viable = {k: v for k, v in scores.items() if v > 0.05}
        if not viable:
            return Decision(
                action="stop",
                rationale=(
                    "every molecule the answer key rates highly has been evaluated; "
                    f"{f['remaining']:.1f} units left unspent"
                ),
                considered=alternatives_from_scores(scores, reasons, None),
                policy=self.name,
                source="rule",
            )

        chosen = max(viable, key=lambda k: viable[k])
        return Decision(
            action="call",
            subsystem=chosen,
            params=params.get(chosen, {}),
            rationale=f"[CEILING ARM -- reads measured potency] {reasons.get(chosen, '')}",
            considered=alternatives_from_scores(scores, reasons, chosen),
            expected_gain=f"oracle score {viable[chosen]:.2f}",
            policy=self.name,
            source="rule",
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "top_fraction": self.top_fraction,
            "answer_key": ANSWER_KEY,
            "warning": "ceiling arm: reads measured potency; not a deployable method",
        }
