"""Hand-written routing rules over the same state the LLM reads. The control that matters.

Random and round-robin answer "does *any* switching help?". This one answers the question a
reviewer will actually ask: **does the LLM beat a competent afternoon's worth of hand-written
rules on identical inputs?** If it does not, the contribution is a wrapper around a state
machine, and the paper should say so.

So this arm is written to be good, not to lose. The rules are the ones a medicinal chemist
would give you if you asked how to spend a compute budget:

* get the target identity -- nothing works without it;
* pull actives before generating, because generating from nothing wastes the budget;
* profile new molecules first, since properties cost ~2% of a docking call and can tell you
  a molecule is not worth docking;
* dock when enough unscored molecules have accumulated to make a batch worthwhile;
* generate when the scored pool has been mined and the run needs new chemistry;
* stop early when the budget cannot buy another useful action, rather than burning it.

Every rule reports a score and a reason, so this arm's traces are as auditable as the LLM's.
"""

from __future__ import annotations

from ..orchestrator.state import Decision, RunState
from ..subsystems.base import Capability
from .base import affordable, alternatives_from_scores, assess, score_unknown

# Enough unscored molecules to make a docking batch worth its fixed overhead.
DOCK_BATCH_FLOOR = 4
# Below this fraction of budget left, stop opening new expensive work.
ENDGAME_FRACTION = 0.12
# A run that keeps paying for actions it rates as near-worthless is not being frugal, it is
# being stubborn. Below this score, stopping beats spending.
MIN_USEFUL_SCORE = 0.1


class ScriptedPolicy:
    """Deterministic rules over ``RunState.summary()``."""

    name = "scripted"

    def __init__(
        self,
        dock_batch_floor: int = DOCK_BATCH_FLOOR,
        endgame_fraction: float = ENDGAME_FRACTION,
        stall_limit: int = 4,
        min_useful_score: float = MIN_USEFUL_SCORE,
    ):
        self.dock_batch_floor = dock_batch_floor
        self.endgame_fraction = endgame_fraction
        self.stall_limit = stall_limit
        self.min_useful_score = min_useful_score

    def reset(self, state: RunState) -> None:
        return None

    def _score_options(self, state: RunState, manifest: list[Capability]) -> tuple[dict, dict, dict]:
        """``(scores, reasons, params)`` for every subsystem, higher score = more urgent."""
        f = assess(state, manifest)
        scores: dict[str, float] = {}
        reasons: dict[str, str] = {}
        params: dict[str, dict] = {}

        def avail(k: str) -> bool:
            return k in f["available"] and affordable(state, manifest, k)

        endgame = f["remaining"] / max(state.budget.total, 1e-9) < self.endgame_fraction

        # --- retrieval ---------------------------------------------------------
        if avail("retrieval"):
            if not f["has_identity"]:
                scores["retrieval"] = 1.0
                reasons["retrieval"] = "target is not resolved; nothing else can proceed"
                params["retrieval"] = {"fetch": "identity"}
            elif not f["has_actives"] and not endgame:
                scores["retrieval"] = 0.85
                reasons["retrieval"] = (
                    "no known actives yet; generating without seeds wastes budget"
                )
                params["retrieval"] = {"fetch": "actives"}
            elif not f["has_structure"] and "docking" in f["available"] and not endgame:
                scores["retrieval"] = 0.8
                reasons["retrieval"] = "no structure selected; docking cannot run without one"
                params["retrieval"] = {"fetch": "structures"}
            else:
                scores["retrieval"] = 0.05
                reasons["retrieval"] = "target data already complete"
                params["retrieval"] = {"fetch": "all"}

        # --- properties --------------------------------------------------------
        if avail("properties"):
            n = f["n_unprofiled"]
            if n > 0:
                # Nearly free and it informs every later choice, so it outranks anything
                # except resolving the target.
                scores["properties"] = 0.9 if n >= 5 else 0.6
                reasons["properties"] = f"{n} molecule(s) have no property panel; this is cheap"
                params["properties"] = {}
            else:
                scores["properties"] = 0.02
                reasons["properties"] = "every molecule already has a property panel"
                params["properties"] = {}

        # --- docking -----------------------------------------------------------
        if avail("docking"):
            n = f["n_unscored"]
            if not f["has_structure"]:
                scores["docking"] = 0.0
                reasons["docking"] = "no receptor structure selected yet"
            elif n >= self.dock_batch_floor:
                scores["docking"] = 0.75 if not endgame else 0.5
                reasons["docking"] = (
                    f"{n} molecule(s) have no binding estimate; this is the largest "
                    "open uncertainty and the most expensive thing to resolve"
                )
            elif n > 0:
                scores["docking"] = 0.3
                reasons["docking"] = (
                    f"only {n} unscored molecule(s); below the batch floor of "
                    f"{self.dock_batch_floor}, the fixed overhead is not yet repaid"
                )
            else:
                scores["docking"] = 0.0
                reasons["docking"] = "nothing left to score"
            params["docking"] = {}

        # --- generation --------------------------------------------------------
        if avail("generation"):
            if endgame:
                scores["generation"] = 0.05
                reasons["generation"] = (
                    "endgame: new molecules could not be scored before the budget runs out"
                )
            elif f["n_molecules"] == 0:
                scores["generation"] = 0.4
                reasons["generation"] = "no molecules at all; generate even without seeds"
            elif f["n_unscored"] >= self.dock_batch_floor * 3:
                scores["generation"] = 0.1
                reasons["generation"] = (
                    "a large backlog of unscored molecules already exists; "
                    "generating more would not add information"
                )
            elif f["stalled_for"] >= self.stall_limit:
                scores["generation"] = 0.8
                reasons["generation"] = (
                    f"no improvement for {f['stalled_for']} steps; the current pool is "
                    "mined out and the run needs new chemistry"
                )
            else:
                scores["generation"] = 0.55
                reasons["generation"] = "expand the pool with variants of the best molecules"
            params["generation"] = {}

        # Anything discovered that these rules know nothing about -- a plugin -- still gets
        # scored and reported, rather than vanishing from the trace.
        score_unknown(state, manifest, f, scores, reasons, params)

        return scores, reasons, params

    def decide(self, state: RunState, manifest: list[Capability]) -> Decision:
        f = assess(state, manifest)
        scores, reasons, params = self._score_options(state, manifest)

        viable = {k: v for k, v in scores.items() if v > 0.0}
        if not viable:
            return Decision.stop(
                "no subsystem is available, affordable and useful: "
                + "; ".join(f"{k} ({reasons.get(k, 'n/a')})" for k in scores)
                or "nothing available",
                policy=self.name,
                source="rule",
            )

        chosen = max(viable, key=lambda k: viable[k])

        # Stop when the best remaining action is one this policy itself rates as near
        # worthless. Budget left over is not waste -- repeating a no-op until a safety guard
        # trips is. This is also the behaviour that distinguishes a policy that knows when it
        # is finished from one that merely spends everything.
        if viable[chosen] < self.min_useful_score:
            return Decision(
                action="stop",
                rationale=(
                    f"nothing worth doing: the best remaining option ({chosen}) scores "
                    f"{viable[chosen]:.2f}, below the useful threshold of "
                    f"{self.min_useful_score:.2f}. {f['remaining']:.1f} units left unspent."
                ),
                considered=alternatives_from_scores(scores, reasons, None),
                policy=self.name,
                source="rule",
            )

        return Decision(
            action="call",
            subsystem=chosen,
            params=params.get(chosen, {}),
            rationale=self._rationale(chosen, viable, reasons),
            considered=alternatives_from_scores(scores, reasons, chosen),
            expected_gain=f"rule score {viable[chosen]:.2f}",
            policy=self.name,
            source="rule",
        )

    def _rationale(self, chosen: str, viable: dict[str, float], reasons: dict[str, str]) -> str:
        """Explain the choice, not just the score.

        Each rule's ``reason`` justifies the *score* it assigned, which reads backwards when
        a low-scoring option wins because everything else scored lower still. Saying "chose
        generation: generating more would not add information" is actively misleading in a
        trace a reader is using to judge the routing. So a weak winner is labelled as one,
        and the runner-up is named to show what it was weak relative to.
        """
        score = viable[chosen]
        reason = reasons.get(chosen, "")
        rest = sorted(
            ((k, v) for k, v in viable.items() if k != chosen), key=lambda kv: -kv[1]
        )
        if score >= 0.5:
            return f"{reason} (score {score:.2f}, the clear best option)."
        if not rest:
            return f"{reason} (score {score:.2f}); it is the only option available."
        runner, runner_score = rest[0]
        return (
            f"Weak choice: {reason} (score {score:.2f}). Taken anyway because every "
            f"alternative scored lower -- next best was {runner} at {runner_score:.2f}."
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "dock_batch_floor": self.dock_batch_floor,
            "endgame_fraction": self.endgame_fraction,
            "stall_limit": self.stall_limit,
        }
