"""What each routing policy is, stated in a form a scientist can audit before choosing one.

Choosing a routing policy is the one decision this system does *not* make for you. That is a
design position, not an omission: the policy is the independent variable of the experiment,
so handing it to the human keeps the comparison honest and keeps the person running the study
accountable for the arm they report. A card therefore has to be enough to choose on.

Each card states the same six things, so two policies can be read side by side:

``reads``
    The exact fields of ``RunState.summary()`` the policy consults. Every arm is *offered*
    the identical summary -- that is what makes the ablation valid -- but they differ in how
    much of it they actually look at, and that difference is the method.
``emits``
    Which fields of the resulting ``Decision`` carry real information for this policy.
``rule``
    How it chooses, in prose, with no hedging.
``example``
    A worked input/output pair. For every rule-based policy this is **computed by running
    the policy** on a fixed synthetic state at import time, so the example cannot drift away
    from the implementation the way a hand-written one would. The LLM card carries a recorded
    illustration instead, marked as such, because generating it would cost an API call.
``pattern``
    The family of published system this arm stands in for, so a result can be read as
    evidence about a literature position rather than about one implementation.
``caveat``
    The thing that would make a reviewer distrust a result from this arm.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class PolicyCard:
    """An auditable description of one routing policy."""

    name: str
    title: str
    one_line: str
    role: str
    """What picking this arm commits the scientist to."""
    rule: str
    reads: tuple[str, ...]
    emits: tuple[str, ...]
    pattern: str
    reference: str
    determinism: str
    calls_api: bool
    use_when: str
    caveat: str
    example: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --- the fixed synthetic state used for every worked example --------------------
#
# One state for all arms, so the examples are directly comparable: any difference between
# two cards' outputs is a difference in the rules, not in the situation. It is deliberately
# mid-run and slightly awkward -- molecules exist, some are unscored, the budget is half
# gone, nothing has improved for two rounds -- because that is where policies disagree. On
# step 0 every sensible policy does the same thing and the examples would teach nothing.

EXAMPLE_SITUATION = (
    "Mid-run on a JAK2 campaign: the target is resolved to a structure, 120 molecules are "
    "in hand, 40 of them have never been docked, half the budget is gone and the best score "
    "has not moved for two rounds."
)


def _example_state():
    """Build the shared synthetic state. Pure in-memory; touches no network and no disk."""
    from ..budget import BudgetLedger
    from ..orchestrator.state import Decision, DecisionRecord, RunState
    from ..types import TargetSpec

    state = RunState(
        run_id="example",
        goal="find selective JAK2 inhibitors with good oral bioavailability",
        target=TargetSpec(
            query="JAK2",
            uniprot="O60674",
            chembl_id="CHEMBL2971",
            pdb_id="7LL4",
            gene="JAK2",
            name="Tyrosine-protein kinase JAK2",
            organism="Homo sapiens",
            sequence="M" * 40,
        ),
        budget=BudgetLedger(total=120.0),
        seed=42,
    )
    state.step = 7
    state.facts = {
        "target_identity": {"gene": "JAK2", "organism": "Homo sapiens"},
        "known_actives": {"n": 60, "chembl_target": "CHEMBL2971"},
        "pdb_shortlist": [{"pdb_id": "7LL4", "ligand_code": "Y5D", "score": 14.2}],
        "receptor": {"pdb_id": "7LL4", "chain": "A"},
    }

    # 120 molecules: 80 fully worked up and scored, 40 freshly generated and untouched.
    # Distinct homologues rather than a short repeating pattern: the state dedupes on
    # canonical SMILES, so a cycling template would silently collapse to a handful of
    # molecules and the example would contradict the situation it describes.
    scored = [
        {
            "smiles": "C" * (i + 2) + "Oc1ccc(cc1)C(=O)Nc1ccncc1",
            "props": {"panel_version": "v1"},
            "scores": {"vina_score": 7.0 + (i % 23) * 0.12},
        }
        for i in range(80)
    ]
    fresh = [{"smiles": "N#CC" + "C" * (i + 2) + "c1cc(F)ccc1N"} for i in range(40)]
    state.add_records(scored, origin="retrieval")
    state.add_records(fresh, origin="generation")

    state.budget.charge("retrieval", 2.0, step=1)
    state.budget.charge("properties", 6.0, step=2)
    state.budget.charge("docking", 52.0, step=3)

    # Two completed steps that did not improve the best value, so `rounds_since_improvement`
    # reports a genuine stall rather than a fabricated one.
    best = state.best_value
    for step in (5, 6):
        state.history.append(
            DecisionRecord(
                step=step,
                decision=Decision(
                    action="call", subsystem="generation", params={}, policy="example"
                ),
                executed=True,
                ok=True,
                best_after=best,
            )
        )
    return state


def _worked_example(policy_name: str) -> dict:
    """Run a rule-based policy on the shared state and record what went in and came out."""
    from ..subsystems.registry import build_registry
    from . import build_policy

    state = _example_state()
    manifest = build_registry().capabilities()
    summary = state.summary()
    policy = build_policy(policy_name, seed=42)
    policy.reset(state)
    decision = policy.decide(state, manifest)

    return {
        "situation": EXAMPLE_SITUATION,
        "computed": True,
        "input": {
            "note": "Excerpted from the state.summary() every arm receives.",
            "step": summary["step"],
            "budget": {
                "spent": summary["budget"]["spent"],
                "remaining": summary["budget"]["remaining"],
                "total": summary["budget"]["total"],
            },
            "molecules": {
                "total": summary["molecules"]["total"],
                "without_properties": summary["molecules"]["without_properties"],
                "without_vina_score": summary["molecules"].get("without_vina_score"),
            },
            "objective": {
                "key": summary["objective"]["key"],
                "best": summary["objective"]["best"],
            },
            "rounds_since_improvement": summary["rounds_since_improvement"],
            "facts_available": summary["facts_available"],
        },
        "output": decision.to_dict(),
    }


# --- the cards ------------------------------------------------------------------

_CARD_DEFS: tuple[PolicyCard, ...] = (
    PolicyCard(
        name="fixed",
        title="Fixed pipeline",
        one_line="Runs each stage in a fixed order without using run results to choose.",
        role="Baseline for a pipeline that does not adapt its routing to run results.",
        rule=(
            "Walk a fixed list -- retrieval, properties, docking, generation -- then cycle "
            "the last three until the budget runs out. The same order every time, whatever "
            "the run has learned."
        ),
        reads=("step",),
        emits=("subsystem", "params"),
        pattern="Hard-coded discovery pipeline",
        reference="Fixed-order baseline with full stage coverage.",
        determinism="Fully deterministic. Identical for every seed and every target.",
        calls_api=False,
        use_when="Include it as a baseline in policy comparisons.",
        caveat="Does not adapt to unavailable tools or changes in run state.",
    ),
    PolicyCard(
        name="random",
        title="Random choice",
        one_line="Picks uniformly among the subsystems it can currently afford.",
        role="Control for routing with no information about run progress.",
        rule="Sample uniformly from the affordable, available subsystems. Seeded.",
        reads=("budget.remaining", "unavailable"),
        emits=("subsystem",),
        pattern="Control",
        reference="Standard ablation control; no published system proposes this.",
        determinism="Deterministic given the seed; differs across seeds by design.",
        calls_api=False,
        use_when="Compare across multiple seeds to account for random variation.",
        caveat="A single seed does not show variation across random choices.",
    ),
    PolicyCard(
        name="roundrobin",
        title="Round robin",
        one_line="Rotates through the available subsystems in a fixed cycle.",
        role="Control for a fixed rotation through available subsystems.",
        rule="Advance to the next affordable subsystem in a fixed rotation each step.",
        reads=("budget.remaining", "unavailable"),
        emits=("subsystem",),
        pattern="Control",
        reference="Standard ablation control.",
        determinism="Fully deterministic.",
        calls_api=False,
        use_when="Compare alongside random and adaptive policies.",
        caveat=(
            "Spends evenly across subsystems regardless of cost, so it burns budget on "
            "docking at the same rate as on the near-free property panel."
        ),
    ),
    PolicyCard(
        name="scripted",
        title="Scripted rules",
        one_line="Hand-written rules reading exactly the state the LLM reads.",
        role="Compare model-based routing with deterministic rules using the same run state.",
        rule=(
            "Score every affordable subsystem on the current situation: resolve the target "
            "first, profile molecules before paying to dock them, dock what is unscored, "
            "generate when scored molecules exist to breed from, and stop when the best "
            "option scores below a usefulness floor. Highest score wins; ties break by a "
            "fixed order."
        ),
        reads=(
            "target.resolved",
            "target.pdb_id",
            "molecules.without_properties",
            "molecules.without_vina_score",
            "objective.best",
            "budget.remaining",
            "rounds_since_improvement",
            "facts_available",
            "unavailable",
        ),
        emits=("subsystem", "params", "rationale", "considered"),
        pattern="Expert heuristic routing",
        reference="Hand-written routing rules provide a deterministic policy baseline.",
        determinism="Fully deterministic.",
        calls_api=False,
        use_when="Include it when comparing the LLM policy with rules-based routing.",
        caveat="Inspect the trace to confirm the rules are making sensible decisions.",
    ),
    PolicyCard(
        name="oracle",
        title="Oracle (answer key)",
        one_line="Reads the known-actives answer key and routes toward it.",
        role="Upper-bound reference that uses known actives unavailable to a prospective campaign.",
        rule=(
            "Consult the retrieved ChEMBL actives and their measured potencies, then route "
            "to confirm what the answer key already indicates."
        ),
        reads=("facts_available", "objective.top", "budget.remaining"),
        emits=("subsystem", "params", "rationale"),
        pattern="Ceiling",
        reference="Standard upper-bound control in retrospective benchmarks.",
        determinism="Fully deterministic given the retrieved actives.",
        calls_api=False,
        use_when="Use as an upper bound when interpreting retrospective benchmark results.",
        caveat=(
            "Leaks the evaluation target into the method. Any number from this arm is "
            "meaningless as a performance claim."
        ),
    ),
    PolicyCard(
        name="llm",
        title="LLM planner (Claude)",
        one_line="Claude reads the state summary and the capability manifest and chooses.",
        role="Uses a language model to choose the next subsystem from the current run state.",
        rule=(
            "Send the system prompt, the capability manifest (cached, stable within a run) "
            "and the current state summary (volatile, last). The response is constrained to "
            "a JSON schema naming one subsystem, its parameters, a rationale and the "
            "alternatives considered. Parameters outside a subsystem's declared range are "
            "clamped, an unavailable subsystem is rejected, and any failure degrades to the "
            "scripted decision recorded as source=fallback."
        ),
        reads=("the entire state.summary()", "the capability manifest"),
        emits=("subsystem", "params", "rationale", "considered", "expected_gain"),
        pattern="Adaptive cross-stage routing",
        reference=(
            "The gap this system targets. Published adaptive routing (e.g. TRACE-style tool "
            "selection) operates inside a single stage; routing across stages under an "
            "explicit budget is what is being tested here."
        ),
        determinism=(
            "Not deterministic. Pinned for replay by a response cache keyed on "
            "sha256(backend, system, prompt, schema, version) -- a cached run reproduces "
            "its decision sequence exactly, and offline mode makes a miss an error."
        ),
        calls_api=True,
        use_when="Compare with scripted routing using the same goal, seed, and budget.",
        caveat=(
            "Costs money and wall time per decision, and a silent fallback looks like a "
            "normal run. Check the LLM log for source=fallback before reporting a result."
        ),
        example={
            "situation": EXAMPLE_SITUATION,
            "computed": False,
            "note": (
                "Recorded illustration, not computed at load time -- generating it would "
                "cost an API call. The live log for any run shows the real exchanges."
            ),
            "input": {
                "note": "The full state.summary() plus the capability manifest.",
                "step": 7,
                "budget": {"spent": 60.0, "remaining": 60.0, "total": 120.0},
                "molecules": {
                    "total": 120,
                    "without_properties": 40,
                    "without_vina_score": 40,
                },
                "objective": {"key": "vina_score", "best": 9.64},
                "rounds_since_improvement": 2,
                "facts_available": [
                    "known_actives",
                    "pdb_shortlist",
                    "receptor",
                    "target_identity",
                ],
            },
            "output": {
                "action": "call",
                "subsystem": "docking",
                "params": {"max_molecules": 8, "exhaustiveness": 8},
                "rationale": (
                    "40 generated molecules have never been scored, so the best value is "
                    "measured over a stale subset and the two-round stall may be an "
                    "artefact of not having looked. Docking 8 of them costs 8 units of the "
                    "60 remaining and either breaks the stall or confirms it is real."
                ),
                "considered": [
                    {
                        "subsystem": "generation",
                        "score": 0.55,
                        "reason": "More molecules without scoring the current batch "
                        "compounds the backlog.",
                    },
                    {
                        "subsystem": "properties",
                        "score": 0.3,
                        "reason": "Nearly free, but a panel does not move the objective.",
                    },
                    {
                        "subsystem": "retrieval",
                        "score": 0.05,
                        "reason": "Target and actives are already in hand.",
                    },
                ],
                "expected_gain": "Breaks or confirms the stall for 8 of 60 remaining units.",
                "policy": "llm",
                "source": "llm",
            },
        },
    ),
)

CARDS: dict[str, PolicyCard] = {c.name: c for c in _CARD_DEFS}

_EXAMPLES_BUILT = False


def _ensure_examples() -> None:
    """Compute the rule-based worked examples once, lazily.

    Lazy because building them imports the subsystem registry, and a card lookup should not
    drag that in at module import. A failure here degrades the card to 'no example' rather
    than taking down the endpoint -- an example is documentation, not a result.
    """
    global _EXAMPLES_BUILT  # noqa: PLW0603 - module-level memo
    if _EXAMPLES_BUILT:
        return
    _EXAMPLES_BUILT = True
    for name, card in CARDS.items():
        if card.example:
            continue
        try:
            object.__setattr__(card, "example", _worked_example(name))
        except Exception as exc:  # noqa: BLE001
            object.__setattr__(
                card, "example", {"error": f"{type(exc).__name__}: {exc}", "computed": False}
            )


def get_card(name: str) -> PolicyCard | None:
    _ensure_examples()
    return CARDS.get(name)


def all_cards() -> list[dict]:
    """Every card, in ladder order, with examples filled in."""
    _ensure_examples()
    return [CARDS[c.name].to_dict() for c in _CARD_DEFS]


__all__ = ["CARDS", "EXAMPLE_SITUATION", "PolicyCard", "all_cards", "get_card"]
