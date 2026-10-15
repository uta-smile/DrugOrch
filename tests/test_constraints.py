"""Scoring a molecule against a MolDesignBench constraint set.

The load-bearing test here is :func:`test_reference_answers_score_perfectly`, which replays
real instances with recorded predictions. Every number this project reports on that benchmark
is a comparison against its answer key, so an evaluator that disagrees with the key makes all
of them noise -- and it would do so silently, since a wrong property value looks exactly like
a right one. The rest of the file pins the three things that were actually wrong on the way
to agreement: which RDKit computes a descriptor, what precision the comparison uses, and what
happens when no predictor is reachable.
"""

from __future__ import annotations

import pytest

from drugorch.harness.external.constraints import (
    BENCH_COLUMNS,
    NEEDS_PREDICTOR,
    Constraints,
    evaluate,
    evaluate_many,
    unsatisfied,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

# Four real instances with the predictions ADMET-AI returned for their reference answers,
# recorded so this runs without the worker. Replaying the live model here would make the
# test depend on a virtualenv, a subprocess and 5 seconds of model load to assert something
# about arithmetic.
CASES = [
    {
        "id": "generation_4072",
        # logP is 2.0000999... and the window closes at 2.0. Their tool rounds to two
        # decimals before comparing, so this reference answer passes for them and would fail
        # for us at full precision. One instance in 1,800, and it is the answer key.
        "smiles": "COc1ccc2c3c1O[C@H]1[C@]34CCN(C)[C@H](C2)[C@]42C=C[C@@]1(OC)C(C(=O)O)C2",
        "answer": {"properties": [{"property": "logP", "min": 0.0, "max": 2.0}],
                   "fragments": []},
        "predictions": {},
    },
    {
        "id": "generation_4454",
        # HBA is 4 under RDKit 2025.03.6 and 3 under 2026.09.1. The window starts at 4.
        "smiles": "O=C(O)c1ccc(-n2nc(C(=O)c3c(Cl)cccc3Cl)c3ccccc32)cc1",
        "answer": {"properties": [{"property": "HBA", "min": 4.0, "max": None}],
                   "fragments": []},
        "predictions": {"hydrogen_bond_acceptors": 4.0},
    },
    {
        "id": "mutag_window",
        "smiles": ASPIRIN,
        "answer": {"properties": [{"property": "Mutag", "min": 0.0, "max": 0.2}],
                   "fragments": []},
        "predictions": {"AMES": 0.051},
    },
    {
        "id": "fragment_and_bound",
        "smiles": ASPIRIN,
        "answer": {
            "properties": [{"property": "MW", "min": 150.0, "max": 200.0}],
            "fragments": [{"carboxylic acid": 1}],
        },
        "predictions": {"molecular_weight": 180.16},
    },
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_reference_answers_score_perfectly(case):
    """A reference answer must score exactly 1.0, with nothing skipped.

    Anything less means the evaluator and the benchmark disagree about what the constraint
    says, and every downstream number inherits that disagreement.
    """
    c = Constraints.from_answer(case["answer"])
    ev = evaluate(case["smiles"], c, predictions=case["predictions"])
    assert ev.skipped == 0, ev.details
    assert ev.csr == 1.0, unsatisfied(ev)
    assert ev.all_satisfied


def test_supplied_value_beats_the_local_descriptor():
    """When the benchmark supplies a property, that value wins.

    Not a preference. ``CalcNumHBA`` changed meaning between RDKit releases, so a locally
    computed HBA is a different quantity from the one in the answer key. The regression this
    guards is subtle: it would not raise, it would just score a sixth of every molecule's
    constraints against the wrong number.
    """
    answer = {"properties": [{"property": "HBA", "min": 4.0, "max": 4.0}], "fragments": []}
    c = Constraints.from_answer(answer)
    smiles = "O=C(O)c1ccc(-n2nc(C(=O)c3c(Cl)cccc3Cl)c3ccccc32)cc1"

    supplied = evaluate(smiles, c, predictions={"hydrogen_bond_acceptors": 4.0})
    assert supplied.details[0]["source"] == "benchmark"
    assert supplied.details[0]["value"] == 4.0
    assert supplied.csr == 1.0

    local = evaluate(smiles, c)
    assert local.details[0]["source"] == "local"
    assert local.skipped == 0, "HBA is computable locally; it must not be skipped"


def test_comparison_rounds_to_two_decimals():
    """Values are compared at the precision the benchmark compares them at."""
    answer = {"properties": [{"property": "logP", "min": 0.0, "max": 2.0}], "fragments": []}
    c = Constraints.from_answer(answer)
    # logP here is 2.0000999..., which is outside [0, 2.0] and inside it once rounded.
    smiles = "COc1ccc2c3c1O[C@H]1[C@]34CCN(C)[C@H](C2)[C@]42C=C[C@@]1(OC)C(C(=O)O)C2"
    ev = evaluate(smiles, c)
    assert ev.details[0]["value"] == 2.0
    assert ev.csr == 1.0


def test_unavailable_predictions_are_skipped_not_guessed():
    """A predicted property with no predictor is excluded, never defaulted.

    Scoring it as failed would punish a molecule for our missing model and scoring it as
    passed would reward it for the same; both produce a number that reads like a measurement.
    The denominator has to shrink instead.
    """
    answer = {
        "properties": [
            {"property": "Mutag", "min": 0.0, "max": 0.2},
            {"property": "MW", "min": 150.0, "max": 200.0},
        ],
        "fragments": [],
    }
    c = Constraints.from_answer(answer)
    ev = evaluate(ASPIRIN, c)

    assert ev.skipped == 1
    assert ev.evaluated == 1
    assert ev.csr == 1.0, "the one constraint we could check passed"
    statuses = {d["constraint"]: d.get("status") for d in ev.details}
    assert statuses["Mutag"] == "unavailable"
    assert all("value" not in d for d in ev.details if d["constraint"] == "Mutag")


def test_needs_predictor_covers_more_than_the_ml_endpoints():
    """Six descriptors are fetched too, because our RDKit is not the answer key's."""
    assert set(BENCH_COLUMNS) > NEEDS_PREDICTOR
    only_ml = Constraints.from_answer(
        {"properties": [{"property": "Mutag", "min": 0.0, "max": 1.0}], "fragments": []}
    )
    descriptor = Constraints.from_answer(
        {"properties": [{"property": "HBA", "min": 0.0, "max": 10.0}], "fragments": []}
    )
    local_only = Constraints.from_answer(
        {"properties": [{"property": "rotB", "min": 0.0, "max": 10.0}], "fragments": []}
    )
    assert only_ml.needs_predictor
    assert descriptor.needs_predictor, "HBA must be fetched; our CalcNumHBA disagrees"
    assert not local_only.needs_predictor, "rotB agrees across versions; do not wake a worker"


def test_contradictory_bounds_are_infeasible():
    """``min > max`` is how the benchmark marks 10% of its instances unsatisfiable."""
    c = Constraints.from_answer(
        {"properties": [{"property": "logP", "min": 4.0, "max": 3.5}], "fragments": []}
    )
    assert c.infeasible
    assert c.contradictions()

    disjoint = Constraints.from_answer(
        {
            "properties": [
                {"property": "MW", "min": 100.0, "max": 200.0},
                {"property": "MW", "min": 400.0, "max": 500.0},
            ],
            "fragments": [],
        }
    )
    assert disjoint.infeasible, "two windows that cannot both hold is also infeasible"

    assert not Constraints.from_answer(
        {"properties": [{"property": "logP", "min": 1.0, "max": 3.5}], "fragments": []}
    ).infeasible


def test_optimization_preserves_fragments_rather_than_counting_them():
    """On optimization tasks the fragment count is the *seed's*, not a requirement.

    Reading it as an exact count failed 185 reference answers. The goal says the group must
    survive the edit, so one or more satisfies it.
    """
    answer = {"properties": [], "fragments": [{"carboxylic acid": 2}]}
    meta = {
        "optimization_goal": {"preserve_fgs": ["carboxylic acid"]},
        "seed_smiles": ASPIRIN,
    }
    c = Constraints.from_answer(answer, meta=meta)
    assert c.fragments[0].preserve_only
    # Aspirin has one carboxylic acid; the seed had two. Preservation is satisfied.
    assert evaluate(ASPIRIN, c).csr == 1.0

    strict = Constraints.from_answer(answer)
    assert not strict.fragments[0].preserve_only
    assert evaluate(ASPIRIN, strict).csr == 0.0


def test_evaluate_many_asks_the_predictor_once():
    """One batch, not one call per molecule -- the worker costs seconds to wake."""
    calls: list[list[str]] = []

    def predictor(smiles):
        calls.append(list(smiles))
        return {s: {"AMES": 0.05} for s in smiles}

    c = Constraints.from_answer(
        {"properties": [{"property": "Mutag", "min": 0.0, "max": 0.2}], "fragments": []}
    )
    evs = evaluate_many([ASPIRIN, "CCO", "c1ccccc1"], c, predictor=predictor)

    assert len(calls) == 1
    assert len(calls[0]) == 3
    assert all(e.csr == 1.0 for e in evs)


def test_a_predictor_that_fails_degrades_to_skipping():
    """An unreachable model must not take the run down with it."""

    def predictor(smiles):
        raise RuntimeError("the worker is not available")

    c = Constraints.from_answer(
        {"properties": [{"property": "Mutag", "min": 0.0, "max": 0.2},
                        {"property": "rotB", "min": 0.0, "max": 10.0}],
         "fragments": []}
    )
    (ev,) = evaluate_many([ASPIRIN], c, predictor=predictor)
    assert ev.skipped == 1
    assert ev.evaluated == 1


def test_an_unparseable_molecule_is_invalid_not_zero():
    """A string that is not a molecule has no CSR, and must not read as a bad score."""
    c = Constraints.from_answer(
        {"properties": [{"property": "MW", "min": 0.0, "max": 500.0}], "fragments": []}
    )
    ev = evaluate("not a molecule at all", c)
    assert not ev.valid
    assert ev.evaluated == 0
    assert ev.skipped == c.total
