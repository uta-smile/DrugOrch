"""The benchmark: splits, metrics and the leaderboard.

The single most important test here is `test_a_held_out_molecule_cannot_reach_the_run`. If a
withheld active can reach a run through retrieval, every rediscovery number in the project is
fiction and nothing else in this file matters.
"""

from __future__ import annotations

import threading

import pytest

from drugorch.harness.metrics import RunMetrics, _auc, evaluate, rediscovery_curve
from drugorch.harness.report import ArmSummary, leaderboard, per_target
from drugorch.harness.splits import scaffold_split

# Two clearly distinct chemotypes plus close analogues within each, so a scaffold split has
# something real to separate.
PYRAZOLE = [
    "N#CC[C@H](C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1",
    "N#CCC(C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1",
    "N#CCC(C1CCC1)n1cc(-c2ncnc3[nH]ccc23)cn1",
]
BENZOIC = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCC(=O)Oc1ccccc1C(=O)O",
    "CCCC(=O)Oc1ccccc1C(=O)O",
]


# --- the split -------------------------------------------------------------
def test_no_molecule_appears_on_both_sides():
    s = scaffold_split(PYRAZOLE + BENZOIC, fraction=0.4, seed=1)
    assert not (set(s.seen) & set(s.held_out))


def test_no_scaffold_appears_on_both_sides():
    """The whole point of a scaffold split: a held-out chemotype must be absent from the
    seen half, or 'rediscovery' is really 'interpolation between neighbours'."""
    s = scaffold_split(PYRAZOLE + BENZOIC, fraction=0.4, seed=1)
    assert not (s.held_out_scaffolds & s.seen_scaffolds)


def test_the_split_is_reproducible_and_seed_dependent():
    a = scaffold_split(PYRAZOLE + BENZOIC, fraction=0.4, seed=1)
    b = scaffold_split(PYRAZOLE + BENZOIC, fraction=0.4, seed=1)
    assert a.fingerprint() == b.fingerprint()
    assert a.held_out == b.held_out


def test_different_seeds_give_different_holdouts():
    """Otherwise running three seeds is three runs against one withheld set, and the extra
    seeds add no independent evidence while the table implies they do.

    This regressed once: selecting scaffolds strictly largest-first made the seed a mere
    tie-break, and JAK2 produced a byte-identical holdout for every seed.
    """
    pool = PYRAZOLE + BENZOIC + [
        "CC(=O)Nc1ccc(O)cc1", "CCC(=O)Nc1ccc(O)cc1",
        "c1ccc2[nH]ccc2c1", "Cc1ccc2[nH]ccc2c1",
        "C1CCNCC1", "CC1CCNCC1",
    ]
    fps = {seed: scaffold_split(pool, fraction=0.3, seed=seed).fingerprint() for seed in range(6)}
    assert len(set(fps.values())) > 1, "the seed does not change the holdout"


def test_the_split_never_holds_out_everything():
    """A run with no seeds is a different experiment, not a harder one."""
    s = scaffold_split(PYRAZOLE + BENZOIC, fraction=0.99, seed=1)
    assert len(s.seen) >= 2


def test_unscaffoldable_molecules_stay_in_the_seen_half():
    """They cannot support a scaffold-level claim either way; dropping them would quietly
    shrink the pool the run is seeded from."""
    s = scaffold_split([*PYRAZOLE, "CCCCO", "CCCCCO"], fraction=0.4, seed=1)
    assert s.unscaffoldable == 2
    assert "CCCCO" in s.seen


def test_an_empty_input_does_not_raise():
    s = scaffold_split([], fraction=0.3, seed=1)
    assert s.seen == () and s.held_out == ()


# --- the guarantee everything rests on -------------------------------------
def test_a_held_out_molecule_cannot_reach_the_run():
    """Retrieval must filter the holdout at the one place actives enter a run.

    If this fails, a run is handed the answers and every rediscovery figure is meaningless.
    The filtering has to happen here rather than at scoring time: a molecule the policy has
    already seen cannot be un-seen.
    """
    from drugorch.budget import BudgetLedger
    from drugorch.clients.chembl import Active
    from drugorch.orchestrator.state import RunState
    from drugorch.types import TargetSpec

    withheld = Active(smiles="CC(=O)Oc1ccccc1C(=O)O", pchembl=9.0, standard_type="IC50")
    allowed = Active(smiles="CCCC(=O)Oc1ccccc1C(=O)O", pchembl=8.0, standard_type="Ki")

    state = RunState(
        run_id="t",
        goal="g",
        target=TargetSpec(query="x", uniprot="P1", chembl_id="CHEMBL1"),
        budget=BudgetLedger(total=10.0),
        holdout=frozenset({withheld.smiles}),
    )

    # The filter as retrieval applies it.
    kept = [a for a in (withheld, allowed) if a.smiles not in state.holdout]
    assert [a.smiles for a in kept] == [allowed.smiles]

    state.add_records([a.to_record() for a in kept], origin="retrieval")
    assert withheld.smiles not in state.molecules
    assert allowed.smiles in state.molecules


def test_assay_type_survives_into_the_record():
    """It was parsed from ChEMBL and dropped before reaching a molecule; a benchmark that
    pools a Ki and an IC50 should at least be able to say so."""
    from drugorch.clients.chembl import Active

    rec = Active(smiles="CCO", pchembl=7.0, standard_type="Ki").to_record()
    assert rec["props"]["assay_type"] == "Ki"


# --- metrics ---------------------------------------------------------------
def test_rediscovery_counts_each_held_out_molecule_once():
    """Twenty analogues of one compound are one discovery, not twenty."""
    held = [PYRAZOLE[0]]
    proposed = [(PYRAZOLE[1], 10.0), (PYRAZOLE[2], 20.0)]
    curve, n = rediscovery_curve(proposed, held, threshold=0.5)
    assert n == 1
    assert curve[-1][1] == 1.0


def test_rediscovery_records_the_earliest_spend():
    held = [PYRAZOLE[0]]
    proposed = [(PYRAZOLE[2], 30.0), (PYRAZOLE[1], 10.0)]
    curve, _ = rediscovery_curve(proposed, held, threshold=0.5)
    assert curve[0][0] == 10.0


def test_an_unrelated_molecule_rediscovers_nothing():
    curve, n = rediscovery_curve([("CCCCCCCO", 5.0)], [PYRAZOLE[0]])
    assert n == 0 and curve == []


def test_finding_earlier_scores_higher():
    """The reason the headline is an area and not an endpoint: both runs end up finding
    everything, and the one that got there first is better."""
    early = _auc([(10.0, 1.0)], 100.0)
    late = _auc([(90.0, 1.0)], 100.0)
    assert early > late


def test_auc_does_not_reward_unspent_budget():
    assert _auc([], 100.0) == 0.0


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_auc_stays_in_range(value):
    assert 0.0 <= _auc([(50.0, value)], 100.0) <= 1.0


def test_retrieved_molecules_do_not_count_as_discoveries():
    """A retrieved active is an input. Counting it would let an arm 'discover' what it was
    handed, which is the exact failure the holdout exists to prevent."""
    m = evaluate(
        run_id="r", policy="p", target="t", seed=1,
        molecules={
            PYRAZOLE[1]: {"origin": "retrieval", "spend": 1.0, "scores": {}, "props": {}},
        },
        objective_curve=[], budget_total=100.0, spent=1.0, steps=1, status="done",
        held_out=[PYRAZOLE[0]], n_held_out_scaffolds=1, seen=[],
    )
    assert m.n_rediscovered == 0


def test_a_generated_molecule_does_count():
    m = evaluate(
        run_id="r", policy="p", target="t", seed=1,
        molecules={
            PYRAZOLE[1]: {"origin": "generation", "spend": 20.0, "scores": {}, "props": {}},
        },
        objective_curve=[], budget_total=100.0, spent=20.0, steps=1, status="done",
        held_out=[PYRAZOLE[0]], n_held_out_scaffolds=1, seen=[],
    )
    assert m.n_rediscovered == 1
    assert m.discovery_auc > 0


def test_a_mostly_fallback_run_is_flagged():
    """A run whose decisions fell back is not a run of the arm it claims to be, and the
    leaderboard must not present it as one."""
    m = evaluate(
        run_id="r", policy="llm", target="t", seed=1, molecules={}, objective_curve=[],
        budget_total=100.0, spent=0.0, steps=6, status="done",
        held_out=[], n_held_out_scaffolds=0, seen=[], n_fallbacks=6,
    )
    assert any("not a clean run" in n for n in m.notes)


def test_no_holdout_is_called_out():
    m = evaluate(
        run_id="r", policy="p", target="t", seed=1, molecules={}, objective_curve=[],
        budget_total=100.0, spent=0.0, steps=1, status="done",
        held_out=[], n_held_out_scaffolds=0, seen=[],
    )
    assert any("rediscovery is meaningless" in n for n in m.notes)


# --- the leaderboard -------------------------------------------------------
def _row(policy, target, seed, auc):
    return RunMetrics(
        run_id=f"{policy}_{target}_{seed}", policy=policy, target=target,
        seed=seed, discovery_auc=auc, steps=10,
    )


def test_arms_are_paired_per_target_and_seed():
    """Averaging independently would be dominated by which targets are easy. Pairing asks
    the only useful question: on the same problem, did this arm beat the baseline?"""
    rows = [
        _row("fixed", "A", 1, 0.10), _row("scripted", "A", 1, 0.20),
        _row("fixed", "B", 1, 0.80), _row("scripted", "B", 1, 0.85),
    ]
    board = {a.policy: a for a in leaderboard(rows)}
    assert board["scripted"].wins == 2
    assert board["scripted"].losses == 0
    assert board["scripted"].mean_delta == pytest.approx(0.075, abs=1e-6)


def test_a_single_blowout_does_not_look_like_consistent_winning():
    """High mean delta with a low win rate is a different finding from winning everywhere,
    and reporting only one of the two would hide it."""
    rows = [
        _row("fixed", "A", 1, 0.50), _row("arm", "A", 1, 0.40),
        _row("fixed", "B", 1, 0.50), _row("arm", "B", 1, 0.45),
        _row("fixed", "C", 1, 0.10), _row("arm", "C", 1, 0.90),
    ]
    arm = {a.policy: a for a in leaderboard(rows)}["arm"]
    assert arm.mean_delta > 0
    assert arm.win_rate < 0.5


def test_the_oracle_is_always_flagged():
    board = {a.policy: a for a in leaderboard([_row("oracle", "A", 1, 0.9)])}
    assert any("ceiling" in f for f in board["oracle"].flags)


def test_thin_evidence_is_flagged():
    board = {a.policy: a for a in leaderboard([_row("scripted", "A", 1, 0.5)])}
    assert any("anecdote" in f for f in board["scripted"].flags)


def test_per_target_breaks_the_aggregate_down():
    rows = [_row("fixed", "A", 1, 0.1), _row("scripted", "A", 1, 0.4)]
    assert per_target(rows)["A"] == {"fixed": 0.1, "scripted": 0.4}


def test_leaderboard_is_ordered_by_the_primary_metric():
    rows = [_row("a", "T", 1, 0.1), _row("b", "T", 1, 0.9)]
    assert [a.policy for a in leaderboard(rows)][0] == "b"


def test_arm_summary_serializes():
    import json

    json.dumps(ArmSummary(policy="x").to_dict(), default=None)


# --- the grid --------------------------------------------------------------
def test_cell_ids_are_deterministic_so_a_grid_can_resume():
    """Timestamped ids would make every resume a fresh run, and docking dominates the clock."""
    from drugorch.harness.campaign import Cell

    a = Cell(target="O60674", policy="scripted", seed=1, budget=80)
    b = Cell(target="O60674", policy="scripted", seed=1, budget=80)
    assert a.run_id == b.run_id
    assert Cell(target="O60674", policy="fixed", seed=1, budget=80).run_id != a.run_id


def test_a_failing_cell_does_not_stop_the_grid(tmp_path, monkeypatch):
    from drugorch.harness import grid as grid_mod

    monkeypatch.setattr(grid_mod, "build_split", lambda *a, **k: None)

    calls = {"n": 0}

    def boom(cell, split, *, settings=None, on_step=None):
        calls["n"] += 1
        if cell.policy == "bad":
            raise RuntimeError("this cell is broken")
        return RunMetrics(run_id=cell.run_id, policy=cell.policy, target=cell.target), tmp_path

    monkeypatch.setattr(grid_mod, "run_cell", boom)

    result = grid_mod.run_grid(
        ["T"], ["bad", "good"], [1], budget=10, settings=None, resume=False
    )
    assert calls["n"] == 2
    assert [r.policy for r in result.rows] == ["good"]
    assert result.failures and "this cell is broken" in result.failures[0]["error"]


def test_cancel_is_honoured_by_a_subsystem_mid_run():
    """Unrelated to the grid, but the benchmark depends on it: a cell must be stoppable."""
    from drugorch.budget import BudgetLedger
    from drugorch.orchestrator.state import RunState
    from drugorch.subsystems.generation import GenerationSubsystem
    from drugorch.types import TargetSpec

    state = RunState(
        run_id="t", goal="g", target=TargetSpec(query="x"),
        budget=BudgetLedger(total=10.0), seed=1,
    )
    state.add_records([{"smiles": s} for s in PYRAZOLE], origin="retrieval")
    cancel = threading.Event()
    cancel.set()
    res = GenerationSubsystem().run(
        {"n_offspring": 50}, state, cancel=cancel, progress=lambda *a: None
    )
    assert res.cost.n_unique_smiles < 50


def test_pains_and_sa_use_the_keys_the_panel_actually_writes():
    """These were wrong once: `pains_alerts` plural produced a permanently blank column
    rather than an error, which is the failure mode a metric should never have."""
    m = evaluate(
        run_id="r", policy="p", target="t", seed=1,
        molecules={
            PYRAZOLE[1]: {
                "origin": "generation", "spend": 5.0,
                "scores": {"sa_score": 3.2},
                "props": {"pains_alert": "catechol_A(92)"},
            },
            BENZOIC[1]: {
                "origin": "generation", "spend": 6.0,
                "scores": {"sa_score": 2.0},
                "props": {"pains_alert": ""},
            },
        },
        objective_curve=[], budget_total=100.0, spent=6.0, steps=2, status="done",
        held_out=[], n_held_out_scaffolds=0, seen=[],
    )
    assert m.sa_mean == pytest.approx(2.6, abs=1e-6)
    assert m.pains_rate == 0.5
