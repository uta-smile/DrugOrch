"""The workbench: generation as an operation rather than as a step in an experiment.

The property that matters most here is one a "does it return something" test would miss.
Every number this project quotes is downstream of a conformer, and conformer generation is
stochastic -- so a workbench that returned different coordinates for the same molecule on
two requests would make everything computed from them unquotable. The determinism tests are
therefore the point of this file, not a formality.

Nothing here touches the network or Vina. Docking is exercised through the real code path in
the API tests only where an executable is present; what is checked here is the part that
decides *what gets docked* and *what gets reported back*.
"""

from __future__ import annotations

import pytest

from drugorch import workbench as wb
from drugorch.chem import canonical

RUXOLITINIB = "N#CC[C@H](C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"


# --- SMILES -> 3D ---------------------------------------------------------------


def test_embedding_the_same_molecule_twice_gives_the_same_coordinates():
    """The property every downstream number depends on.

    A SMILES has no geometry and distance-geometry embedding is stochastic. If two requests
    for the same molecule produced different conformers, any energy, descriptor or docking
    score computed from one could not be reconciled with the other -- so the seed is derived
    from the molecule itself rather than drawn.
    """
    a = wb.embed_3d(RUXOLITINIB)
    b = wb.embed_3d(RUXOLITINIB)

    assert a["sdf"] == b["sdf"]
    assert a["seed"] == b["seed"]


def test_a_different_run_seed_gives_a_different_conformer():
    """Reproducible is not the same as fixed: the seed has to actually do something."""
    a = wb.embed_3d(RUXOLITINIB, seed=1)
    b = wb.embed_3d(RUXOLITINIB, seed=2)

    assert a["seed"] != b["seed"]
    assert a["sdf"] != b["sdf"]


def test_the_structure_is_three_dimensional_and_has_hydrogens():
    """A 'conformer' that is flat, or that has no hydrogens, is not one.

    Worth asserting explicitly because both failure modes produce a well-formed SDF that
    renders without complaint -- it just would not be the molecule anyone asked for.
    """
    out = wb.embed_3d(ASPIRIN)
    coords = []
    for line in out["sdf"].splitlines()[4 : 4 + out["n_atoms"]]:
        parts = line.split()
        coords.append(tuple(float(p) for p in parts[:3]))

    assert len(coords) == out["n_atoms"]
    assert out["n_atoms"] > out["n_heavy_atoms"], "hydrogens were not added"
    assert any(abs(z) > 0.01 for _, _, z in coords), "every atom is in one plane"


def test_an_unreadable_molecule_is_refused_with_its_own_reason():
    with pytest.raises(wb.WorkbenchError, match="not a readable SMILES"):
        wb.embed_3d("this-is-not-a-molecule")


def test_one_bad_molecule_does_not_discard_the_batch():
    """A batch is usually a generated set, and one that will not embed is a fact about it."""
    good, failed = wb.embed_many([ASPIRIN, "nonsense", CAFFEINE])

    assert [g["smiles"] for g in good] == [canonical(ASPIRIN), canonical(CAFFEINE)]
    assert len(failed) == 1
    assert failed[0]["smiles"] == "nonsense"


def test_the_panel_travels_with_the_structure():
    """Geometry alone does not say whether a molecule is worth anything."""
    out = wb.embed_3d(RUXOLITINIB)

    assert 0.0 <= out["panel"]["scores"]["qed"] <= 1.0
    assert out["panel"]["flags"]["lipinski_violations"] == 0
    assert out["n_rotatable_bonds"] >= 0


# --- seed -> analogues ----------------------------------------------------------


def test_elaboration_is_reproducible_under_its_seed():
    a = wb.elaborate([RUXOLITINIB], n=8, seed=3, with_properties=False)
    b = wb.elaborate([RUXOLITINIB], n=8, seed=3, with_properties=False)

    assert [c["smiles"] for c in a["children"]] == [c["smiles"] for c in b["children"]]


def test_a_single_seed_produces_only_mutations_and_says_so():
    """Crossover needs two distinct parents, so with one seed it cannot run at all.

    Silently producing mutations while the caller believes crossover is contributing would
    misrepresent where the chemistry came from, so the note states it.
    """
    out = wb.elaborate([RUXOLITINIB], n=6, seed=11, with_properties=False)

    assert set(out["operators"]) <= {"mutation"}
    assert "crossover needs two seeds" in out["note"]


def test_children_are_new_molecules_and_carry_their_parentage():
    out = wb.elaborate([RUXOLITINIB], n=6, seed=5, with_properties=False)
    seeds = set(out["seeds"])
    children = [c["smiles"] for c in out["children"]]

    assert len(set(children)) == len(children), "duplicates were returned"
    assert not seeds & set(children), "a seed was returned as its own analogue"
    assert all(c["parents"] and c["parents"][0] in seeds for c in out["children"])


def test_a_failed_crossover_is_attributed_to_crossover():
    """The answer to "why did I get no crossovers?".

    Crossover cuts at acyclic single bonds, so two small rigid molecules offer almost
    nowhere to cut and it fails on nearly every attempt. Mutation then quietly fills the
    quota, and a single 'invalid' counter made that look like a choice rather than a limit.
    """
    out = wb.elaborate([ASPIRIN, CAFFEINE], n=6, seed=5, mutation_rate=0.5,
                       with_properties=False)

    assert not out["operators"].get("crossover")
    assert out["rejected"].get("crossover_failed", 0) > 0
    assert "nowhere to cut" in out["note"]


def test_two_flexible_seeds_do_recombine():
    """The negative test above must not be passing because crossover is simply broken."""
    out = wb.elaborate(
        [RUXOLITINIB, "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"],
        n=10,
        seed=5,
        with_properties=False,
    )

    assert out["operators"].get("crossover", 0) > 0


def test_unreadable_seeds_are_refused():
    with pytest.raises(wb.WorkbenchError, match="could be read"):
        wb.elaborate(["not-a-molecule"], n=4)


def test_the_batch_ceiling_is_enforced():
    """These endpoints are synchronous; an unbounded request holds a worker and draws nothing."""
    out = wb.elaborate([RUXOLITINIB], n=10_000, seed=1, with_properties=False)

    assert len(out["children"]) <= wb.MAX_BATCH


# --- jobs -----------------------------------------------------------------------


def test_a_job_reports_a_failure_rather_than_killing_the_worker():
    board = wb.JobBoard()

    def boom(_job):
        raise ValueError("deliberate")

    job = board.submit("test", boom)
    for _ in range(200):
        if job.status != "running":
            break
        import time

        time.sleep(0.01)

    assert job.status == "failed"
    assert "deliberate" in (job.error or "")
    assert board.get(job.job_id) is job


def test_a_finished_job_carries_its_result_and_a_running_one_does_not():
    """The snapshot withholds the result until it is complete.

    A half-filled result handed to a polling page would be read as the answer.
    """
    board = wb.JobBoard()
    job = board.submit("test", lambda j: {"rows": [1, 2, 3]})
    for _ in range(200):
        if job.status != "running":
            break
        import time

        time.sleep(0.01)

    assert job.snapshot()["result"] == {"rows": [1, 2, 3]}
    job.status = "running"
    assert job.snapshot()["result"] is None
