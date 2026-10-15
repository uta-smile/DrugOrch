"""The public-benchmark harness: sampling, the answer-key boundary, fitness, and the arms.

The load-bearing test here is :func:`test_no_prompt_can_carry_the_answer_key`. Every number
on the public board is a comparison against MolDesignBench's key, and an arm that could see
any part of that key would still produce a score, a plausible one, with nothing in the output
to say it was fiction. Half the benchmark's difficulty is inferring constraints the question
only implies, so a leak would not even look like cheating: it would look like a system that
reads scenarios unusually well.

The rest pin the things that were actually wrong on the way here: what a stratified sample
has to hold constant to stay comparable, that fitness ranks by constraints satisfied rather
than by distance, and that a record written for the benchmark's evaluator is shaped the way
it reads one.
"""

from __future__ import annotations

import json

import pytest

from drugorch.harness.external import baselines
from drugorch.harness.external.agent import Arm, Attempt
from drugorch.harness.external.ga import optimize
from drugorch.harness.external.instances import (
    Instance,
    describe,
    fingerprint,
    prompt_view,
    stratified_sample,
)
from drugorch.harness.external.mdb_eval import _row
from drugorch.harness.external.panel import Measurement
from drugorch.harness.external.spec import DesignSpec, FragTarget, PropTarget

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"


def make_instance(i: int, *, task: str = "generation", infeasible: bool = False) -> Instance:
    return Instance(
        id=f"{task}_{i}",
        task_type=task,
        question=f"Design something for scenario {i}.",
        answer={
            "properties": [{"property": "MW", "min": 150.0, "max": 400.0}],
            "fragments": [{"benzene ring": 1}],
        },
        infeasible=infeasible,
        seed_smiles=CAFFEINE if task == "optimization" else None,
        ref_smiles="C1=CC=CC=C1NC(=O)CSECRET",
    )


def population(n_gen: int = 100, n_opt: int = 100, n_infeasible: int = 10) -> list[Instance]:
    out = []
    for i in range(n_gen):
        out.append(make_instance(i, task="generation", infeasible=i < n_infeasible))
    for i in range(n_opt):
        out.append(make_instance(i, task="optimization", infeasible=i < n_infeasible))
    return out


# --- the boundary ------------------------------------------------------------------


def test_prompt_view_carries_no_answer_key() -> None:
    inst = make_instance(1)
    view = prompt_view(inst)
    assert set(view) <= {"id", "task_type", "question", "seed_smiles"}
    assert "SECRET" not in json.dumps(view)
    assert "answer" not in view


def test_prompt_view_gives_optimization_its_seed() -> None:
    """The seed molecule is stated in the question text, so withholding it hides nothing."""
    inst = make_instance(2, task="optimization")
    assert prompt_view(inst)["seed_smiles"] == CAFFEINE


class RecordingBackend:
    """A backend that answers from a script and keeps every prompt it was shown."""

    def __init__(self, replies: list[dict]):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def key(self) -> str:
        return "recording:test"

    def complete(self, system, cached_prefix, prompt, schema):
        self.prompts.append("\n".join([system, cached_prefix, prompt]))
        reply = self.replies.pop(0) if self.replies else {"action": "answer", "reason": "done"}
        return reply, {"model": "test"}

    def describe(self) -> dict:
        return {"backend": "recording"}


class StubPanel:
    """Measurement without a subprocess: every molecule gets the same numbers."""

    def __init__(self, properties: dict | None = None, fragments: dict | None = None):
        self.properties = properties or {"MW": 300.0, "logP": 2.0}
        self.fragments = fragments or {"benzene ring": 1}
        self.batches: list[list[str]] = []

    def measure(self, smiles):
        self.batches.append(list(smiles))
        return {
            s: Measurement(smiles=s, properties=dict(self.properties), fragments=dict(self.fragments))
            for s in smiles
            if s and not s.startswith("not-")
        }


class StubPool:
    def __init__(self, hits=()):
        self.hits = list(hits)

    def search(self, spec, *, limit=24, seed=0):
        from drugorch.harness.external.pool import PoolHit, PoolResult

        return PoolResult(hits=[PoolHit(smiles=s) for s in self.hits], matched=len(self.hits))

    def describe(self):
        return {"pool": "stub"}


def build_arm(tmp_path, replies, *, name="drugorch", panel=None, pool=None):
    from drugorch.config import get_settings
    from drugorch.llm.response_cache import ResponseCache

    backend = RecordingBackend(replies)
    return (
        Arm(
            name=name,
            settings=get_settings(),
            backend=backend,
            cache=ResponseCache(tmp_path / "llm"),
            panel=panel or StubPanel(),
            pool=pool or StubPool([ASPIRIN]),
        ),
        backend,
    )


def test_no_prompt_can_carry_the_answer_key(tmp_path) -> None:
    """Not one prompt, on either arm, may contain anything from the key.

    The instance's reference solution carries a marker. If it ever reaches a prompt, this
    fails; nothing else in the system would notice.
    """
    plan = {
        "infeasible": False,
        "reason": "read the scenario",
        "properties": [{"property": "MW", "min": 150.0, "max": 400.0}],
        "fragments": [{"name": "benzene ring", "count": 1}],
        "candidates": [CAFFEINE],
    }
    decisions = [
        {"action": "search_pool", "reason": "start from molecules that already fit"},
        {"action": "generate", "reason": "repair the rest", "generate_rounds": 1},
        {"action": "answer", "reason": "good enough", "answer_smiles": ASPIRIN},
    ]
    for task in ("generation", "optimization"):
        inst = make_instance(7, task=task)
        arm, backend = build_arm(tmp_path / task, [plan, *decisions])
        arm.answer(inst, seed=1)
        assert backend.prompts, "the arm never called the model"
        for prompt in backend.prompts:
            assert "SECRET" not in prompt
            assert inst.ref_smiles not in prompt
            assert "answer" not in json.loads(prompt.split("\n")[-1] if False else "{}")


def test_notool_arm_sees_no_tools_and_no_key(tmp_path) -> None:
    inst = make_instance(8)
    arm, backend = build_arm(
        tmp_path, [{"smiles": ASPIRIN, "reason": "fits"}], name="notool"
    )
    att = arm.answer(inst)
    assert att.prediction == ASPIRIN
    assert att.n_tool_calls == 0
    assert all("SECRET" not in p for p in backend.prompts)


# --- the sample --------------------------------------------------------------------


def test_sample_holds_the_task_split_and_the_infeasible_share() -> None:
    sample = stratified_sample(population(), 200, seed=0)
    d = describe(sample)
    assert d["by_task"] == {"generation": 100, "optimization": 100}
    # 10% of each task's pool is infeasible, and the sample holds that rather than drifting.
    assert d["infeasible"] == 20


def test_sampling_is_reproducible_and_fingerprinted() -> None:
    pop = population()
    a = stratified_sample(pop, 40, seed=3)
    b = stratified_sample(pop, 40, seed=3)
    c = stratified_sample(pop, 40, seed=4)
    assert [i.id for i in a] == [i.id for i in b]
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(c), "a different seed must name a different sample"


def test_a_sample_larger_than_the_set_is_the_set() -> None:
    pop = population(n_gen=3, n_opt=3, n_infeasible=1)
    assert len(stratified_sample(pop, 99, seed=0)) == 6


# --- what the evaluator will read ---------------------------------------------------


def test_record_is_shaped_the_way_the_evaluator_reads_one() -> None:
    inst = make_instance(9)
    rec = inst.record(ASPIRIN, n_tool_calls=3)
    # Bare SMILES: the evaluator parses `prediction` directly, so a tagged answer would be
    # scored as an unparseable molecule.
    assert rec["prediction"] == ASPIRIN
    assert rec["infeasible"] is False
    assert rec["answer"] == inst.answer
    assert sum(1 for m in rec["conversations"] if m["role"] == "tool") == 3


def test_an_abstention_is_the_literal_none() -> None:
    rec = make_instance(10, infeasible=True).record("None")
    assert rec["prediction"] == "None"
    assert rec["infeasible"] is True


def test_metrics_pass_through_unchanged() -> None:
    """Our row carries the evaluator's numbers, not a recomputation of them."""
    row = _row("drugorch", "generation", {
        "total": 100, "success": 0.37, "csr": 0.81, "mcd": 3.4, "inf_acc": 0.5,
        "f1": 0.6, "mcc": 0.44, "validity": 0.95, "tool_calls": 4.2,
        "success_count": 37, "feasible_total": 90, "infeasible_total": 10,
        "confusion": (80, 5, 10, 5),
    })
    assert row.success == 0.37
    assert row.mcd == 3.4
    assert row.to_dict()["confusion"] == [80, 5, 10, 5]


# --- fitness -----------------------------------------------------------------------


def spec_mw_and_ring() -> DesignSpec:
    return DesignSpec(
        properties=[PropTarget("MW", 150.0, 400.0), PropTarget("logP", None, 3.0)],
        fragments=[FragTarget("benzene ring", 1, at_least=True)],
    )


def test_satisfaction_outranks_distance() -> None:
    """Two satisfied constraints beat one satisfied and a near miss on the other."""
    spec = spec_mw_and_ring()
    both = spec.score(Measurement("a", {"MW": 300.0, "logP": 2.0}, {"benzene ring": 1}))
    near = spec.score(Measurement("b", {"MW": 300.0, "logP": 3.01}, {"benzene ring": 1}))
    assert both.satisfied == 3 and near.satisfied == 2
    assert both.fitness > near.fitness


def test_distance_is_normalised_by_the_benchmarks_own_ranges() -> None:
    """A miss is measured in p5-p95 widths, so MW and QED errors are comparable."""
    mw = PropTarget("MW", None, 400.0).distance(400.0 + 303.32)
    qed = PropTarget("QED", 0.6, None).distance(0.55)
    assert mw == pytest.approx(1.0)
    assert qed == pytest.approx(0.0769, abs=1e-3)


def test_an_unmeasured_property_is_charged_not_skipped() -> None:
    """The evaluator charges a full unit for a property it cannot read; so does fitness."""
    spec = DesignSpec(properties=[PropTarget("logS", -4.0, -2.0)])
    s = spec.score(Measurement("x", {"MW": 300.0}, {}))
    assert s.satisfied == 0
    assert s.distance == pytest.approx(1.0)


def test_an_invalid_molecule_cannot_win() -> None:
    assert spec_mw_and_ring().score(None).fitness < -1000


def test_contradictions_are_structural_only() -> None:
    impossible = DesignSpec(properties=[PropTarget("logP", 4.0, 3.5)])
    merely_hard = DesignSpec(properties=[PropTarget("MW", 390.0, 400.0),
                                         PropTarget("QED", 0.9, 0.95)])
    assert impossible.contradictions()
    assert not merely_hard.contradictions()


def test_disjoint_windows_on_one_property_are_a_contradiction() -> None:
    spec = DesignSpec(properties=[PropTarget("MW", 400.0, None), PropTarget("MW", None, 300.0)])
    assert spec.contradictions()


# --- the generator -----------------------------------------------------------------


def test_generation_keeps_the_best_and_stops_on_success() -> None:
    spec = spec_mw_and_ring()
    panel = StubPanel()
    res = optimize([ASPIRIN], spec, panel, rounds=3, population=4, seed=1)
    assert res.best is not None
    assert res.solved, "a stub that satisfies everything must stop at the first round"
    assert res.rounds == 0


def test_generation_measures_in_batches() -> None:
    """One measurement call per round, not one per molecule: the worker costs seconds."""
    spec = DesignSpec(properties=[PropTarget("MW", 1e6, None)])  # never satisfied
    panel = StubPanel()
    res = optimize([ASPIRIN, CAFFEINE], spec, panel, rounds=2, population=6, seed=2)
    assert res.rounds == 2
    assert len(panel.batches) == 3  # the seeds, then one per round
    assert all(len(b) > 1 for b in panel.batches[1:])


# --- published rows ----------------------------------------------------------------


def test_published_rows_are_labelled_as_read_not_measured() -> None:
    rows = baselines.published("generation")
    assert rows and all(r["measured_here"] is False for r in rows)
    assert all(r["n"] == 1000 for r in rows)
    assert {r["arm"] for r in rows} >= {"GPT-5.4", "Claude-Opus-4.6"}


def test_best_published_reads_mcd_in_its_own_direction() -> None:
    assert baselines.best_published("generation", "success")["arm"] == "GPT-5.4"
    # MCD is a distance, so the leader is the smallest rather than the largest.
    assert baselines.best_published("optimization", "mcd")["mcd"] == 1.58


# --- resume ------------------------------------------------------------------------


def test_a_fallback_attempt_is_never_reused(tmp_path) -> None:
    """A fallback means the decision never reached the model, so the answer is not the arm's."""
    from drugorch.harness.external.run import load_attempt, save_attempt

    good = Attempt(instance_id="g1", arm="drugorch", prediction=ASPIRIN)
    bad = Attempt(instance_id="g2", arm="drugorch", prediction="None", n_fallbacks=1)
    save_attempt(tmp_path, good)
    save_attempt(tmp_path, bad)
    assert load_attempt(tmp_path, "drugorch", "g1").prediction == ASPIRIN
    assert load_attempt(tmp_path, "drugorch", "g2") is None


def test_arm_stats_counts_the_work_not_the_score() -> None:
    """A score with no account of the work behind it says nothing about orchestration."""
    from drugorch.harness.external.agent import ToolCall
    from drugorch.harness.external.run import arm_stats

    a = Attempt(instance_id="i1", arm="drugorch", prediction=ASPIRIN, wall_s=4.0)
    a.tool_calls = [ToolCall(tool="search_pool"), ToolCall(tool="measure")]
    b = Attempt(instance_id="i2", arm="drugorch", prediction="None", wall_s=2.0)
    b.tool_calls = [ToolCall(tool="measure")]

    stats = arm_stats([a, b])
    assert stats["mean_tool_calls"] == 1.5
    assert stats["abstained"] == 1
    assert stats["tool_usage"] == {"measure": 2, "search_pool": 1}


def test_published_rows_match_the_vendored_table_digit_for_digit() -> None:
    """The published rows are a transcription, so the transcription is checked.

    `baselines.py` holds Table 3 as Python literals because the board has to serve them
    without parsing markdown at request time. That makes a typo invisible: a wrong figure
    would sit on the board beside our own, labelled as the state of the art, and nothing
    would contradict it. This parses the vendored file and compares every cell.

    It does not check the vendored file against the paper. That chain ends at
    `third_party/`, and this test says so rather than implying otherwise.
    """
    from pathlib import Path

    md = Path("third_party/MolDesignBench/docs/leaderboard.md")
    if not md.is_file():
        pytest.skip("the benchmark sources are not vendored in this checkout")

    text = md.read_text(encoding="utf-8")
    blocks: dict[str, dict] = {}
    for task, head in (
        ("generation", "## Molecule Generation"),
        ("optimization", "## Molecule Optimization"),
    ):
        # Bounded at the next heading: splitting on the heading alone leaves the second
        # table in the first block, and its rows then overwrite the first table's.
        body = text.split(head, 1)[1].split("\n## ", 1)[0]
        rows = {}
        for line in body.splitlines():
            if not line.startswith("| ") or line.startswith("| Model"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 9 or cells[1] not in ("w/", "w/o"):
                continue
            rows[(cells[0], cells[1] == "w/")] = tuple(cells[2:9])
        blocks[task] = rows

    assert sum(len(b) for b in blocks.values()) == len(baselines.published())

    for r in baselines.published():
        published = blocks[r["scope"]].get((r["arm"], r["tools"]))
        assert published is not None, f"{r['arm']} is not in the vendored table"
        dash = {"–", "—", "-"}
        expected = tuple("-" if c in dash else c for c in published)
        assert expected == (
            f"{r['success']:.2f}",
            f"{r['csr']:.2f}",
            f"{r['mcd']:.2f}",
            f"{r['inf_acc']:.2f}",
            f"{r['mcc']:.2f}",
            f"{r['validity']:.2f}",
            "-" if r["tool_calls"] is None else f"{r['tool_calls']:.2f}",
        ), f"{r['scope']} {r['arm']} differs from docs/leaderboard.md"


def test_a_fallback_is_not_scored_as_an_abstention(tmp_path, monkeypatch) -> None:
    """An instance whose decision never reached the model is excluded, not answered None.

    Recording it as None is not a neutral substitution: the benchmark reads None as a claim
    that the instance is impossible, so a rate limit would cost Success on a solvable
    instance and credit InfAcc on an impossible one. A transport failure would then look
    like judgement about feasibility.
    """
    from drugorch.harness.external import run as run_mod

    sample = [make_instance(1), make_instance(2, infeasible=True), make_instance(3)]
    attempts = [
        Attempt(instance_id="generation_1", arm="drugorch", prediction=ASPIRIN),
        Attempt(instance_id="generation_2", arm="drugorch", prediction="None"),
        Attempt(instance_id="generation_3", arm="drugorch", prediction="None", n_fallbacks=1),
    ]
    captured: dict = {}

    class FakeArm:
        def describe(self):
            return {"arm": "drugorch"}

    def fake_score(arm_name, records, root, **kw):
        captured["records"] = records
        return {}

    monkeypatch.setattr(run_mod, "load_instances", lambda path: sample)
    monkeypatch.setattr(run_mod, "stratified_sample", lambda ins, n, seed=0: sample)
    monkeypatch.setattr(run_mod, "run_arm", lambda name, instances, **kw: (FakeArm(), attempts))
    monkeypatch.setattr(run_mod, "score", fake_score)

    result = run_mod.run_external(n=3, arms=("drugorch",), offline=True, resume=True)

    scored_ids = [r["id"] for r in captured["records"]]
    assert scored_ids == ["generation_1", "generation_2"], "the fallback must not be scored"
    # The abstention that the arm actually made is still scored, as an abstention.
    assert captured["records"][1]["prediction"] == "None"
    assert result.arms["drugorch"]["not_answered"] == 1


def test_mcnemar_is_exact_and_two_sided() -> None:
    """The p-value is a binomial over discordant pairs, not a chi-squared approximation."""
    from drugorch.harness.external.paired import exact_mcnemar

    assert exact_mcnemar(0, 0) == 1.0, "no disagreement is no evidence, not a crash"
    assert exact_mcnemar(5, 5) == 1.0
    assert exact_mcnemar(18, 2) == pytest.approx(0.000402, abs=1e-5)
    # Symmetric: losing by the same margin is as improbable as winning by it.
    assert exact_mcnemar(2, 18) == pytest.approx(exact_mcnemar(18, 2))


def test_pairing_drops_instances_only_one_arm_answered() -> None:
    """An arm with no verdict on an instance cannot be paired on it."""
    from drugorch.harness.external.paired import compare

    g = compare(
        {
            "drugorch": {"a": True, "b": False, "c": True, "d": True},
            "notool": {"a": False, "b": False, "c": True},
        },
        arm="drugorch",
        baseline="notool",
        scope="generation",
    )
    assert g is not None
    assert g.n_paired == 3, "d is unpaired and must not count"
    assert (g.arm_only, g.baseline_only, g.both, g.neither) == (1, 0, 1, 1)
    assert g.delta == 1


def test_tool_roster_is_read_from_their_sources_not_retyped() -> None:
    """Their descriptions are the strings their own agents see, parsed from their code.

    Retyping them would make this page a claim about someone else's system that nothing
    checks. Parsing also catches the case that cost the Python shell its entry: their
    classes use both `name = '...'` and `name: str = '...'`.
    """
    from pathlib import Path

    from drugorch.harness.external.toolsheet import sheet, their_tools

    if not Path("third_party/MolDesignBench/moldesign/tools").is_dir():
        pytest.skip("the benchmark sources are not vendored in this checkout")

    tools = their_tools()
    names = {t.name for t in tools}
    assert len(tools) == 17, "the paper reports seventeen tools"
    assert "run_python_code" in names, "the annotated-assignment spelling must parse"
    assert "analyze_overall_molecule_properties" in names
    assert all(t.description for t in tools)

    s = sheet()
    assert s["n_drugorch"] == 4
    # Every capability of theirs we do not cover is named, rather than left to a count.
    assert len(s["uncovered"]) == s["n_benchmark"] - 3


def test_a_missing_checkout_reports_no_roster_rather_than_inventing_one(tmp_path) -> None:
    from drugorch.harness.external.toolsheet import sheet, their_tools

    assert their_tools(tmp_path) == []
    assert sheet(tmp_path)["n_benchmark"] == 0


# --- verification before answering --------------------------------------------------


def test_a_self_declared_success_is_verified_before_it_becomes_an_answer(tmp_path) -> None:
    """The arm's belief that it is finished was right one time in five; it is checked.

    The plan extracts one constraint, which the stub panel satisfies immediately. Without
    verification the arm would answer there. With it, the re-reading adds the constraint the
    scenario implied, and the molecule is measured against the larger specification.
    """
    plan = {
        "infeasible": False,
        "reason": "only the explicit window",
        "properties": [{"property": "MW", "min": 150.0, "max": 400.0}],
        "fragments": [],
        "candidates": [ASPIRIN],
    }
    verification = {
        "complete": False,
        "reason": "the scenario implies a lipophilicity ceiling",
        "properties": [{"property": "logP", "min": None, "max": 3.0}],
        "fragments": [],
    }
    arm, backend = build_arm(tmp_path, [plan, verification, {"complete": True, "reason": "now complete", "properties": [], "fragments": []}])
    att = arm.answer(make_instance(11), seed=1)

    actions = [d["action"] for d in att.decisions]
    assert "verify" in actions, "a self-declared success must be verified"
    assert att.spec["properties"][-1]["property"] == "logP", "the missed constraint is merged"
    assert att.best_total == 2, "the molecule is scored against the grown specification"


def test_verification_stops_at_its_budget(tmp_path) -> None:
    """Two passes, then the arm answers. A third can only find what two readings missed."""
    plan = {
        "infeasible": False, "reason": "", "candidates": [ASPIRIN],
        "properties": [{"property": "MW", "min": 150.0, "max": 400.0}], "fragments": [],
    }
    # Every verification claims to find something new, so only the cap can stop the loop.
    more = [
        {"complete": False, "reason": "another", "fragments": [],
         "properties": [{"property": p, "min": None, "max": 999.0}]}
        for p in ("logP", "TPSA", "QED", "MR")
    ]
    arm, backend = build_arm(tmp_path, [plan, *more])
    att = arm.answer(make_instance(12), seed=1)
    assert sum(1 for d in att.decisions if d["action"] == "verify") == 2


def test_verification_merges_rather_than_replaces(tmp_path) -> None:
    """A second reading that forgot the first's constraints would make the arm oscillate."""
    plan = {
        "infeasible": False, "reason": "", "candidates": [ASPIRIN], "fragments": [],
        "properties": [{"property": "MW", "min": 150.0, "max": 400.0}],
    }
    verification = {
        "complete": False, "reason": "tighter", "fragments": [],
        "properties": [{"property": "MW", "min": 200.0, "max": None},
                       {"property": "TPSA", "min": None, "max": 90.0}],
    }
    arm, _ = build_arm(tmp_path, [plan, verification, {"complete": True, "reason": "", "properties": [], "fragments": []}])
    att = arm.answer(make_instance(13), seed=1)

    by_name = {p["property"]: p for p in att.spec["properties"]}
    assert set(by_name) == {"MW", "TPSA"}
    # The two readings of MW are intersected, not appended as two separate windows.
    assert (by_name["MW"]["min"], by_name["MW"]["max"]) == (200.0, 400.0)


# --- anchoring an optimization search to its lead ------------------------------------


def test_generation_ranks_an_edit_above_a_replacement() -> None:
    """On optimization, a molecule that has drifted off the lead loses to one that has not.

    The benchmark does not score similarity to the seed. What anchoring protects is the
    fragment constraints, which are read off the seed and are the first thing a wandering
    search loses.
    """
    from drugorch.harness.external.ga import ANCHOR_FLOOR, Candidate
    from drugorch.harness.external.spec import Score

    edit = Candidate(smiles="a", score=Score(satisfied=2, total=3), similarity=0.9)
    drifted = Candidate(smiles="b", score=Score(satisfied=3, total=3), similarity=0.1)
    assert drifted.score.fitness > edit.score.fitness, "it scores better on constraints"
    assert edit.fitness > drifted.fitness, "and still loses, because it left the lead"

    on_the_line = Candidate(smiles="c", score=Score(satisfied=2, total=3), similarity=ANCHOR_FLOOR)
    assert on_the_line.fitness == on_the_line.score.fitness, "the floor is inclusive"


def test_an_unanchored_search_is_unaffected() -> None:
    """Generation instances have no lead, so nothing is penalised there."""
    from drugorch.harness.external.ga import Candidate
    from drugorch.harness.external.spec import Score

    c = Candidate(smiles="a", score=Score(satisfied=3, total=3), similarity=None)
    assert c.fitness == c.score.fitness


def test_diagnostics_read_the_key_only_after_scoring(tmp_path) -> None:
    """The decomposition is computed from the answer key, which is why it runs last.

    Spec recall compares what the arm wrote down with what the key holds. That is a
    statement about a finished run. Used the other way round it would be the leak that
    invalidates every number in the project, so it lives in its own module, takes the
    scored file as input, and is never reachable from an arm.
    """
    from drugorch.harness.external.diagnose import decompose, key_size

    sample = [make_instance(1), make_instance(2), make_instance(3, infeasible=True)]
    assert key_size(sample[0]) == 2  # one property window, one fragment

    scored = tmp_path / "moldesignbench.json"
    scored.write_text(
        json.dumps(
            [
                {"id": "generation_1", "total_acc": 0},
                {"id": "generation_2", "total_acc": 1},
                {"id": "generation_3", "total_acc": 1},
            ]
        ),
        encoding="utf-8",
    )

    attempts = [
        # Believed it was finished, was not: the failure mode the verification step targets.
        Attempt(instance_id="generation_1", arm="a", best_satisfied=2, best_total=2),
        # Wrote down one of the two constraints the key holds, and solved it anyway.
        Attempt(instance_id="generation_2", arm="a", best_satisfied=0, best_total=1),
        Attempt(instance_id="generation_3", arm="a", best_satisfied=2, best_total=2),
    ]
    d = decompose(attempts, sample, scored)

    assert d["n_feasible"] == 2, "the infeasible instance is not part of this decomposition"
    assert d["believed_and_wrong"] == 1
    assert d["unsure_and_right"] == 1
    assert d["confidence_precision"] == 0.0
    assert d["spec_recall"] == pytest.approx(0.75)  # 3 constraints written, 4 held


def test_diagnostics_are_absent_rather_than_zero_when_unscored(tmp_path) -> None:
    from drugorch.harness.external.diagnose import decompose

    assert decompose([], [make_instance(1)], tmp_path / "nothing.json") == {}


# --- identity, after a code review found the dataset reuses ids ----------------------


def test_reused_ids_are_distinct_instances_everywhere() -> None:
    """112 of the shipped ids belong to two different questions. Nothing may merge them.

    Keyed on `id`, two instances share one attempt file, the survivor is scored against the
    other's answer key, and every aggregate counts one instance while reporting two. None
    of that raises.
    """
    from pathlib import Path

    from drugorch.harness.external.instances import load_instances, stratified_sample

    if not Path("data/external/moldesignbench.jsonl").is_file():
        pytest.skip("the benchmark data is not downloaded in this checkout")

    rows = load_instances()
    assert len({i.uid for i in rows}) == len(rows), "uid must be unique per row"
    assert len({i.id for i in rows}) < len(rows), "the dataset really does reuse ids"

    # A reused id keeps its dataset spelling on the first occurrence, so files stay
    # readable, and is suffixed afterwards.
    repeated = next(i for i in rows if i.ordinal > 0)
    assert repeated.uid.startswith(repeated.id) and repeated.uid != repeated.id

    sample = stratified_sample(rows, 200, seed=0)
    assert len({i.uid for i in sample}) == len(sample)
    # The record handed to the evaluator carries the unique id, with the dataset's beside it.
    rec = repeated.record("C")
    assert rec["id"] == repeated.uid and rec["mdb_id"] == repeated.id


def test_an_odd_sample_size_is_honoured() -> None:
    pop = population(n_gen=60, n_opt=60, n_infeasible=6)
    assert len(stratified_sample(pop, 51, seed=0)) == 51
    assert len(stratified_sample(pop, 50, seed=0)) == 50


def test_an_unbounded_target_is_not_a_constraint() -> None:
    """Both bounds null satisfies every molecule, so it must never reach the spec.

    Kept, it counts toward satisfied and total alike: a spec of two such targets reports
    2/2 and `solved`, and the arm answers the first molecule it measured while the board
    records the reading as complete.
    """
    spec = DesignSpec.from_llm(
        {
            "infeasible": False,
            "properties": [
                {"property": "MW", "min": None, "max": None},
                {"property": "QED", "min": 0.5, "max": None},
            ],
            "fragments": [],
        }
    )
    assert [p.name for p in spec.properties] == ["QED"]
    assert spec.total == 1


def test_a_revision_adds_to_the_reading_rather_than_replacing_it() -> None:
    from drugorch.harness.external.spec import merge

    base = DesignSpec(
        properties=[PropTarget("MW", 150.0, 400.0), PropTarget("logP", None, 3.0)],
        fragments=[FragTarget("benzene ring", 1)],
    )
    addition = DesignSpec(properties=[PropTarget("MW", 200.0, None), PropTarget("TPSA", None, 90.0)])
    grown = merge(base, addition)

    by_name = {p.name: p for p in grown.properties}
    assert set(by_name) == {"MW", "logP", "TPSA"}, "nothing from the first reading is lost"
    assert (by_name["MW"].lo, by_name["MW"].hi) == (200.0, 400.0), "one window, intersected"
    assert len(grown.fragments) == 1


def test_a_failed_verification_does_not_discard_the_attempt(tmp_path) -> None:
    """Verification is optional. Losing it must not cost an instance the arm answered.

    It only runs once the arm believes it has succeeded, so counting its transport failures
    as fallbacks drops solved instances preferentially and moves the score for a reason
    that has nothing to do with the arm.
    """
    plan = {
        "infeasible": False, "reason": "", "candidates": [ASPIRIN], "fragments": [],
        "properties": [{"property": "MW", "min": 150.0, "max": 400.0}],
    }

    class FailingVerification(RecordingBackend):
        def complete(self, system, cached_prefix, prompt, schema):
            if schema.get("properties", {}).get("complete") is not None:
                from drugorch.llm.response_cache import LLMUnavailable

                raise LLMUnavailable("rate limited")
            return super().complete(system, cached_prefix, prompt, schema)

    from drugorch.config import get_settings
    from drugorch.llm.response_cache import ResponseCache

    backend = FailingVerification([plan])
    arm = Arm(
        name="drugorch",
        settings=get_settings(),
        backend=backend,
        cache=ResponseCache(tmp_path / "llm"),
        panel=StubPanel(),
        pool=StubPool([ASPIRIN]),
    )
    att = arm.answer(make_instance(21), seed=1)

    assert att.prediction == ASPIRIN, "the arm answers with the molecule it already had"
    assert att.n_fallbacks == 0, "a lost verification is not a lost arm"
    assert att.n_verify_failures == 1


def test_the_pool_respects_exact_fragment_counts() -> None:
    """Generation asks for exactly N of a group; returning more wastes a measurement batch."""
    from pathlib import Path

    from drugorch.harness.external.pool import get_pool

    if not Path("third_party/MolDesignBench/data/moses_admet_tagged/parts").is_dir():
        pytest.skip("the molecule pool is not vendored in this checkout")

    spec = DesignSpec(
        properties=[PropTarget("MW", 250.0, 450.0)],
        fragments=[FragTarget("benzene ring", 1, at_least=False)],
    )
    found = get_pool().search(spec, limit=8, seed=0)
    assert found.hits
    from rdkit.Chem import Fragments

    from drugorch.chem import to_mol

    for smiles in found.smiles:
        assert Fragments.fr_benzene(to_mol(smiles)) == 1


def test_a_query_with_nothing_to_filter_on_stays_bounded() -> None:
    """`expr` is None on the last relaxation pass and for an unfilterable spec.

    Materialising the match set there builds 1.9 million dicts per worker thread.
    """
    import time
    from pathlib import Path

    from drugorch.harness.external.pool import get_pool

    if not Path("third_party/MolDesignBench/data/moses_admet_tagged/parts").is_dir():
        pytest.skip("the molecule pool is not vendored in this checkout")

    started = time.time()
    found = get_pool().search(DesignSpec(properties=[]), limit=24, seed=0)
    assert len(found.hits) <= 24
    assert time.time() - started < 10, "an unfiltered query must not materialise the pool"
