"""What a call reports about itself, and what the operator controls before one happens.

Three things are under test here, and they share a theme: a number with no account of how it
was reached is not evidence.

* **The safety gate** (:mod:`drugorch.subsystems.similarity`) decides what catalogued
  chemistry is allowed into a run. The declared rule table has to be what actually runs, in
  every mode -- a gate documented one way and implemented another is worse than no gate.
* **The property panel** has to say which properties it looked at, not only how many
  molecules it looked at.
* **Tool selection** has to remove a subsystem from the policy's schema, not merely mark it
  unavailable. Those differ: an unavailable subsystem can still be named by a model, fail
  validation, and silently degrade the step to the fallback policy.

Nothing here touches the network. The bank search is driven through an injected client.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from drugorch.clients.neighbour import Neighbour
from drugorch.llm.response_cache import request_key
from drugorch.policy import build_policy
from drugorch.policy.llm import SYSTEM, build_schema, compose_system, prompt_kind
from drugorch.subsystems import similarity as sim
from drugorch.subsystems.properties import PropertiesSubsystem
from drugorch.subsystems.registry import DEFAULT_TOOLS, build_registry

# --- doubles -------------------------------------------------------------------

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"


class FakeBank:
    """Stands in for ChEMBL. Returns a fixed neighbour set regardless of the query.

    The subsystem recomputes similarity locally, so what this returns is only a list of
    structures to be gated -- which is exactly the part under test.
    """

    def __init__(self, neighbours):
        self.neighbours = neighbours
        self.n_requests = 0
        self.n_cache_hits = 0
        self.queries: list[str] = []

    def similar(self, _client, smiles, *, threshold=0.7, limit=25):
        self.queries.append(smiles)
        self.n_requests += 1
        return list(self.neighbours)

    def describe(self):
        return {"fake": True}


@pytest.fixture
def banked(monkeypatch):
    """A similarity subsystem wired to a fake bank. Returns ``(subsystem, bank)``."""

    def _make(neighbours):
        bank = FakeBank(neighbours)
        monkeypatch.setattr(sim.chembl, "similar", bank.similar)
        sub = sim.SimilaritySubsystem(client=bank)
        return sub, bank

    return _make


def _run(sub, state, **params):
    merged = sub.capability.coerce_params({"source": "chembl", **params})
    return sub.run(merged, state, cancel=threading.Event(), progress=lambda *_: None)


def _generated(state, *smiles):
    state.add_records([{"smiles": s} for s in smiles], origin="generation")


# --- the safety gate -----------------------------------------------------------


def test_withdrawn_compound_is_refused_and_counted(banked, state):
    """The strongest negative signal either bank carries must not enter a run quietly."""
    sub, _ = banked([Neighbour(ASPIRIN, "chembl", "CHEMBL25", withdrawn=True)])
    # Caffeine, not aspirin: a query molecule is in the run by definition, so querying with
    # the same structure the bank returns would trip the duplicate check first.
    _generated(state, CAFFEINE)

    res = _run(sub, state, query_from="explicit", smiles=[ASPIRIN], safety="standard")

    assert res.ok
    assert res.explain["safety"]["rejected"]["withdrawn"] == 1
    assert not res.explain["kept"]
    assert any(x["rule"] == "withdrawn" for x in res.explain["safety"]["examples"])


def test_each_mode_does_what_the_rule_table_declares(banked, state):
    """The declared verdict and the observed behaviour are the same thing in all 3 modes.

    This is the assertion that keeps :data:`similarity.SAFETY_RULES` honest: the table is
    rendered to the operator as the description of the gate, so a mode whose behaviour
    drifts from its row is a lie told in the UI.
    """
    black_box = Neighbour(ASPIRIN, "chembl", "CHEMBL25", black_box=True)

    for mode, expect_kept in (("off", True), ("standard", True), ("strict", False)):
        sub, _ = banked([black_box])
        _generated(state, CAFFEINE)
        res = _run(sub, state, query_from="explicit", smiles=[ASPIRIN], safety=mode)

        declared = sim.RULES_BY_KEY["black_box"].verdict(mode)
        kept = res.explain["kept"]
        assert bool(kept) is expect_kept, f"{mode}: {declared} did not match behaviour"
        if declared == "reject":
            assert res.explain["safety"]["rejected"]["black_box"] == 1
        elif declared == "flag":
            assert kept[0]["flags"] == ["black_box"]
        else:
            assert kept[0]["flags"] == []


def test_structural_rules_never_relax(banked, state):
    """``safety='off'`` still refuses what cannot be docked or profiled.

    'Off' means "do not apply medicinal-chemistry judgement", not "accept anything". A
    metal complex or a fragment entering the run would fail later in the pipeline and read
    as a bug in docking rather than as a molecule that should never have been admitted.
    """
    sub, _ = banked(
        [
            Neighbour("[Pt](Cl)(Cl)(N)N", "chembl", "CHEMBL2068237"),  # metal
            Neighbour("CCO", "chembl", "CHEMBL545"),  # too small
        ]
    )
    _generated(state, CAFFEINE)

    res = _run(sub, state, query_from="explicit", smiles=[ASPIRIN], safety="off")

    rejected = res.explain["safety"]["rejected"]
    assert not res.explain["kept"]
    # Named by their actual defect, not lumped under "below threshold" -- neither would
    # fingerprint usefully, so the ordering of the gate is what makes this diagnosis right.
    assert rejected.get("element") == 1
    assert rejected.get("size") == 1


def test_similarity_is_recomputed_not_quoted(banked, state):
    """A bank claiming a match does not get to decide what the run is filtered on.

    ChEMBL reports its own similarity and PubChem reports none, so a threshold applied to
    "whatever the bank said" would mean different things per source. The real case this
    catches is a salt: ChEMBL matches the active component and keeping the largest fragment
    can leave the counter-ion, which is not what matched.
    """
    unrelated = Neighbour(
        "CCCCCCCCCCCCCCCC(=O)O", "chembl", "CHEMBL82293", bank_similarity=0.99
    )
    sub, _ = banked([unrelated])
    _generated(state, ASPIRIN)

    res = _run(sub, state, query_from="explicit", smiles=[ASPIRIN], min_similarity=0.7)

    assert res.explain["safety"]["rejected"]["threshold"] == 1
    assert not res.explain["kept"]


def test_molecules_already_in_the_run_are_not_readmitted(banked, state):
    sub, _ = banked([Neighbour(CAFFEINE, "chembl", "CHEMBL113")])
    _generated(state, CAFFEINE)

    res = _run(sub, state, query_from="explicit", smiles=[CAFFEINE])

    assert res.explain["safety"]["rejected"]["duplicate"] == 1


# --- not searching the same molecules forever ----------------------------------


def test_a_fruitless_search_is_still_recorded_as_done(banked, state):
    """The regression this marker exists for.

    A search that returns nothing used to leave its query molecule indistinguishable from
    one never searched, so the next call picked the same molecules, found nothing again,
    and only the orchestrator's stuck-detector ended it several wasted calls later.
    """
    sub, bank = banked([])  # the bank knows nothing about anything
    _generated(state, ASPIRIN, CAFFEINE)

    first = _run(sub, state, query_from="generated", max_queries=2)
    assert not first.explain["kept"]
    state.add_records(first.records, origin=sub.capability.key)

    usable, why = sub.preflight(
        sub.capability.coerce_params({"query_from": "generated"}), state
    )
    assert usable is False
    assert "already been searched" in why

    second_queries, _ = sub._queries({"query_from": "generated", "max_queries": 2}, state)
    assert second_queries == []
    assert len(bank.queries) == 2  # and nothing was asked of the bank a second time


def test_preflight_distinguishes_nothing_yet_from_nothing_left(banked, state):
    """Two blocked calls that need different responses from the policy."""
    sub, _ = banked([])
    params = sub.capability.coerce_params({"query_from": "generated"})

    usable, why = sub.preflight(params, state)
    assert usable is False
    assert "run generation first" in why

    _generated(state, ASPIRIN)
    state.add_records(
        [{"smiles": ASPIRIN, "props": {sim.SEARCHED_MARKER: sub.capability.version}}],
        origin="similarity",
    )
    usable, why = sub.preflight(params, state)
    assert usable is False
    assert "already been searched" in why


def test_urgency_falls_off_once_searching_stops_paying(banked, state):
    """A generator that has left catalogued chemistry should stop attracting searches."""
    sub, _ = banked([])
    many = [f"{'C' * (i + 4)}OC(=O)c1ccccc1" for i in range(14)]
    _generated(state, *many)

    high, _ = sub.urgency(state, {})
    assert high > 0.5

    for s in many[:12]:
        state.molecules[s].props[sim.SEARCHED_MARKER] = sub.capability.version
    low, why = sub.urgency(state, {})

    assert low < 0.2
    assert "unlikely to pay" in why


# --- the explain blobs ---------------------------------------------------------


def test_similarity_names_its_queries_and_its_rule(banked, state):
    """The opacity this subsystem exists not to repeat: which molecules, chosen how."""
    sub, _ = banked([Neighbour(CAFFEINE, "chembl", "CHEMBL113", max_phase=4.0)])
    _generated(state, CAFFEINE, ASPIRIN)

    res = _run(sub, state, query_from="generated", max_queries=2)
    q = res.explain["queries"]

    assert q["rule"]
    assert {u["smiles"] for u in q["used"]} == {CAFFEINE, ASPIRIN}
    assert all("n_hits" in u and "origin" in u for u in q["used"])
    assert q["available"]["generated"] == 2
    # Every declared rule is reported, including the ones that caught nothing: a gate is
    # only auditable if the rules that did not fire are visible too.
    assert {r["key"] for r in res.explain["safety"]["rules"]} == {
        r.key for r in sim.SAFETY_RULES
    }


def test_properties_reports_the_panel_not_just_a_count(state):
    """"Profiled 20 molecules" does not say which properties, or against what cutoffs."""
    sub = PropertiesSubsystem()
    state.add_records([{"smiles": ASPIRIN}, {"smiles": CAFFEINE}], origin="retrieval")

    res = sub.run({"smiles": []}, state, cancel=threading.Event(), progress=lambda *_: None)
    panel = res.explain["panel"]

    assert {d["key"] for d in panel["descriptors"]} >= {"mw", "logp", "tpsa", "hbd", "hba"}
    assert all(d["title"] for d in panel["descriptors"])
    assert {s["key"] for s in panel["scores"]} == {"qed", "sa_score"}
    assert {f["key"] for f in panel["flags"]} == {
        "lipinski_violations",
        "veber_pass",
        "pains_alert",
    }
    # Each flag states the rule it applies, so a reader can disagree with the cutoff.
    assert all(f["rule"] and f["of"] == 2 for f in panel["flags"])
    assert res.explain["selection"]
    assert res.explain["batch"]["profiled"] == 2
    mw = next(d for d in panel["descriptors"] if d["key"] == "mw")
    assert mw["spread"]["min"] <= mw["spread"]["median"] <= mw["spread"]["max"]


def test_unparseable_molecules_are_named_not_only_counted(state):
    sub = PropertiesSubsystem()
    state.add_records([{"smiles": "not-a-molecule"}, {"smiles": ASPIRIN}], origin="retrieval")

    res = sub.run({"smiles": []}, state, cancel=threading.Event(), progress=lambda *_: None)

    assert res.explain["batch"]["failed"] == 1
    assert res.explain["batch"]["failed_examples"] == ["not-a-molecule"]


# --- tool selection ------------------------------------------------------------


def test_disabled_tools_leave_the_schema_entirely(settings_tmp=None):
    """Not merely 'unavailable'.

    An unavailable subsystem is still in the enum the model answers under, so a model can
    name it, fail validation and degrade the step to the fallback policy -- which looks like
    a normal run in every view except the fallback count. A tool the operator switched off
    should be one the model was never offered.
    """
    full = build_registry()
    assert "docking" in full

    reduced = build_registry(enabled=["properties", "retrieval"])
    assert reduced.keys == ["properties", "retrieval"]
    assert reduced.disabled == sorted(set(full.keys) - {"properties", "retrieval"})

    enum = build_schema(reduced.capabilities())["properties"]["subsystem"]["enum"]
    assert set(enum) == {"properties", "retrieval", None}
    assert "docking" not in enum


def test_the_similarity_tool_is_installed_but_off_by_default():
    """Adding a fifth subsystem changes what round-robin does, so it is opt-in.

    Every run recorded before it existed had four tools; a form nobody touched should
    still produce that experiment.
    """
    assert "similarity" in build_registry()
    assert "similarity" not in DEFAULT_TOOLS


def test_an_unknown_tool_name_does_not_produce_an_empty_registry():
    """The caller is a web form; a stale tick must not be able to stop a run starting."""
    reg = build_registry(enabled=["properties", "no-such-tool"])
    assert reg.keys == ["properties"]


# --- the prompt ----------------------------------------------------------------


def test_prompt_additions_are_distinguishable_from_the_standard_one():
    """A run given edited instructions must not be able to read as a standard one."""
    assert compose_system() == SYSTEM
    assert compose_system("   ") == SYSTEM
    assert prompt_kind() == "standard"
    assert prompt_kind("   ", "  ") == "standard"

    appended = compose_system("Prefer cheap information.")
    assert appended.startswith(SYSTEM)
    assert appended.endswith("Prefer cheap information.")
    assert prompt_kind("Prefer cheap information.") == "appended"

    replaced = compose_system("", "You are something else entirely.")
    assert SYSTEM not in replaced
    assert replaced == "You are something else entirely."
    assert prompt_kind("", "You are something else entirely.") == "replaced"


def test_an_edited_prompt_reaches_the_policy_and_misses_the_cache():
    """Two runs with different instructions must not replay each other's decisions.

    The response cache is keyed on the system text among other things, so this holds by
    construction -- but it holds only as long as the policy uses the composed text rather
    than the module constant, which is exactly the thing that could silently regress.
    """
    with patch("drugorch.llm.build_backend") as backend:
        backend.return_value.key.return_value = "fake"
        backend.return_value.model = "m"
        backend.return_value.effort = "low"
        one = build_policy("llm", prompt_extra="Prefer structures.").inner
        two = build_policy("llm", prompt_extra="Prefer generation.").inner
        plain = build_policy("llm").inner

    assert one.system.endswith("Prefer structures.")
    assert one.prompt_kind == "appended"
    assert plain.prompt_kind == "standard"

    keys = {request_key("fake", p.system, "prefix", {}, "v1/v1") for p in (one, two, plain)}
    assert len(keys) == 3


def test_text_an_operator_typed_is_never_silently_dropped():
    """Both knobs at once applies both, rather than quietly discarding one.

    The web form only ever sends one of them, but the API takes both, and text that
    vanished from a run with nothing saying it had been ignored would be the worse failure.
    """
    both = compose_system("and also this", "Replaced base.")

    assert both == "Replaced base.\n\nand also this"
    assert prompt_kind("and also this", "Replaced base.") == "replaced+appended"
