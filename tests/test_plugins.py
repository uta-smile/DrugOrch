"""Subsystem discovery: loading, failing visibly, and staying visible to every policy.

The point of a plugin system is not that a plugin can run -- it is that a plugin behaves like
a first-class subsystem everywhere, including in the places that used to key on the four
built-in names. The two tests that matter most here are the ones asserting a plugin appears
in `considered` under the rule-based arms, and that a plugin which fails to load is recorded
rather than silently absent. Both were real defects before this module existed.
"""

from __future__ import annotations

import textwrap

import pytest

from drugorch.subsystems.base import BaseSubsystem, Capability, ParamSpec
from drugorch.subsystems.plugins import (
    LoadFailure,
    SubsystemContext,
    discover_paths,
    instantiate,
)
from drugorch.subsystems.registry import BUILTIN_ORDER, Registry, build_registry


def _plugin_source(key: str, *, params: str = "", broken: bool = False) -> str:
    if broken:
        return "raise RuntimeError('this plugin is deliberately broken')\n"
    return textwrap.dedent(f"""
        from drugorch.subsystems.base import BaseSubsystem, Capability, ParamSpec

        CAPABILITY = Capability(
            key="{key}", title="t", summary="s",
            params=({params}),
        )

        class Sub(BaseSubsystem):
            capability = CAPABILITY

        def build(ctx):
            return Sub()
    """)


def _write(tmp_path, name: str, body: str):
    d = tmp_path / "plugins"
    d.mkdir(exist_ok=True)
    (d / f"{name}.py").write_text(body, encoding="utf-8")
    return d


# --- discovery -------------------------------------------------------------
def test_a_plugin_is_found_in_a_directory(tmp_path):
    d = _write(tmp_path, "mytool", _plugin_source("mytool"))
    found, failed = discover_paths([d])
    assert [f.name for f in found] == ["mytool"]
    assert failed == []
    assert found[0].source.startswith("path:")


def test_a_broken_plugin_is_recorded_not_swallowed(tmp_path):
    """The old code caught ImportError and passed, so a plugin that could not load simply
    was not there -- no log line, nothing in the run record, a normal-looking run."""
    d = _write(tmp_path, "broken", _plugin_source("x", broken=True))
    found, failed = discover_paths([d])
    assert found == []
    assert len(failed) == 1
    assert failed[0].name == "broken"
    assert "deliberately broken" in failed[0].error


def test_a_plugin_that_raises_at_construction_is_recorded(tmp_path):
    d = _write(tmp_path, "badfactory", "def build(ctx):\n    raise ValueError('no')\n")
    found, _ = discover_paths([d])
    built, _origins, failed = instantiate(found, SubsystemContext(settings=None, cache=None))
    assert built == []
    assert "ValueError" in failed[0].error


def test_a_factory_returning_a_non_subsystem_is_rejected(tmp_path):
    """Catches a factory that returned a config dict or None by mistake."""
    d = _write(tmp_path, "notasub", "def build(ctx):\n    return {'not': 'a subsystem'}\n")
    found, _ = discover_paths([d])
    built, _o, failed = instantiate(found, SubsystemContext(settings=None, cache=None))
    assert built == []
    assert "does not satisfy Subsystem" in failed[0].error


def test_a_subsystem_missing_run_is_rejected(tmp_path):
    """The structural gate earns its place here: `run` is the one method the orchestrator
    calls unconditionally, so a subsystem without it would fail at the worst moment."""
    body = textwrap.dedent("""
        from drugorch.subsystems.base import BaseSubsystem, Capability

        class S(BaseSubsystem):
            capability = Capability(key="norun", title="t", summary="s")

        def build(ctx):
            return S()
    """)
    d = _write(tmp_path, "norun", body)
    found, _ = discover_paths([d])
    built, _o, failed = instantiate(found, SubsystemContext(settings=None, cache=None))
    assert built == []
    assert "does not satisfy Subsystem" in failed[0].error


def test_a_module_without_a_build_factory_is_rejected(tmp_path):
    d = _write(tmp_path, "nofactory", "X = 1\n")
    _found, failed = discover_paths([d])
    assert "build() factory" in failed[0].error


def test_a_missing_plugin_directory_is_reported(tmp_path):
    _found, failed = discover_paths([tmp_path / "nope"])
    assert failed[0].error == "not a directory"


def test_one_factory_may_produce_several_subsystems(tmp_path):
    body = textwrap.dedent("""
        from drugorch.subsystems.base import BaseSubsystem, Capability

        def _mk(k):
            class S(BaseSubsystem):
                capability = Capability(key=k, title="t", summary="s")
                def run(self, params, state, *, cancel, progress):
                    return None
            return S()

        def build(ctx):
            return [_mk("alpha"), _mk("beta")]
    """)
    d = _write(tmp_path, "family", body)
    found, _ = discover_paths([d])
    built, _o, failed = instantiate(found, SubsystemContext(settings=None, cache=None))
    assert sorted(s.capability.key for s in built) == ["alpha", "beta"]
    assert failed == []


# --- collisions ------------------------------------------------------------
def _sub(key: str, params: tuple = ()):
    class S(BaseSubsystem):
        capability = Capability(key=key, title="t", summary="s", params=params)

    return S()


def test_a_duplicate_key_keeps_the_first_and_records_the_loser():
    """A plugin must not be able to silently substitute itself for a built-in."""
    r = Registry([_sub("docking"), _sub("docking")])
    assert r.keys == ["docking"]
    assert any("duplicate capability key" in f.error for f in r.load_failures)


def test_a_param_type_collision_is_detected():
    """The LLM's params object unions names across subsystems and the first wins, so a
    plugin declaring `smiles` as an int would corrupt docking's schema invisibly."""
    a = _sub("a", (ParamSpec(name="smiles", type="smiles_list", default=[]),))
    b = _sub("b", (ParamSpec(name="smiles", type="int", default=1),))
    r = Registry([a, b])
    assert any(f.source == "param_collision" for f in r.load_failures)
    assert "will not match" in next(f for f in r.load_failures if f.source == "param_collision").error


def test_the_same_param_name_with_the_same_type_is_fine():
    a = _sub("a", (ParamSpec(name="smiles", type="smiles_list", default=[]),))
    b = _sub("b", (ParamSpec(name="smiles", type="smiles_list", default=[]),))
    assert Registry([a, b]).load_failures == []


# --- the built-ins go through the same path --------------------------------
def test_built_ins_load_through_discovery():
    r = build_registry()
    assert set(BUILTIN_ORDER) <= set(r.keys)
    for key in BUILTIN_ORDER:
        assert r.origins[key].source.startswith(("entry_point:", "builtin:"))


def test_registry_order_is_pinned():
    """Order is load-bearing: RoundRobinPolicy cycles it, so alphabetical discovery would
    silently change that arm and break comparability with every earlier run."""
    r = build_registry()
    assert [k for k in r.keys if k in BUILTIN_ORDER] == list(BUILTIN_ORDER)


def test_describe_carries_load_failures():
    r = Registry([_sub("x"), _sub("x")])
    assert "_load_failures" in r.describe()


def test_describe_is_json_safe_with_failures():
    import json

    r = Registry([_sub("x"), _sub("x")])
    json.dumps(r.describe(), default=None)  # default=None => raise on anything unserializable


# --- the regression that would make plugins pointless ----------------------
@pytest.mark.parametrize("policy_name", ["scripted", "oracle", "random", "roundrobin", "fixed"])
def test_every_policy_can_see_a_subsystem_it_has_no_rule_for(policy_name, tmp_path):
    """`scripted` and `oracle` score by literal name. A plugin key never entered `scores`,
    so it never entered `considered` -- unroutable *and* invisible in the trace. A plugin
    may lose, but it must be seen losing.
    """
    from drugorch.budget import BudgetLedger
    from drugorch.orchestrator.state import RunState
    from drugorch.policy import build_policy
    from drugorch.types import TargetSpec

    reg = Registry([_sub("retrieval"), _sub("properties"), _sub("docking"), _sub("newtool")])
    state = RunState(
        run_id="t",
        goal="g",
        target=TargetSpec(query="JAK2", uniprot="O60674", pdb_id="7LL4"),
        budget=BudgetLedger(total=100.0),
        seed=42,
    )
    state.registry = reg
    state.step = 3
    state.facts = {"known_actives": {"n": 5}}
    state.add_records([{"smiles": "CCO", "scores": {"pchembl": 8.0}}], origin="retrieval")

    policy = build_policy(policy_name, seed=42)
    policy.reset(state)
    decision = policy.decide(state, reg.capabilities())

    seen = {a.subsystem for a in decision.considered} | {decision.subsystem}
    assert "newtool" in seen, f"{policy_name} erased a subsystem it had no rule for"


def test_a_plugin_whose_urgency_raises_does_not_break_routing():
    """A bad plugin degrades its own score, never the run."""
    from drugorch.budget import BudgetLedger
    from drugorch.orchestrator.state import RunState
    from drugorch.policy import build_policy
    from drugorch.types import TargetSpec

    class Exploding(BaseSubsystem):
        capability = Capability(key="exploding", title="t", summary="s")

        def urgency(self, state, facts):
            raise RuntimeError("boom")

    reg = Registry([_sub("retrieval"), Exploding()])
    state = RunState(
        run_id="t", goal="g", target=TargetSpec(query="x", uniprot="P1"),
        budget=BudgetLedger(total=100.0), seed=1,
    )
    state.registry = reg
    policy = build_policy("scripted", seed=1)
    policy.reset(state)
    decision = policy.decide(state, reg.capabilities())

    alt = next(a for a in decision.considered if a.subsystem == "exploding")
    assert "urgency() failed" in alt.reason
    assert decision.action in ("call", "stop")


def test_load_failure_round_trips():
    f = LoadFailure("n", "s", "e")
    assert f.to_dict() == {"name": "n", "source": "s", "error": "e"}
