"""Chemistry correctness, cache versioning, and provenance.

The cache-invalidation test is the one that protects every number in the project: if bumping
a recipe version does not invalidate the rows that version produced, two recipes silently
blend into one column and no result can be trusted.
"""

from __future__ import annotations

import json
import threading

import pytest

from drugorch.budget import BudgetLedger
from drugorch.cache import ValueCache
from drugorch.chem import canonical, dedupe, is_valid, murcko_scaffold
from drugorch.orchestrator.state import RunState
from drugorch.subsystems.properties import PropertiesSubsystem, compute_panel
from drugorch.types import PANEL_MARKER, TargetSpec


# --- standardization -------------------------------------------------------
def test_salt_is_stripped_and_charge_neutralized():
    """Sodium aspirinate and aspirin are the same molecule for budget purposes."""
    assert canonical("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]") == "CC(=O)Oc1ccccc1C(=O)O"


def test_canonicalization_is_order_independent():
    """Two spellings of caffeine must key identically, or budgets double-count."""
    a = canonical("CN1C=NC2=C1C(=O)N(C)C(=O)N2C")
    b = canonical("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
    assert a == b is not None


def test_invalid_smiles_returns_none_rather_than_raising():
    """A generator will emit garbage; the run must charge for it and carry on."""
    assert canonical("not_a_molecule") is None
    assert not is_valid("C(((")


def test_dedupe_keeps_first_seen_order():
    out = dedupe(["CCO", "OCC", "CC(=O)O", "CCO"])
    assert out == ["CCO", "CC(=O)O"]


def test_murcko_scaffold_strips_side_chains():
    sc = murcko_scaffold("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    assert sc == "c1ccccc1"


# --- the property panel ----------------------------------------------------
def test_panel_reproduces_known_values():
    """Aspirin's descriptors are textbook; if these drift, RDKit or the recipe changed."""
    p = compute_panel("CC(=O)Oc1ccccc1C(=O)O")
    d = p["descriptors"]
    assert d["mw"] == pytest.approx(180.16, abs=0.05)
    assert d["heavy_atoms"] == 13
    assert d["hbd"] == 1
    assert d["hba"] == 3
    assert 0.0 <= p["scores"]["qed"] <= 1.0
    assert 1.0 <= p["scores"]["sa_score"] <= 10.0
    assert p["flags"]["lipinski_violations"] == 0
    assert p["flags"]["veber_pass"] is True


def test_lipinski_counts_violations():
    """A large greasy molecule should trip the rule of five."""
    big = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCOc1ccc(cc1)C(=O)NCCCCCCCCCCCCCCCC"
    p = compute_panel(big)
    assert p["flags"]["lipinski_violations"] >= 2


def test_panel_marks_molecules_as_profiled(tmp_path):
    """Provenance in ``props`` must not be mistaken for a computed panel.

    Retrieval also writes to ``props``; if 'has any props' counted as profiled, the
    properties subsystem would never run on retrieved molecules.
    """
    sub = PropertiesSubsystem(cache=ValueCache(tmp_path / "v.sqlite"))
    state = RunState(
        run_id="t",
        goal="g",
        target=TargetSpec(query="x"),
        budget=BudgetLedger(total=100.0),
    )
    # A molecule carrying only provenance, as retrieval leaves it.
    state.add_records([{"smiles": "CCO", "props": {"chembl_id": "CHEMBL1"}}], origin="retrieval")
    assert state.unprofiled() == ["CCO"]

    res = sub.run({"smiles": []}, state, cancel=threading.Event(), progress=lambda *a: None)
    assert res.ok
    state.add_records(res.records, origin="properties")
    assert state.unprofiled() == []
    assert PANEL_MARKER in state.molecules["CCO"].props


# --- the value cache -------------------------------------------------------
def test_cache_round_trips_and_counts_hits(tmp_path):
    c = ValueCache(tmp_path / "v.sqlite")
    c.put("k|v1", "CCO", {"score": 1.5})
    assert c.get("k|v1", "CCO") == ({"score": 1.5}, True)
    assert c.get("k|v1", "CCC") is None


def test_a_version_bump_invalidates_rather_than_blends(tmp_path):
    """The guarantee the whole provenance scheme rests on."""
    c = ValueCache(tmp_path / "v.sqlite")
    c.put("docking|v1|7LL4|ex8", "CCO", -7.5)
    # Same molecule, same receptor, different recipe version: must be a miss.
    assert c.get("docking|v2|7LL4|ex8", "CCO") is None
    # Same recipe, different receptor: also a miss.
    assert c.get("docking|v1|8BXH|ex8", "CCO") is None
    # The original is untouched.
    assert c.get("docking|v1|7LL4|ex8", "CCO") == (-7.5, True)


def test_cache_stores_failures_as_real_hits(tmp_path):
    """Re-running a deterministic failure just burns the same time to fail again."""
    c = ValueCache(tmp_path / "v.sqlite")
    c.put("k|v1", "BAD", None, ok=False)
    assert c.get("k|v1", "BAD") == (None, False)


# --- provenance ------------------------------------------------------------
def test_every_component_describes_itself_as_json(tmp_path):
    """A run.json that cannot be serialized is a run with no provenance."""
    from drugorch.config import get_settings
    from drugorch.policy import POLICY_NAMES, build_policy
    from drugorch.subsystems.registry import build_registry

    settings = get_settings()
    registry = build_registry(settings)
    blobs = {"settings": settings.describe(), "registry": registry.describe()}
    for name in POLICY_NAMES:
        blobs[name] = build_policy(name, seed=1, settings=settings).describe()

    text = json.dumps(blobs, default=None)  # default=None => raise on anything unserializable
    assert "versions" in text
    # The key must never appear in provenance that gets written to disk.
    assert "sk-ant" not in text


def test_settings_describe_never_leaks_the_api_key():
    from drugorch.config import Settings

    s = Settings(anthropic_api_key="sk-ant-secret-value")
    d = s.describe()
    assert d["has_api_key"] is True
    assert "secret" not in json.dumps(d)
