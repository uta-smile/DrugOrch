"""The bridge to ADMET-AI: the worker protocol and the subsystem over it.

Almost everything here is a regression guard for a bug that had already happened. The worker
speaks JSON lines over a pipe it shares with two chatty ML libraries and, on Windows, with
whatever multiprocessing decides to spawn, so its failures are not exceptions -- they are
*plausible wrong answers* arriving on the channel where the right one belongs. The tests
below are mostly about that channel.
"""

from __future__ import annotations

import io
import json
import threading

import pytest

from drugorch.admet import BENCH_COLUMNS, PANEL
from drugorch.admet.worker import AdmetUnavailable, AdmetWorker, worker_source
from drugorch.versions import ADMET_V


class _FakeProc:
    """Stands in for the subprocess: records what was written, replays canned lines."""

    def __init__(self, replies: list[str]):
        self.stdin = io.StringIO()
        self.stdout = None
        self._replies = list(replies)
        self.killed = False

    def poll(self):
        return None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0

    def next_line(self) -> str:
        return self._replies.pop(0) if self._replies else ""


def _worker_with(replies: list[str]) -> AdmetWorker:
    w = AdmetWorker()
    proc = _FakeProc(replies)
    w._proc = proc  # noqa: SLF001 - driving the protocol is the point of the test
    w._readline = lambda _timeout: proc.next_line()  # noqa: SLF001
    return w


def test_worker_script_compiles():
    compile(worker_source(), "worker", "exec")


def test_worker_script_guards_main():
    """The guard is not style. Without it, batches of 100+ silently return the wrong object.

    ADMET-AI opens a multiprocessing Pool at 100 molecules. Windows spawns those children by
    re-importing the parent script, which re-runs ``main()`` and prints a *second* startup
    banner into the response pipe. The reader then takes that banner as the answer to its
    prediction request. Nothing crashes; a batch of 50 works and a batch of 200 does not.
    """
    src = worker_source()
    assert '__name__ == "__main__"' in src
    assert src.rstrip().endswith("main()\n'''".rstrip("'")) or "main()" in src


def test_a_second_banner_is_reported_as_what_it_is():
    """A banner where a prediction belongs names the process tree, not the model."""
    w = _worker_with([json.dumps({"ready": True})])
    with pytest.raises(AdmetUnavailable, match="second startup banner"):
        w.predict(["CCO"])


def test_library_chatter_cannot_be_mistaken_for_a_reply():
    """chemprop writes progress bars to stdout, which is the response channel.

    The worker swaps ``sys.stdout`` for a bin before importing anything and keeps the real
    handle to itself, so this is enforced at the source rather than filtered downstream --
    where a stray line that happened to parse as JSON would be indistinguishable from a reply.
    """
    src = worker_source()
    assert "_REAL = sys.stdout" in src
    assert "sys.stdout = io.StringIO()" in src
    assert src.index("sys.stdout = io.StringIO()") < src.index("def main()")


def test_predict_returns_rows_keyed_by_smiles():
    reply = json.dumps({"ok": True, "predictions": {"CCO": {"AMES": 0.1}}})
    w = _worker_with([reply])
    assert w.predict(["CCO"]) == {"CCO": {"AMES": 0.1}}
    sent = json.loads(w._proc.stdin.getvalue().strip())  # noqa: SLF001
    assert sent["smiles"] == ["CCO"]


def test_requested_columns_travel_with_the_request():
    """The model returns 98 columns and callers want 19; asking for all of them is waste."""
    w = _worker_with([json.dumps({"ok": True, "predictions": {}})])
    w.predict(["CCO"], columns=["AMES", "hERG"])
    sent = json.loads(w._proc.stdin.getvalue().strip())  # noqa: SLF001
    assert sent["columns"] == ["AMES", "hERG"]


def test_an_empty_batch_never_touches_the_worker():
    w = _worker_with([])
    assert w.predict([]) == {}


def test_a_failed_prediction_raises_rather_than_returning_defaults():
    w = _worker_with([json.dumps({"ok": False, "error": "model exploded"})])
    with pytest.raises(AdmetUnavailable, match="model exploded"):
        w.predict(["CCO"])


def test_silence_is_a_timeout_not_an_empty_result():
    w = _worker_with([""])
    with pytest.raises(AdmetUnavailable, match="timed out"):
        w.predict(["CCO"])


def test_missing_venv_explains_how_to_build_it():
    w = AdmetWorker(venv="does-not-exist-anywhere")
    with pytest.raises(AdmetUnavailable, match="uv venv"):
        w.python()


def test_panel_does_not_collide_with_the_properties_subsystem():
    """Two recipes in one column is the failure ``versions.py`` exists to prevent.

    ADMET-AI also reports logP, molecular weight, TPSA, QED and friends. Writing those into
    the same props a molecule already carries from the RDKit panel would blend two different
    computations into one column with nothing downstream able to tell.
    """
    from drugorch.subsystems.properties import DESCRIPTORS

    owned = set(DESCRIPTORS) | {"qed", "sa"}
    assert not (set(PANEL) & owned)
    assert all(p.startswith("admet_") for p in PANEL)


def test_bench_columns_cover_the_four_endpoints_rdkit_cannot_reach():
    assert {"Mutag", "BBBP", "logS", "logD"} <= set(BENCH_COLUMNS)
    assert BENCH_COLUMNS["Mutag"] == "AMES"
    assert BENCH_COLUMNS["BBBP"] == "BBB_Martins"
    assert BENCH_COLUMNS["logS"] == "Solubility_AqSolDB"
    assert BENCH_COLUMNS["logD"] == "Lipophilicity_AstraZeneca"


# --- the subsystem ----------------------------------------------------------------


def _plugin(cache=None, worker=None):
    import plugins.admet as mod

    sub = mod.AdmetSubsystem(cache=cache)
    if worker is not None:
        mod.get_worker = lambda _venv=None: worker  # noqa: ARG005
    return sub


class _StubWorker:
    def __init__(self, table=None, fail=None):
        self.table = table or {}
        self.fail = fail
        self.calls: list[list[str]] = []
        self.running = True
        self.venv = ".venv-mdb"

    def predict(self, smiles, columns=None):
        self.calls.append(list(smiles))
        if self.fail:
            raise AdmetUnavailable(self.fail)
        return {s: self.table[s] for s in smiles if s in self.table}


@pytest.fixture
def admet_state():
    from drugorch.budget import BudgetLedger
    from drugorch.orchestrator.state import RunState
    from drugorch.types import MoleculeRecord, TargetSpec

    st = RunState(
        run_id="t", goal="g", target=TargetSpec(query="n/a"), budget=BudgetLedger(total=100.0)
    )
    for s in ("CCO", "CCN"):
        st.molecules[s] = MoleculeRecord(smiles=s)
    return st


def test_an_unavailable_model_fails_the_call_instead_of_guessing(admet_state, monkeypatch):
    """No defaults. A substituted value ranks identically to a predicted one."""
    import plugins.admet as mod

    monkeypatch.setattr(mod, "get_worker", lambda _v=None: _StubWorker(fail="no venv"))
    sub = mod.AdmetSubsystem()
    res = sub.run({}, admet_state, cancel=threading.Event(), progress=lambda *a: None)

    assert not res.ok
    assert res.records == []
    assert res.cost.budget_units == 0.0
    assert "no venv" in res.error


def test_cached_molecules_cost_wall_time_but_not_budget(admet_state, monkeypatch, tmp_path):
    """A cache hit saves compute, never budget -- the run did not know the answer."""
    import plugins.admet as mod
    from drugorch.cache import ValueCache

    table = {s: dict.fromkeys((c for c, _u in PANEL.values()), 0.5) for s in ("CCO", "CCN")}
    worker = _StubWorker(table=table)
    monkeypatch.setattr(mod, "get_worker", lambda _v=None: worker)

    cache = ValueCache(tmp_path / "cache.sqlite")
    sub = mod.AdmetSubsystem(cache=cache)

    first = sub.run({}, admet_state, cancel=threading.Event(), progress=lambda *a: None)
    assert first.ok
    assert first.cost.budget_units == pytest.approx(0.1 * 2)
    assert first.cost.cache_hits == 0

    second = mod.AdmetSubsystem(cache=cache).run(
        {}, admet_state, cancel=threading.Event(), progress=lambda *a: None
    )
    assert second.ok
    assert second.cost.cache_hits == 2
    assert second.cost.budget_units == 0.0
    assert len(worker.calls) == 1, "the second call must not reach the model"


def test_the_cache_key_moves_with_the_model(tmp_path):
    """v1 rows must not be served to v2. The two models disagree by about a window width."""
    import plugins.admet as mod

    sub = mod.AdmetSubsystem()
    assert ADMET_V in sub.cache_key()
    assert mod.PANEL_V in sub.cache_key()


def test_preflight_refuses_when_there_is_nothing_to_do(admet_state):
    import plugins.admet as mod

    sub = mod.AdmetSubsystem()
    ok, _ = sub.preflight({}, admet_state)
    assert ok

    for rec in admet_state.molecules.values():
        rec.props[mod.PRIMARY] = 0.1
    ok, why = sub.preflight({}, admet_state)
    assert not ok
    assert "already" in why


def test_urgency_stays_below_docking(admet_state):
    """It characterises molecules; it never moves the objective.

    A run that spends its budget profiling molecules it never measured against the target has
    learned a great deal about nothing.
    """
    import plugins.admet as mod

    score, why = mod.AdmetSubsystem().urgency(admet_state, {})
    assert 0.0 < score < 1.0
    assert why

    for rec in admet_state.molecules.values():
        rec.props[mod.PRIMARY] = 0.1
    assert mod.AdmetSubsystem().urgency(admet_state, {})[0] == 0.0
