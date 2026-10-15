"""Running one benchmark cell, and reading one back.

A cell is one (target, policy, seed). It runs headlessly through the same `Orchestrator` the
web server drives -- not a parallel implementation -- so a benchmark number and a number you
watched appear in the browser come from the same code path.

Two details that matter more than they look:

* The run is given a **holdout**, and every arm on the same (target, seed) is given the
  *same* holdout. Arms compared against different holdouts are not comparable and the
  difference does not show up in any summary statistic, so the split's fingerprint is
  recorded on every cell and checked when the grid aggregates.
* `cli.py run` omits `llm_log=` when it persists, so CLI runs silently lose their LLM traces.
  This passes it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from ..budget import BudgetLedger
from ..config import Settings, get_settings
from ..orchestrator.events import EventLog
from ..orchestrator.run import Orchestrator, drive
from ..orchestrator.state import RunState
from ..policy import build_policy
from ..store.runs import write_run_dir
from ..subsystems.registry import build_registry
from ..types import TargetSpec
from .metrics import RunMetrics, evaluate
from .splits import Split


@dataclass(slots=True)
class Cell:
    """One point in the grid."""

    target: str
    policy: str
    seed: int
    budget: float = 120.0
    max_steps: int = 40

    @property
    def run_id(self) -> str:
        # Deterministic, so a resumed grid recognises what it already ran. The timestamped
        # ids the CLI uses would make every resume a fresh run.
        safe = self.target.replace("/", "_")
        return f"bench_{safe}_{self.policy}_{self.seed}_{int(self.budget)}"

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "policy": self.policy,
            "seed": self.seed,
            "budget": self.budget,
            "run_id": self.run_id,
        }


def run_cell(
    cell: Cell,
    split: Split | None,
    *,
    settings: Settings | None = None,
    on_step=None,
) -> tuple[RunMetrics, Path]:
    """Execute one cell and score it."""
    st = settings or get_settings()
    st.ensure_dirs()

    state = RunState(
        run_id=cell.run_id,
        goal=f"find potent, selective inhibitors of {cell.target}",
        target=TargetSpec(query=cell.target),
        budget=BudgetLedger(total=cell.budget),
        seed=cell.seed,
        holdout=frozenset(split.held_out) if split else frozenset(),
    )
    registry = build_registry(st)
    log = EventLog(cell.run_id, path=st.runs_dir / cell.run_id / "events.jsonl")
    policy = build_policy(cell.policy, seed=cell.seed, settings=st)
    orch = Orchestrator(state, registry, policy, log, settings=st)
    orch.termination.max_steps = cell.max_steps

    t0 = time.perf_counter()
    drive(orch, on_step=on_step)
    wall = time.perf_counter() - t0

    out = write_run_dir(
        st.runs_dir,
        state,
        policy_describe=policy.describe(),
        registry_describe=registry.describe(),
        settings_describe=st.describe(),
        termination={
            "code": orch.stop_reason.code if orch.stop_reason else "",
            "reason": orch.stop_reason.reason if orch.stop_reason else "",
        },
        llm_log=orch.llm_log(),
    )

    # The split travels with the run so a cell can be rescored later without refetching it,
    # and so a mismatched holdout is detectable rather than merely unlikely.
    if split is not None:
        (out / "split.json").write_text(
            json.dumps({**split.to_dict(), "held_out": list(split.held_out)}, indent=2),
            encoding="utf-8",
        )

    metrics = score_run(out, cell, split)
    metrics.notes.append(f"wall {wall:.1f}s")
    return metrics, out


def score_run(run_dir: Path, cell: Cell, split: Split | None) -> RunMetrics:
    """Score a finished run directory. Works on a run recorded months ago."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    molecules = _read_molecules(run_dir)

    return evaluate(
        run_id=summary.get("run_id", cell.run_id),
        policy=summary.get("policy") or cell.policy,
        target=cell.target,
        seed=cell.seed,
        molecules=molecules,
        objective_curve=summary.get("objective_curve") or _curve(run_dir),
        budget_total=float((summary.get("budget") or {}).get("total") or cell.budget),
        spent=float((summary.get("budget") or {}).get("spent") or 0.0),
        steps=int(summary.get("steps") or 0),
        status=summary.get("status", ""),
        held_out=list(split.held_out) if split else [],
        n_held_out_scaffolds=split.n_held_out_scaffolds if split else 0,
        seen=list(split.seen) if split else [],
        n_fallbacks=int(summary.get("n_fallbacks") or 0),
    )


def _curve(run_dir: Path) -> list[dict]:
    """`objective_curve` is in the snapshot but not in summary.json, so rebuild it from the
    event log when it is missing -- a run recorded before this harness existed still scores."""
    p = run_dir / "events.jsonl"
    if not p.is_file():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "best_updated":
            out.append(
                {
                    "step": ev.get("step", 0),
                    "spent": (ev.get("budget") or {}).get("spent", 0.0),
                    "best": (ev.get("data") or {}).get("best"),
                }
            )
    return out


def _read_molecules(run_dir: Path) -> dict:
    """Reconstruct molecules from `molecules.csv`, with the spend at which each appeared.

    `molecules.csv` records `step_first_seen`, not spend, so the step is mapped through the
    budget curve in the event log. Without that, a discovery curve would be in steps -- and
    steps are not comparable across arms, which is the entire reason budget exists.
    """
    import csv

    spend_at_step = {0: 0.0}
    p = run_dir / "events.jsonl"
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = ev.get("step")
            spent = (ev.get("budget") or {}).get("spent")
            if step is not None and spent is not None:
                spend_at_step[step] = max(spend_at_step.get(step, 0.0), float(spent))

    mols: dict[str, dict] = {}
    csv_path = run_dir / "molecules.csv"
    if not csv_path.is_file():
        return mols

    with csv_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            smi = row.get("smiles")
            if not smi:
                continue
            step = int(row.get("step_first_seen") or 0)
            scores, props = {}, {}
            for k, v in row.items():
                if k in ("smiles", "origin", "step_first_seen", "parents") or v in (None, ""):
                    continue
                try:
                    num = float(v)
                except (TypeError, ValueError):
                    props[k] = v
                else:
                    (scores if k in _SCORE_KEYS else props)[k] = num
            mols[smi] = {
                "origin": row.get("origin", ""),
                "spend": spend_at_step.get(step, 0.0),
                "scores": scores,
                "props": props,
            }
    return mols


_SCORE_KEYS = {"vina_score", "pchembl", "qed", "sa_score", "selectivity_margin"}


__all__ = ["Cell", "run_cell", "score_run"]
