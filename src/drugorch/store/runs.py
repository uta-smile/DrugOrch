"""What a finished run leaves on disk.

A run directory has to be enough to answer, months later and without the code that produced
it: what was asked, what was decided, what it cost, and what came out. That is why
``run.json`` carries the merged ``describe()`` of every component including the recipe
version constants -- a number whose provenance is "some version of the docking code" is not
a result.

Five files, each with one job:

``run.json``        provenance: config, policy, subsystems, versions, termination.
``events.jsonl``    the full event stream, which replays into the routing trace.
``decisions.csv``   one row per routing decision -- the table the ablation reads.
``molecules.csv``   every molecule with its properties and scores.
``summary.json``    headline numbers: best, budget curve, per-subsystem spend.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

from .. import __version__
from ..orchestrator.state import RunState
from ..versions import all_versions


def _package_versions() -> dict:
    """Versions of the libraries whose behaviour can change a number."""
    out: dict[str, str] = {}
    for mod in ("rdkit", "numpy", "gemmi", "meeko", "anthropic", "httpx"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "not installed"
    return out


def write_run_dir(
    root: Path,
    state: RunState,
    *,
    policy_describe: dict,
    registry_describe: dict,
    settings_describe: dict,
    termination: dict | None = None,
    config_hash: str = "",
    llm_log: list[dict] | None = None,
) -> Path:
    """Write the full run directory. Returns its path."""
    out = Path(root) / state.run_id
    out.mkdir(parents=True, exist_ok=True)

    (out / "run.json").write_text(
        json.dumps(
            {
                "run_id": state.run_id,
                "goal": state.goal,
                "seed": state.seed,
                "status": state.status,
                "target": state.target.to_dict(),
                "objective": {
                    "key": state.objective_key,
                    "maximize": state.objective_maximize,
                },
                "config_hash": config_hash,
                "termination": termination or {},
                "policy": policy_describe,
                "subsystems": registry_describe,
                "settings": settings_describe,
                "versions": all_versions(),
                "packages": _package_versions(),
                "drugorch_version": __version__,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "elapsed_s": round(state.elapsed_s, 3),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    _write_decisions(out / "decisions.csv", state)
    _write_molecules(out / "molecules.csv", state)

    # The two files that make a finished run re-examinable rather than merely reported.
    # `decisions.json` carries what each policy actually saw; `llm_calls.jsonl` carries the
    # exchanges behind an LLM run, including the ones that fell back. Both are written even
    # when empty, so their absence means the run predates this format rather than that
    # nothing happened.
    (out / "decisions.json").write_text(
        json.dumps([r.to_dict_full() for r in state.history], indent=2, default=str),
        encoding="utf-8",
    )
    with (out / "llm_calls.jsonl").open("w", encoding="utf-8") as fh:
        for entry in llm_log or []:
            fh.write(json.dumps(entry, default=str) + "\n")

    (out / "summary.json").write_text(
        json.dumps(summarize(state, policy_describe.get("name", "")), indent=2, default=str),
        encoding="utf-8",
    )
    return out


def _write_decisions(path: Path, state: RunState) -> None:
    """One row per routing decision.

    ``considered`` is flattened to a single field rather than dropped: without it the table
    records what a policy did but not what it weighed, and the rejected alternatives are half
    the evidence that routing was adaptive at all.
    """
    cols = [
        "step",
        "policy",
        "source",
        "action",
        "subsystem",
        "params",
        "executed",
        "ok",
        "cost_units",
        "wall_s",
        "latency_ms",
        "error",
        "rationale",
        "considered",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for rec in state.history:
            d = rec.decision
            w.writerow(
                {
                    "step": rec.step,
                    "policy": d.policy,
                    "source": d.source,
                    "action": d.action,
                    "subsystem": d.subsystem or "",
                    "params": json.dumps(d.params, sort_keys=True),
                    "executed": rec.executed,
                    "ok": "" if rec.ok is None else rec.ok,
                    "cost_units": round(rec.cost_units, 4),
                    "wall_s": round(rec.wall_s, 4),
                    "latency_ms": round(rec.latency_ms, 2),
                    "error": rec.error or "",
                    "rationale": d.rationale,
                    "considered": json.dumps([a.to_dict() for a in d.considered]),
                }
            )


def _write_molecules(path: Path, state: RunState) -> None:
    mols = list(state.molecules.values())
    prop_keys = sorted({k for m in mols for k in m.props})
    score_keys = sorted({k for m in mols for k in m.scores})
    cols = ["smiles", "origin", "step_first_seen", "parents", *score_keys, *prop_keys]

    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for m in sorted(mols, key=lambda m: m.step_first_seen):
            row: dict[str, Any] = {
                "smiles": m.smiles,
                "origin": m.origin,
                "step_first_seen": m.step_first_seen,
                "parents": "|".join(m.parents),
            }
            row.update({k: m.scores.get(k, "") for k in score_keys})
            row.update({k: m.props.get(k, "") for k in prop_keys})
            w.writerow(row)


def summarize(state: RunState, policy_name: str = "") -> dict:
    """Headline numbers for the run list and the comparison view."""
    by_sub: dict[str, dict] = {}
    for rec in state.history:
        key = rec.decision.subsystem or "(stop)"
        e = by_sub.setdefault(
            key, {"calls": 0, "executed": 0, "failed": 0, "units": 0.0, "wall_s": 0.0}
        )
        e["calls"] += 1
        if rec.executed:
            e["executed"] += 1
            e["units"] += rec.cost_units
            e["wall_s"] += rec.wall_s
            if rec.ok is False:
                e["failed"] += 1

    for e in by_sub.values():
        e["units"] = round(e["units"], 4)
        e["wall_s"] = round(e["wall_s"], 3)

    scored = state.scored()
    return {
        "policy": policy_name,
        "run_id": state.run_id,
        "goal": state.goal,
        "status": state.status,
        "seed": state.seed,
        "steps": state.step,
        "target": state.target.label,
        "objective": state.objective_key,
        "best": round(scored[0][1], 4) if scored else None,
        "best_smiles": scored[0][0] if scored else None,
        "n_molecules": len(state.molecules),
        "n_scored": len(scored),
        "budget": state.budget.snapshot(),
        "budget_curve": state.budget.curve(),
        "by_subsystem": by_sub,
        "n_fallbacks": sum(1 for r in state.history if r.decision.source == "fallback"),
        "elapsed_s": round(state.elapsed_s, 2),
        "top": [{"smiles": s, "value": round(v, 4)} for s, v in state.best(20)],
    }
