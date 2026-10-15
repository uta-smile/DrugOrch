"""Scoring our answers with the benchmark's own evaluator.

Nothing in this module computes Success, CSR, MCD, InfAcc, F1, MCC or Validity. It writes the
arm's answers into the layout MolDesignBench's ``evaluate/evaluate_benchmark.py`` reads, runs
that script unmodified in the environment its answer key was built in, and parses the summary
it writes with the benchmark's own ``make_leaderboard.parse_eval_file``.

The alternative was to reimplement seven metrics from the paper's description and put the
result on the same axis as its published numbers. Each of those metrics has a definition that
is easy to get subtly wrong and impossible to notice afterwards: CSR averages per instance
over total feasible rather than over valid, MCD normalises each property by its p5-p95 range
over ten million molecules and charges a full unit for an unmeasurable one, and Success counts
an infeasible instance answered ``None`` as a success. A reimplementation that got one of
these wrong would still produce a plausible number, and the whole point of the comparison is
that our row and theirs mean the same thing.

What this module is therefore responsible for is only the translation, and the one claim it
makes is in :func:`score`: the numbers it returns came out of their evaluator.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .panel import BENCH_REPO, DEFAULT_VENV, PanelUnavailable

EVALUATOR = Path("evaluate/evaluate_benchmark.py")
LEADERBOARD = Path("evaluate/make_leaderboard.py")

TASKS = ("generation", "optimization")


@dataclass
class ArmScore:
    """One arm's row, exactly as the benchmark's evaluator reported it."""

    arm: str
    scope: str = "overall"  # overall | generation | optimization
    n: int = 0
    success: float = 0.0
    csr: float = 0.0
    mcd: float = 0.0
    inf_acc: float = 0.0
    f1: float = 0.0
    mcc: float = 0.0
    validity: float = 0.0
    tool_calls: float = 0.0
    success_count: int = 0
    feasible_total: int = 0
    infeasible_total: int = 0
    confusion: tuple[int, int, int, int] = (0, 0, 0, 0)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "scope": self.scope,
            "n": self.n,
            "success": round(self.success, 4),
            "csr": round(self.csr, 4),
            "mcd": round(self.mcd, 2),
            "inf_acc": round(self.inf_acc, 4),
            "f1": round(self.f1, 4),
            "mcc": round(self.mcc, 4),
            "validity": round(self.validity, 4),
            "tool_calls": round(self.tool_calls, 2),
            "success_count": self.success_count,
            "feasible_total": self.feasible_total,
            "infeasible_total": self.infeasible_total,
            "confusion": list(self.confusion),
            **self.extra,
        }


def _load_parser(repo: Path):
    """Import the benchmark's leaderboard parser from its file.

    By path rather than by package, because importing ``moldesign`` pulls in the LLM clients
    and the tool servers. This module only needs ``parse_eval_file`` and ``combine_metrics``,
    which are regex over a text file and arithmetic over two dicts.
    """
    path = (repo / LEADERBOARD).resolve()
    if not path.is_file():
        raise PanelUnavailable(f"{path} is missing; the benchmark sources are incomplete")
    spec = importlib.util.spec_from_file_location("_mdb_leaderboard", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise PanelUnavailable(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_records(records: list[dict], folder: Path) -> Path:
    """Lay the arm's answers out the way the evaluator expects to find them.

    Summaries from previous runs are removed first. The evaluator names its output after
    the task types present in the file it reads, so a run covering one task leaves the
    other task's summary from an earlier run in place, and :func:`score` would then publish
    those stale numbers as this run's.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.glob("evaluation_*.txt"):
        stale.unlink()
    # One file holding both task types. The evaluator splits on `task_type` itself and
    # writes one summary per task, which is the split the paper reports in.
    path = folder / "moldesignbench.json"
    path.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def run_evaluator(
    base_dir: Path, folder: str, *, venv: Path = DEFAULT_VENV, repo: Path = BENCH_REPO,
    timeout_s: float = 3600.0,
) -> str:
    """Run their evaluator over one folder and return its stdout.

    It runs from the benchmark's own directory because the tools it loads resolve their data
    paths relative to it, and in the pinned interpreter because the properties it measures
    have to be the ones the answer key was built from.
    """
    repo = Path(repo).resolve()
    for rel in ("Scripts/python.exe", "bin/python"):
        py = Path(venv) / rel
        if py.is_file():
            break
    else:
        raise PanelUnavailable(
            f"no interpreter in {venv}; the benchmark is scored in its own environment"
        )

    # PYTHONPATH rather than cwd: a script run by path puts its own directory on sys.path,
    # not the working directory, so `import moldesign` fails from inside evaluate/ even with
    # the repository as cwd. The cwd still has to be the repository, because the tools it
    # loads resolve their data paths relative to it.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["TQDM_DISABLE"] = "1"

    proc = subprocess.run(
        [
            str(py.resolve()),
            str(repo / EVALUATOR),
            "--base_dir",
            str(Path(base_dir).resolve()),
            "--folder",
            folder,
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise PanelUnavailable(
            "the benchmark evaluator failed: " + (tail[-1] if tail else "no output")
        )
    return proc.stdout


def score(
    arm: str,
    records: list[dict],
    base_dir: Path,
    *,
    venv: Path = DEFAULT_VENV,
    repo: Path = BENCH_REPO,
) -> dict[str, ArmScore]:
    """Score one arm's answers. Returns a row per task plus the combined row.

    The combined row is built with the benchmark's own ``combine_metrics``, which pools the
    confusion matrices before recomputing MCC rather than averaging two MCCs -- a weighted
    mean of two correlation coefficients is not the coefficient of the pooled set.
    """
    base_dir = Path(base_dir)
    folder = base_dir / arm
    write_records(records, folder)
    run_evaluator(base_dir, arm, venv=venv, repo=repo)

    parser = _load_parser(Path(repo))
    rows: dict[str, ArmScore] = {}
    per_task: dict[str, dict] = {}

    # Which tasks this run actually covered. The evaluator drops the task suffix when the
    # file it reads holds one task type, and the undecorated summary then belongs to that
    # task alone. Reading it for both -- which is what happens when one task's instances
    # all fell back -- publishes one task's numbers under the other's name and makes the
    # combined row count every instance twice.
    present = {str(r.get("task_type")) for r in records}

    for task in TASKS:
        if task not in present:
            per_task[task] = parser.empty_metrics()
            continue
        path = folder / f"evaluation_moldesignbench_{task}.txt"
        if not path.is_file() and len(present) == 1:
            path = folder / "evaluation_moldesignbench.txt"
        if not path.is_file():
            per_task[task] = parser.empty_metrics()
            continue
        m = parser.parse_eval_file(str(path))
        per_task[task] = m
        rows[task] = _row(arm, task, m)

    combined = parser.combine_metrics(per_task[TASKS[0]], per_task[TASKS[1]])
    rows["overall"] = _row(arm, "overall", combined)
    return rows


def _row(arm: str, scope: str, m: dict) -> ArmScore:
    return ArmScore(
        arm=arm,
        scope=scope,
        n=int(m.get("total", 0)),
        success=float(m.get("success", 0.0)),
        csr=float(m.get("csr", 0.0)),
        mcd=float(m.get("mcd", 0.0)),
        inf_acc=float(m.get("inf_acc", 0.0)),
        f1=float(m.get("f1", 0.0)),
        mcc=float(m.get("mcc", 0.0)),
        validity=float(m.get("validity", 0.0)),
        tool_calls=float(m.get("tool_calls", 0.0)),
        success_count=int(m.get("success_count", 0)),
        feasible_total=int(m.get("feasible_total", 0)),
        infeasible_total=int(m.get("infeasible_total", 0)),
        confusion=tuple(int(x) for x in (m.get("confusion") or (0, 0, 0, 0))),  # type: ignore[arg-type]
    )


__all__ = ["ArmScore", "TASKS", "run_evaluator", "score", "write_records"]
