"""Running the public benchmark: sample, answer, score, record.

The grid here is smaller than the routing benchmark's because its cells are expensive in a
different way. A cell is one instance answered by one arm, and the cost is model calls rather
than docking, so the loop is resumable at the instance rather than at the run: every attempt
is written to its own file as it completes, and a second invocation re-scores what is on disk
instead of paying for it again. Interrupting a 200-instance run and restarting it costs the
instances that were in flight, and nothing else.

Scoring is a separate pass over the finished attempts, for the same reason it is a separate
module: the evaluator is the benchmark's, it runs in the benchmark's environment, and it is
given the whole set at once because several of its metrics are defined over the set rather
than over an instance.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from ...config import Settings, get_settings
from ...versions import all_versions
from . import baselines
from .agent import EXT_V, MAX_DECISIONS, MAX_TOOL_CALLS, Arm, Attempt
from .diagnose import decompose
from .instances import (
    DEFAULT_PATH,
    Instance,
    describe,
    load_instances,
    stratified_sample,
)
from .mdb_eval import ArmScore, score
from .paired import compare, read_verdicts
from .panel import PanelUnavailable, get_panel
from .pool import get_pool

ARMS = ("drugorch", "notool")

ProgressFn = Callable[[str, Instance, Attempt], None]


@dataclass
class ExternalResult:
    """Everything the board needs, and everything an auditor needs."""

    sample: dict = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    arms: dict = field(default_factory=dict)
    attempts: dict[str, list[dict]] = field(default_factory=dict)
    paired: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "sample": self.sample,
            "rows": self.rows,
            "arms": self.arms,
            "attempts": self.attempts,
            "paired": self.paired,
            "diagnostics": self.diagnostics,
            "provenance": self.provenance,
            "published": baselines.published(),
            "published_source": baselines.SOURCE,
            "elapsed_s": round(self.elapsed_s, 1),
        }


def attempt_path(root: Path, arm: str, instance_id: str) -> Path:
    return Path(root) / "attempts" / arm / f"{instance_id}.json"


def load_attempt(root: Path, arm: str, instance_id: str) -> Attempt | None:
    """A finished attempt from disk, if there is one.

    An attempt that recorded a fallback is not reused. A fallback means the arm's decision
    never reached the model, so the stored answer is the absence of the arm rather than the
    arm's answer, and replaying it would make a transport failure permanent.
    """
    p = attempt_path(root, arm, instance_id)
    if not p.is_file():
        return None
    try:
        att = Attempt.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, TypeError):
        return None
    return None if att.n_fallbacks else att


def save_attempt(root: Path, att: Attempt) -> None:
    p = attempt_path(root, att.arm, att.instance_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(att.to_dict(), indent=1), encoding="utf-8")
    tmp.replace(p)


def run_arm(
    arm_name: str,
    instances: list[Instance],
    *,
    root: Path,
    settings: Settings,
    seed: int = 0,
    workers: int = 4,
    resume: bool = True,
    offline: bool = False,
    on_attempt: ProgressFn | None = None,
) -> tuple[Arm, list[Attempt]]:
    """Answer every instance with one arm, resuming whatever is already on disk.

    The arm is returned with its attempts because it carries what the board has to report
    beside the score: which model answered, by which route, under which budgets.
    """
    arm = Arm(name=arm_name, settings=settings, offline=offline)
    done: dict[str, Attempt] = {}
    todo: list[Instance] = []

    for inst in instances:
        cached = load_attempt(root, arm_name, inst.uid) if resume else None
        if cached is not None:
            done[inst.uid] = cached
            if on_attempt:
                on_attempt("cached", inst, cached)
        else:
            todo.append(inst)

    if todo:
        # Threads rather than processes: the work is a model call, a parquet scan and a
        # subprocess round trip, all of which release the interpreter lock. The measurement
        # worker serialises itself behind its own lock, so more threads than this mostly
        # queue up there.
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(arm.answer, inst, seed=seed): inst for inst in todo}
            for fut in as_completed(futures):
                inst = futures[fut]
                try:
                    att = fut.result()
                except Exception as exc:  # noqa: BLE001 - one instance must not end the run
                    att = Attempt(
                        instance_id=inst.uid,
                        arm=arm_name,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                done[inst.uid] = att
                save_attempt(root, att)
                if on_attempt:
                    on_attempt("ran", inst, att)

    return arm, [done[i.uid] for i in instances if i.uid in done]


def arm_stats(attempts: list[Attempt]) -> dict:
    """What the arm did, beside what it scored.

    Tool usage and fallbacks belong on the board rather than in a file nobody opens. A row
    with fallbacks is not a row for that arm, and an arm that answered without ever calling
    a subsystem is not the arm it is labelled as.
    """
    tools: dict[str, int] = {}
    for att in attempts:
        for call in att.tool_calls:
            tools[call.tool] = tools.get(call.tool, 0) + 1
    n = len(attempts) or 1
    return {
        "n_attempts": len(attempts),
        "n_fallbacks": sum(a.n_fallbacks for a in attempts),
        "n_errors": sum(1 for a in attempts if a.error),
        "mean_tool_calls": round(sum(a.n_tool_calls for a in attempts) / n, 2),
        "mean_wall_s": round(sum(a.wall_s for a in attempts) / n, 1),
        "total_wall_s": round(sum(a.wall_s for a in attempts), 1),
        "tool_usage": dict(sorted(tools.items(), key=lambda kv: -kv[1])),
        "abstained": sum(1 for a in attempts if a.prediction == "None"),
    }


def run_external(
    *,
    n: int = 200,
    arms: tuple[str, ...] | list[str] = ARMS,
    seed: int = 0,
    workers: int = 4,
    resume: bool = True,
    offline: bool = False,
    settings: Settings | None = None,
    data_path: Path = DEFAULT_PATH,
    on_attempt: ProgressFn | None = None,
) -> ExternalResult:
    """Run the arms over a stratified sample and score them with the benchmark's evaluator."""
    started = time.time()
    st = settings or get_settings()
    st.ensure_dirs()
    root = st.data_dir / "bench" / "external"
    root.mkdir(parents=True, exist_ok=True)

    sample = stratified_sample(load_instances(data_path), n, seed=seed)
    result = ExternalResult(sample=describe(sample))

    rows: list[dict] = []
    verdicts: dict[str, dict] = {}
    attempts_by_arm: dict[str, list[Attempt]] = {}
    for arm_name in arms:
        arm, attempts = run_arm(
            arm_name,
            sample,
            root=root,
            settings=st,
            seed=seed,
            workers=workers,
            resume=resume,
            offline=offline,
            on_attempt=on_attempt,
        )
        by_id = {a.instance_id: a for a in attempts}
        # Only instances the arm actually answered are scored. Two cases are excluded and
        # they are the same case: an instance with no attempt, and an instance whose
        # decision never reached the model. Both leave the arm's answer unknown, and both
        # would otherwise be recorded as the literal "None" -- which the benchmark reads as
        # a deliberate claim that the instance is impossible. That is not a neutral
        # substitution. It costs Success on a solvable instance and it *credits* InfAcc on
        # an impossible one, so a rate limit would show up as the arm having better
        # judgement about feasibility. What was left out is counted into the arm's stats,
        # and every row carries the n it was scored over.
        answered = [
            inst
            for inst in sample
            if inst.uid in by_id and not by_id[inst.uid].n_fallbacks
        ]
        records = [
            inst.record(
                by_id[inst.uid].prediction, n_tool_calls=by_id[inst.uid].n_tool_calls
            )
            for inst in answered
        ]
        scored: dict[str, ArmScore] = score(arm_name, records, root)
        stats = {**arm_stats(attempts), "not_answered": len(sample) - len(answered)}
        for scope, row in scored.items():
            d = row.to_dict()
            d["measured_here"] = True
            d["tools"] = arm_name != "notool"
            if scope == "overall":
                d.update(stats)
            rows.append(d)
        result.arms[arm_name] = {**arm.describe(), **stats}
        result.attempts[arm_name] = [a.to_dict() for a in attempts]
        attempts_by_arm[arm_name] = attempts
        # The evaluator writes its per-instance judgement back into the file it read, so
        # the pairing below uses its verdicts rather than a second opinion formed here.
        verdicts[arm_name] = read_verdicts(root / arm_name / "moldesignbench.json")

    # The orchestrated arm against its own control, on the instances both answered. Two
    # success rates over a hundred instances can differ by chance; this says whether they
    # did. No pairing is possible against a published row, which was measured elsewhere on
    # other instances.
    for scope in ("generation", "optimization", "overall"):
        gain = compare(
            {a: v.get(scope, {}) for a, v in verdicts.items()},
            arm="drugorch",
            baseline="notool",
            scope=scope,
        )
        if gain is not None:
            result.paired.append(gain.to_dict())

    # Where the score was lost, decomposed: reading the question against solving what was
    # read. No model calls; the arm's own extracted specification is compared with the key
    # after scoring, never during it.
    for arm_name in arms:
        d = decompose(
            attempts_by_arm.get(arm_name, []),
            sample,
            root / arm_name / "moldesignbench.json",
        )
        if d:
            result.diagnostics[arm_name] = d

    result.rows = rows
    result.elapsed_s = time.time() - started
    result.provenance = {
        "date": date.today().isoformat(),
        "benchmark": "MolDesignBench",
        "benchmark_data": str(data_path),
        "scored_by": "MolDesignBench evaluate/evaluate_benchmark.py, unmodified",
        "measured_in": str(get_panel().venv),
        "pool": get_pool().describe(),
        "recipe": EXT_V,
        "max_tool_calls": MAX_TOOL_CALLS,
        "max_decisions": MAX_DECISIONS,
        "seed": seed,
        "versions": all_versions(),
    }
    return result


def save(result: ExternalResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result.to_dict(), indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rescore(
    *,
    n: int = 200,
    arms: tuple[str, ...] | list[str] = ARMS,
    seed: int = 0,
    settings: Settings | None = None,
    data_path: Path = DEFAULT_PATH,
) -> ExternalResult:
    """Score what is already on disk, without answering anything new.

    Separate from :func:`run_external` because scoring is cheap and answering is not: a
    change to how a metric is read should never risk re-spending a run's model budget.
    """
    return run_external(
        n=n,
        arms=arms,
        seed=seed,
        resume=True,
        offline=True,
        settings=settings,
        data_path=data_path,
    )


__all__ = [
    "ARMS",
    "ExternalResult",
    "PanelUnavailable",
    "arm_stats",
    "load",
    "load_attempt",
    "rescore",
    "run_arm",
    "run_external",
    "save",
]
