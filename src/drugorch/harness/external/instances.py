"""Loading MolDesignBench instances, and the line between a question and its answer key.

Each instance carries three things that must not be confused: the ``question`` an agent is
allowed to read, the ``answer`` the evaluator scores against, and ``meta_info`` which holds
both an agent-visible seed molecule (optimization tasks start from one) and a reference
solution that no agent may see.

:func:`prompt_view` is that boundary made explicit. Everything an arm receives passes through
it, and it returns a new dict rather than filtering in place, so an arm cannot reach the
answer key by following an attribute off the object it was handed. ``tests/test_external.py``
asserts the property this buys: no field of ``answer``, and no ``ref_smiles``, appears in any
prompt sent for an instance. A leak here would not fail loudly. It would simply produce a
score, and every number on the public board would be fiction.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path("data/external/moldesignbench.jsonl")

TASK_TYPES = ("generation", "optimization")


@dataclass(frozen=True, slots=True)
class Instance:
    """One benchmark question, with its answer key kept beside it rather than inside it."""

    id: str
    task_type: str
    question: str
    answer: dict = field(default_factory=dict)
    infeasible: bool = False
    seed_smiles: str | None = None
    ref_smiles: str | None = None
    preserve_fgs: tuple[str, ...] = ()
    ordinal: int = 0
    """Which occurrence of :attr:`id` this is, in file order.

    The shipped dataset reuses 112 of its ids for a *second, different* instance: same id,
    different question, different answer. Keying anything on ``id`` therefore silently
    merges two questions. Measured on the default sample, seed 0 draws two such pairs, and
    the damage is invisible: one attempt overwrites the other's file, the survivor is scored
    against the wrong answer key, and every aggregate downstream counts 199 instances while
    reporting 200."""

    @property
    def uid(self) -> str:
        """The identity everything in this package keys on.

        Equal to :attr:`id` for the 1,886 ids that appear once, so files and logs stay
        readable, and suffixed for the rest.
        """
        return self.id if self.ordinal == 0 else f"{self.id}~{self.ordinal}"

    @classmethod
    def from_row(cls, row: dict, *, ordinal: int = 0) -> Instance:
        meta = row.get("meta_info") or {}
        goal = meta.get("optimization_goal") or {}
        return cls(
            id=str(row.get("id", "")),
            ordinal=ordinal,
            task_type=str(row.get("task_type", "generation")),
            question=str(row.get("question", "")),
            answer=row.get("answer") or {},
            infeasible=bool(meta.get("infeasible", False)),
            seed_smiles=meta.get("seed_smiles"),
            ref_smiles=meta.get("ref_smiles"),
            preserve_fgs=tuple(str(g) for g in (goal.get("preserve_fgs") or [])),
        )

    @property
    def n_constraints(self) -> int:
        props = len(self.answer.get("properties") or [])
        frags = sum(len(d) for d in (self.answer.get("fragments") or []) if isinstance(d, dict))
        return props + frags

    def record(self, prediction: str, *, n_tool_calls: int = 0) -> dict:
        """This instance as the benchmark's evaluator expects to read it back.

        The evaluator reads ``prediction``, ``answer``, ``task_type`` and the infeasible flag,
        and counts ``conversations`` entries whose role is ``tool`` for its tool-call column.
        The conversation is reduced to that count rather than carried verbatim: the full
        transcript lives in the arm's own record, and a 40-turn transcript per instance would
        make the file the evaluator reads tens of megabytes for one integer.
        """
        return {
            # The evaluator only carries this field through, so the unique one goes here
            # and the dataset's own id travels beside it. Keying verdicts on a reused id
            # would merge two instances after scoring exactly as it would before.
            "id": self.uid,
            "mdb_id": self.id,
            "task_type": self.task_type,
            "question": self.question,
            "answer": self.answer,
            "infeasible": self.infeasible,
            "prediction": prediction,
            "conversations": [{"role": "tool", "name": "drugorch", "content": ""}]
            * int(n_tool_calls),
        }


def prompt_view(inst: Instance) -> dict:
    """What an arm is allowed to see.

    The seed molecule is included for optimization tasks because the benchmark states it in
    the question text as well; withholding it here would not hide it, it would only make the
    arm read it out of prose. ``ref_smiles`` and ``answer`` are absent, which is the point.
    """
    view = {
        "id": inst.id,
        "task_type": inst.task_type,
        "question": inst.question,
    }
    if inst.task_type == "optimization" and inst.seed_smiles:
        view["seed_smiles"] = inst.seed_smiles
    return view


def load_instances(path: Path | str = DEFAULT_PATH) -> list[Instance]:
    """Read the benchmark file, in file order."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} is missing. Download it with:\n"
            "  curl -L -o data/external/moldesignbench.jsonl "
            "https://huggingface.co/datasets/LG-AI-Research/MolDesignBench/"
            "resolve/main/data/moldesignbench.jsonl"
        )
    out: list[Instance] = []
    seen: dict[str, int] = {}
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = str(row.get("id", ""))
            ordinal = seen.get(rid, 0)
            seen[rid] = ordinal + 1
            out.append(Instance.from_row(row, ordinal=ordinal))
    return out


def stratified_sample(
    instances: list[Instance], n: int, *, seed: int = 0
) -> list[Instance]:
    """``n`` instances, split evenly across tasks and holding the infeasible share.

    Two strata, because both decide a headline number. The task split is even because the
    paper reports generation and optimization separately and a subset weighted toward one of
    them is not comparable with either column. The infeasible share is held at the full set's
    10% because infeasible accuracy is a column of its own, and a sample that drifted to 4%
    would report it from eight instances while looking like a measurement over 200.

    Sampling is seeded and sorted before drawing, so the same ``(n, seed)`` names the same
    instances on any machine and a resumed run scores the cells it already has.
    """
    if n >= len(instances):
        return list(instances)

    rng = random.Random(seed)
    # An odd n used to sample n-1 and report n. The remainder goes to the first task
    # rather than being dropped, so `--n 51` draws 51.
    base, extra = divmod(n, len(TASK_TYPES))
    picked: list[Instance] = []

    for position, task in enumerate(TASK_TYPES):
        per_task = max(1, base + (1 if position < extra else 0))
        pool = [i for i in instances if i.task_type == task]
        infeasible = sorted([i for i in pool if i.infeasible], key=lambda i: i.uid)
        feasible = sorted([i for i in pool if not i.infeasible], key=lambda i: i.uid)
        share = len(infeasible) / len(pool) if pool else 0.0
        n_inf = min(len(infeasible), round(per_task * share))
        n_feas = min(len(feasible), per_task - n_inf)
        picked += rng.sample(infeasible, n_inf) + rng.sample(feasible, n_feas)

    picked.sort(key=lambda i: i.uid)
    return picked


def fingerprint(instances: list[Instance]) -> str:
    """A short hash of exactly which instances were scored.

    Two runs reporting the same success rate over different subsets are not comparable, and
    nothing else in the output would say so. This travels with every result.
    """
    h = hashlib.sha256("\n".join(sorted(i.id for i in instances)).encode()).hexdigest()
    return h[:12]


def describe(instances: list[Instance]) -> dict:
    """The composition of a sample, for the provenance block on the board."""
    return {
        "n": len(instances),
        "fingerprint": fingerprint(instances),
        "by_task": {
            t: sum(1 for i in instances if i.task_type == t) for t in TASK_TYPES
        },
        "infeasible": sum(1 for i in instances if i.infeasible),
        "mean_constraints": (
            round(sum(i.n_constraints for i in instances) / len(instances), 2)
            if instances
            else 0.0
        ),
    }


__all__ = [
    "DEFAULT_PATH",
    "TASK_TYPES",
    "Instance",
    "describe",
    "fingerprint",
    "load_instances",
    "prompt_view",
    "stratified_sample",
]
