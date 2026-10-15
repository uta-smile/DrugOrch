"""Decomposing a success rate into reading the question and solving what was read.

A benchmark that reports one number per system says which is ahead and nothing about why.
MolDesignBench states some constraints outright and implies the rest through the scenario,
so failing an instance has two quite different causes: the specification the agent worked
from was incomplete, or it was complete and the chemistry did not get there. Those call for
opposite fixes, and the published leaderboard cannot distinguish them.

DrugOrch can, because its arm writes down the constraint set it believes it was asked for
before it searches. That written-down set is an object, and after the evaluator has done its
work it can be compared with the key.

**This comparison happens only after scoring.** The answer key is not in the process that
answers instances, and `tests/test_external.py::test_no_prompt_can_carry_the_answer_key`
asserts it. A diagnostic computed from the key is a statement about the run that has already
finished, not an input to it. The distinction is the whole reason the number is safe to
report: used the other way round it would be the leak that invalidates everything.

Two quantities come out of it:

**Spec recall** is how much of the key's constraint set the arm's reading captured. It is
counted rather than matched by name, because a window the arm wrote tighter than the key is
still a constraint it saw, and because the key's names and the arm's vocabulary agree only
for the properties both measure.

**Premature confidence** is how often the arm believed it had satisfied its own
specification and the evaluator disagreed. On the first run of this benchmark that was 82 of
174 feasible instances, which is what sent the next change into verification rather than
into more tools.
"""

from __future__ import annotations

import json
from pathlib import Path

from .agent import Attempt
from .instances import Instance


def key_size(inst: Instance) -> int:
    answer = inst.answer or {}
    props = len(answer.get("properties") or [])
    frags = sum(len(d) for d in (answer.get("fragments") or []) if isinstance(d, dict))
    return props + frags


def decompose(
    attempts: list[Attempt], sample: list[Instance], scored_path: Path | str
) -> dict:
    """Where this arm's score went, over the feasible instances it answered.

    Returns an empty dict when the scored file is missing, rather than a dict of zeros: a
    diagnostic nobody can compute is not a diagnostic reading zero.
    """
    path = Path(scored_path)
    if not path.is_file() or not attempts:
        return {}

    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    solved = {str(r.get("id")): bool(r.get("total_acc")) for r in rows}
    by_id = {a.instance_id: a for a in attempts}

    n = 0
    believed_and_right = believed_and_wrong = 0
    unsure_and_right = unsure_and_wrong = 0
    extracted = held = 0
    under = 0
    verified = 0
    calls_right: list[int] = []
    calls_wrong: list[int] = []

    for inst in sample:
        att = by_id.get(inst.uid)
        if att is None or att.n_fallbacks or inst.infeasible or inst.uid not in solved:
            continue
        n += 1
        believed = att.best_total > 0 and att.best_satisfied == att.best_total
        right = solved[inst.uid]
        if believed and right:
            believed_and_right += 1
        elif believed:
            believed_and_wrong += 1
        elif right:
            unsure_and_right += 1
        else:
            unsure_and_wrong += 1

        (calls_right if right else calls_wrong).append(att.n_tool_calls)

        k = key_size(inst)
        extracted += att.best_total
        held += k
        under += int(att.best_total < k)
        verified += sum(1 for d in att.decisions if d.get("action") == "verify")

    if not n:
        return {}

    declared = believed_and_right + believed_and_wrong
    return {
        "n_feasible": n,
        "believed_and_right": believed_and_right,
        "believed_and_wrong": believed_and_wrong,
        "unsure_and_right": unsure_and_right,
        "unsure_and_wrong": unsure_and_wrong,
        # How often "I am finished" was true. The headline of this decomposition.
        "confidence_precision": round(believed_and_right / declared, 4) if declared else 0.0,
        "spec_recall": round(extracted / held, 4) if held else 0.0,
        "constraints_extracted": round(extracted / n, 2),
        "constraints_in_key": round(held / n, 2),
        "under_extracted": under,
        "verifications": verified,
        "mean_calls_when_solved": round(sum(calls_right) / len(calls_right), 2)
        if calls_right
        else 0.0,
        "mean_calls_when_failed": round(sum(calls_wrong) / len(calls_wrong), 2)
        if calls_wrong
        else 0.0,
    }


__all__ = ["decompose", "key_size"]
