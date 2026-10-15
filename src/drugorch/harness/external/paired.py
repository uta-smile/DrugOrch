"""Comparing two arms on the instances they both answered.

The board's columns are enough to rank arms and not enough to attribute a difference. Two
success rates over the same benchmark can differ because one arm is better, or because the
sample is small and the instances it happened to solve were the easy ones. With a hundred
instances and success rates in the tens, that second explanation is not a remote possibility.

So the orchestrated arm is compared against its own control the way the routing benchmark
compares policies: **paired per instance**. Both arms answered the same questions, so the
question becomes how many instances each solved that the other did not. Instances both
solved and instances neither solved carry no information about which is better and drop out,
which is what McNemar's test formalises.

The test is the exact binomial on the discordant pairs rather than the chi-squared
approximation. With twenty discordant pairs the approximation is doing work it has not
earned, and the exact version costs nothing at this size.

This says nothing about whether the arm beats a published row. Those were measured on
different instances by other people, and no paired test is available across that boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb


@dataclass
class PairedGain:
    """One arm against another, on the instances both answered."""

    arm: str
    baseline: str
    scope: str
    n_paired: int = 0
    both: int = 0
    arm_only: int = 0
    baseline_only: int = 0
    neither: int = 0
    p_value: float = 1.0

    @property
    def delta(self) -> int:
        """Instances the arm solved that the baseline did not, net."""
        return self.arm_only - self.baseline_only

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "baseline": self.baseline,
            "scope": self.scope,
            "n_paired": self.n_paired,
            "both": self.both,
            "arm_only": self.arm_only,
            "baseline_only": self.baseline_only,
            "neither": self.neither,
            "delta": self.delta,
            "p_value": round(self.p_value, 5),
        }


def exact_mcnemar(arm_only: int, baseline_only: int) -> float:
    """Two-sided exact binomial p over the discordant pairs.

    Under the null the two arms are equally likely to be the one that solved a discordant
    instance, so the count follows a binomial with p = 0.5. No discordant pairs means no
    evidence either way, which is a p of 1 rather than a division by zero.
    """
    n = arm_only + baseline_only
    if n == 0:
        return 1.0
    k = min(arm_only, baseline_only)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(
    verdicts: dict[str, dict[str, bool]],
    *,
    arm: str,
    baseline: str,
    scope: str = "overall",
) -> PairedGain | None:
    """Pair two arms' per-instance verdicts.

    ``verdicts`` maps an arm to ``{instance_id: solved}``, as the benchmark's evaluator
    reports it. An instance missing from either arm is dropped: an arm that never answered
    it has no verdict to pair, and inventing one would be the same mistake as scoring a
    fallback as an abstention.
    """
    a, b = verdicts.get(arm), verdicts.get(baseline)
    if not a or not b:
        return None

    shared = set(a) & set(b)
    if not shared:
        return None

    g = PairedGain(arm=arm, baseline=baseline, scope=scope, n_paired=len(shared))
    for i in shared:
        if a[i] and b[i]:
            g.both += 1
        elif a[i]:
            g.arm_only += 1
        elif b[i]:
            g.baseline_only += 1
        else:
            g.neither += 1
    g.p_value = exact_mcnemar(g.arm_only, g.baseline_only)
    return g


def read_verdicts(path) -> dict[str, dict[str, bool]]:
    """Per-instance verdicts from a scored file, keyed by task and by instance.

    The benchmark's evaluator writes its judgement back into the file it read, one
    ``total_acc`` per instance. Reading it back is how this module avoids deciding for
    itself what counts as solved.
    """
    import json
    from pathlib import Path

    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, bool]] = {"overall": {}}
    for r in rows:
        task = str(r.get("task_type") or "overall")
        out.setdefault(task, {})[str(r.get("id"))] = bool(r.get("total_acc"))
        out["overall"][str(r.get("id"))] = bool(r.get("total_acc"))
    return out


__all__ = ["PairedGain", "compare", "exact_mcnemar", "read_verdicts"]
