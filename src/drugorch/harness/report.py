"""The leaderboard: how the arms actually compare.

Aggregation is **paired by (target, seed) against a baseline arm**, not averaged
independently. Arms share a seed and a holdout by construction, so a paired comparison
removes the enormous variance between targets -- JAK2 and ADRB2 differ far more from each
other than two policies differ on either -- and asks the only question worth asking: *on the
same problem, did this arm beat the baseline?*

An unpaired mean across targets would be dominated by which targets happen to be easy, and
with three seeds it would also be meaningless. The win rate and the paired delta are
reported together because either alone can mislead: an arm can win narrowly on every target
(high win rate, small delta) or lose everywhere except one blowout (low win rate, positive
mean delta), and those are different findings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .metrics import RunMetrics

PRIMARY = "discovery_auc"
BASELINE = "fixed"

# Rendered on the leaderboard, so these follow docs/STYLE.md: a statement of direction, then
# what the metric counts. The double hyphen they used to carry was an em dash in disguise.
RANKED_BY = {
    "discovery_auc": "higher is better. Held-out scaffolds found, weighted toward finding them early",
    "rediscovery": "higher is better. Fraction of the held-out set reached at all",
    "best_score": "higher is better. Docking only, and a weak endpoint on its own",
    "score_auc": "higher is better. How early an arm reached its own best score",
}


@dataclass(slots=True)
class ArmSummary:
    """One policy's row on the leaderboard."""

    policy: str
    n_cells: int = 0
    n_targets: int = 0
    primary: float = 0.0
    primary_sd: float = 0.0
    rediscovery: float = 0.0
    best_score: float | None = None
    score_auc: float = 0.0
    novelty: float = 0.0
    diversity: float = 0.0
    sa_mean: float | None = None
    spent: float = 0.0
    steps: float = 0.0

    # paired against the baseline
    wins: int = 0
    losses: int = 0
    ties: int = 0
    mean_delta: float = 0.0
    delta_ci: tuple[float, float] = (0.0, 0.0)

    flags: list[str] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        n = self.wins + self.losses + self.ties
        return round(self.wins / n, 3) if n else 0.0

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "n_cells": self.n_cells,
            "n_targets": self.n_targets,
            "primary": round(self.primary, 4),
            "primary_sd": round(self.primary_sd, 4),
            "rediscovery": round(self.rediscovery, 4),
            "best_score": self.best_score,
            "score_auc": round(self.score_auc, 4),
            "novelty": round(self.novelty, 4),
            "diversity": round(self.diversity, 4),
            "sa_mean": self.sa_mean,
            "spent": round(self.spent, 2),
            "steps": round(self.steps, 1),
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "win_rate": self.win_rate,
            "mean_delta": round(self.mean_delta, 4),
            "delta_ci": [round(self.delta_ci[0], 4), round(self.delta_ci[1], 4)],
            "flags": self.flags,
        }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _sd(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _bootstrap_ci(xs: list[float], *, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """A 95% percentile bootstrap on the paired deltas.

    Bootstrap rather than a t-interval because with three or four seeds the normality
    assumption is doing all the work and none of it is earned. With very few points the
    interval comes out wide, which is the honest outcome.
    """
    if len(xs) < 2:
        return (0.0, 0.0)
    import random

    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [xs[rng.randrange(len(xs))] for _ in range(len(xs))]
        means.append(_mean(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means)) - 1]
    return (lo, hi)


def leaderboard(
    rows: list[RunMetrics],
    *,
    metric: str = PRIMARY,
    baseline: str = BASELINE,
) -> list[ArmSummary]:
    """Rank the arms, paired against ``baseline`` on each (target, seed)."""
    by_policy: dict[str, list[RunMetrics]] = {}
    for r in rows:
        by_policy.setdefault(r.policy, []).append(r)

    base_by_cell = {
        (r.target, r.seed): getattr(r, metric) or 0.0
        for r in by_policy.get(baseline, [])
    }

    out: list[ArmSummary] = []
    for policy, cells in by_policy.items():
        s = ArmSummary(policy=policy, n_cells=len(cells))
        s.n_targets = len({c.target for c in cells})
        vals = [float(getattr(c, metric) or 0.0) for c in cells]
        s.primary = _mean(vals)
        s.primary_sd = _sd(vals)
        s.rediscovery = _mean([c.rediscovery for c in cells])
        s.score_auc = _mean([c.score_auc for c in cells])
        s.novelty = _mean([c.novelty for c in cells])
        s.diversity = _mean([c.diversity for c in cells])
        s.spent = _mean([c.spent for c in cells])
        s.steps = _mean([float(c.steps) for c in cells])

        scored = [c.best_score for c in cells if c.best_score is not None]
        s.best_score = round(_mean(scored), 4) if scored else None
        sas = [c.sa_mean for c in cells if c.sa_mean is not None]
        s.sa_mean = round(_mean(sas), 3) if sas else None

        deltas = []
        for c in cells:
            base = base_by_cell.get((c.target, c.seed))
            if base is None:
                continue
            d = float(getattr(c, metric) or 0.0) - base
            deltas.append(d)
            if d > 1e-9:
                s.wins += 1
            elif d < -1e-9:
                s.losses += 1
            else:
                s.ties += 1
        s.mean_delta = _mean(deltas)
        s.delta_ci = _bootstrap_ci(deltas)

        # Flags that should stop a row being read as a clean result.
        fallbacks = sum(c.n_fallbacks for c in cells)
        if fallbacks:
            s.flags.append(f"{fallbacks} fallback decision(s) across cells")
        if policy == "oracle":
            s.flags.append("reads the answer key -- a ceiling, not a method")
        if s.n_cells < 3:
            s.flags.append(f"only {s.n_cells} cell(s); treat as anecdote")
        if all(c.rediscovery == 0.0 for c in cells) and cells:
            s.flags.append("rediscovered nothing; the primary metric is uninformative here")

        out.append(s)

    out.sort(key=lambda a: (-a.primary, a.policy))
    return out


def console_table(board: list[ArmSummary], *, metric: str = PRIMARY) -> str:
    """The leaderboard as text, for the CLI."""
    if not board:
        return "no rows"
    head = (
        f"{'policy':<12} {'cells':>5} {'tgts':>4} {metric:>13} "
        f"{'redisc':>7} {'best':>7} {'novel':>6} {'vs fixed':>9} {'win%':>5}"
    )
    lines = [head, "-" * len(head)]
    for a in board:
        best = f"{a.best_score:.2f}" if a.best_score is not None else "   -"
        delta = f"{a.mean_delta:+.3f}"
        lines.append(
            f"{a.policy:<12} {a.n_cells:>5} {a.n_targets:>4} {a.primary:>13.4f} "
            f"{a.rediscovery:>7.3f} {best:>7} {a.novelty:>6.2f} {delta:>9} "
            f"{a.win_rate * 100:>4.0f}%"
        )
    for a in board:
        for f in a.flags:
            lines.append(f"  ! {a.policy}: {f}")
    return "\n".join(lines)


def per_target(rows: list[RunMetrics], *, metric: str = PRIMARY) -> dict:
    """The breakdown behind the leaderboard: one number per (target, policy).

    An aggregate hides the case the whole benchmark exists to find -- an arm that wins on
    average because it is excellent on kinases and useless on everything else.
    """
    out: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        out.setdefault(r.target, {}).setdefault(r.policy, []).append(
            float(getattr(r, metric) or 0.0)
        )
    return {t: {p: round(_mean(v), 4) for p, v in by_p.items()} for t, by_p in out.items()}


__all__ = [
    "BASELINE",
    "PRIMARY",
    "RANKED_BY",
    "ArmSummary",
    "console_table",
    "leaderboard",
    "per_target",
]
