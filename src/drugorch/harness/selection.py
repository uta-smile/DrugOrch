"""Why a budget exists when the compute is free.

Every number this framework reports about a molecule comes from an oracle that is wrong.
Vina's correlation with measured affinity is weak -- published error is on the order of
2 kcal/mol, comparable to the whole spread a campaign is trying to resolve -- and the same
is true, to different degrees, of every scoring function anyone screens with.

That changes what searching harder buys you, and the change is not intuitive. Selecting the
best molecule *by measured score* out of N candidates does two things at once: it finds
molecules that are genuinely good, and it finds molecules whose error happens to point the
right way. The second effect grows with N and the first does not, because the pool's real
ceiling is fixed by the chemistry in it. Past some point, extra sampling is a search for
adversarial examples against the scoring function.

This module measures that, so the claim on the Goals page is a computation and not a
rhetorical flourish. The model is deliberately the simplest one that shows the effect:

    true  ~ Normal(center, spread)        the affinity the molecule really has
    noise ~ Normal(0, noise)              the oracle's error, independent per molecule
    measured = true + noise               what a screen actually sees

and then: pick ``argmin(measured)``, which is what a screen does, and report what
``true`` that molecule actually had.

**Sign convention.** Everything here is kcal/mol as a docking program prints it, so *more
negative is better* and "best" means minimum. This is the opposite of the project's internal
``score`` field, which negates Vina's output so that higher is better -- :mod:`drugorch.types`
owns that convention, and mixing the two is exactly the mistake that makes a comparison table
report each arm's worst molecule as its best. The field names here all say ``kcal`` or
``true``/``measured`` rather than ``score`` to keep the two from being confused.

The three quantities worth reading, per pool size:

``best_measured``
    The headline number a screen would report. Improves without limit as N grows.
``true_of_pick``
    What the winning molecule is actually worth. Improves much more slowly.
``best_true_available``
    The best molecule that was *in the pool*. The ceiling honest search could reach.

and the difference between the last two is ``regret`` -- how much better you could have done
with the molecules you already had. It *rises* with N, which is the genuinely surprising part:
more search makes a noisy oracle worse at identifying the best thing in its own pool, because
there are more mediocre molecules available to get lucky and outrank it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Pool sizes spanning "a careful run" to "why not leave it going overnight". Powers of ten
# because the effect is logarithmic in N: the interesting comparison is 10 vs 100,000, and
# a linear sweep would spend every point in the flat part of the curve.
DEFAULT_SIZES: tuple[int, ...] = (10, 100, 1_000, 10_000, 100_000)

# Roughly Vina's published affinity RMSE. Quoted from the literature, not measured here --
# which is precisely why the API lets a reader change it. The argument for budgeting stands
# or falls on this number, so it should be the thing you can turn.
DEFAULT_NOISE = 2.0

# The spread of real affinity across the pool. 1 kcal/mol is a narrow, realistic range for
# analogues of one series; wider pools make the search look better, not worse.
DEFAULT_SPREAD = 1.0

DEFAULT_CENTER = -8.0

# Chosen by measurement, not by feel: see `selection_curve`. Enough that the curve is smooth
# at this scale without making the endpoint feel broken.
DEFAULT_TRIALS = 400

# One trial per draw, deliberately, rather than one `trials x n` array or a tuned block size.
#
# The obvious implementation allocates the whole sweep at once: at the largest pool that is
# 4 x 10^7 floats, 320 MB, the biggest allocation in the process. The obvious fix is to chunk
# it, and the obvious chunk size is a tuning constant -- at which point the published figure
# depends on that constant, because drawing (64, n) and drawing (1, n) sixty-four times
# consume the generator differently and give different numbers. Identical statistics, but not
# the same table, so anyone who lowered the constant for a smaller machine would silently get
# a figure that no longer matched the one on the page.
#
# Drawing one trial at a time removes the knob rather than documenting it. Peak memory is
# O(n) instead of O(trials x n), and the cost is a few hundred extra numpy calls, which is
# microseconds against the arithmetic they carry.


@dataclass(frozen=True)
class SelectionRow:
    """One pool size: what the screen reported, and what it actually picked."""

    n: int
    best_measured: float
    true_of_pick: float
    best_true_available: float

    @property
    def regret(self) -> float:
        """How much worse the pick is than the best molecule that was in the pool.

        Zero for a perfect oracle. Positive and *growing* in ``n`` for a noisy one.
        """
        return self.true_of_pick - self.best_true_available

    @property
    def inflation(self) -> float:
        """How much the reported number overstates what was found.

        ``best_measured`` is the figure that goes in a slide; ``true_of_pick`` is what the
        molecule is worth. The difference is not a bias that cancels across arms -- it grows
        with how hard each arm searched, so an arm with a bigger budget looks better than an
        equally good arm with a smaller one.
        """
        return self.true_of_pick - self.best_measured

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "best_measured": round(self.best_measured, 3),
            "true_of_pick": round(self.true_of_pick, 3),
            "best_true_available": round(self.best_true_available, 3),
            "regret": round(self.regret, 3),
            "inflation": round(self.inflation, 3),
        }


@dataclass(frozen=True)
class SelectionCurve:
    """The whole sweep, with the assumptions that produced it attached.

    The parameters travel with the numbers on purpose. A table of this shape with no stated
    noise level is unreadable -- at ``noise=0`` every row is identical and the entire argument
    for budgeting evaporates, which is a fact worth being able to check rather than take on
    trust.
    """

    rows: list[SelectionRow] = field(default_factory=list)
    noise: float = DEFAULT_NOISE
    spread: float = DEFAULT_SPREAD
    center: float = DEFAULT_CENTER
    trials: int = DEFAULT_TRIALS
    seed: int = 0

    def as_dict(self) -> dict:
        return {
            "rows": [r.as_dict() for r in self.rows],
            "assumptions": {
                "noise": self.noise,
                "spread": self.spread,
                "center": self.center,
                "trials": self.trials,
                "seed": self.seed,
            },
        }


def selection_curve(
    sizes: tuple[int, ...] | list[int] = DEFAULT_SIZES,
    *,
    noise: float = DEFAULT_NOISE,
    spread: float = DEFAULT_SPREAD,
    center: float = DEFAULT_CENTER,
    trials: int = DEFAULT_TRIALS,
    seed: int = 0,
) -> SelectionCurve:
    """Average, over ``trials`` independent pools, what best-by-measured-score selects.

    Deterministic in ``seed``: the figure on the page is reproducible, and a reader who
    reruns it gets the table they were shown rather than one that merely looks similar.

    ``noise=0`` is a meaningful input, not a degenerate one -- it is the control. With a
    perfect oracle the pick is always the best molecule in the pool, regret is exactly zero
    at every size, and nothing here argues for a budget.
    """
    if not sizes:
        raise ValueError("sizes must not be empty")
    if any(n < 1 for n in sizes):
        raise ValueError("every pool size must be at least 1")
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if noise < 0 or spread < 0:
        raise ValueError("noise and spread are standard deviations and cannot be negative")

    rng = np.random.default_rng(seed)
    rows: list[SelectionRow] = []

    for n in sizes:
        sum_measured = sum_true_pick = sum_true_best = 0.0
        for _ in range(trials):
            true = rng.normal(center, spread, size=n)
            measured = true + rng.normal(0.0, noise, size=n) if noise else true

            pick = int(measured.argmin())
            sum_measured += float(measured[pick])
            sum_true_pick += float(true[pick])
            sum_true_best += float(true.min())

        rows.append(
            SelectionRow(
                n=n,
                best_measured=sum_measured / trials,
                true_of_pick=sum_true_pick / trials,
                best_true_available=sum_true_best / trials,
            )
        )

    return SelectionCurve(
        rows=rows, noise=noise, spread=spread, center=center, trials=trials, seed=seed
    )
