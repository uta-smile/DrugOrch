"""What a run is scored on.

One headline number and a set that keeps it honest. The headline is **discovery AUC**: the
area under "fraction of held-out scaffolds rediscovered, against budget spent", normalised to
0..1. It is an area rather than an endpoint because a run that finds everything on its last
unit and one that finds it immediately are not equally good, and an endpoint cannot tell them
apart.

The rest exist because a single number is gameable and this one is no exception:

* **Docking score alone is a weak endpoint.** Generators learn to exploit scoring-function
  artefacts, which is why `best_score` is reported but is not the ranking metric.
* **A run that wins by re-proposing its own seeds has not discovered anything**, so novelty
  and scaffold count are reported beside it.
* **A molecule that cannot be made is not a hit**, so SA score and PAINS rate travel with
  every row.
* **Rediscovery is scaffold-level with a similarity floor**, not exact match. Exact-SMILES
  rediscovery is close to unmeasurable at demo budgets and would report zero everywhere,
  which looks like rigour and is actually just a broken instrument.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..fingerprints import matrix, max_similarity, mean_pairwise_distance, tanimoto

REDISCOVERY_SIMILARITY = 0.7
"""Tanimoto at which a proposed molecule counts as having found a held-out one.

0.7 on ECFP4 is the conventional "same chemotype, plausibly the same series" line. Stated as
a constant because the number materially changes every rediscovery figure and should be
argued with rather than buried."""

HIT_THRESHOLD = 9.0
"""Docking score counted as a hit, for cost-to-first-hit. Arbitrary but fixed; it is used
only to compare arms against each other, never as a claim about real potency."""


@dataclass(slots=True)
class RunMetrics:
    """One cell of the benchmark: one arm, one target, one seed."""

    run_id: str = ""
    target: str = ""
    policy: str = ""
    seed: int = 0

    # --- the headline ---
    discovery_auc: float = 0.0
    rediscovery: float = 0.0
    n_rediscovered: int = 0
    n_held_out_scaffolds: int = 0

    # --- the docking endpoint ---
    best_score: float | None = None
    score_auc: float = 0.0
    cost_to_first_hit: float | None = None

    # --- did it actually discover anything, or just repeat itself ---
    novelty: float = 0.0
    n_scaffolds: int = 0
    diversity: float = 0.0

    # --- would a chemist accept these molecules ---
    sa_mean: float | None = None
    pains_rate: float | None = None

    # --- was this even a clean run of the arm it claims to be ---
    n_molecules: int = 0
    n_scored: int = 0
    steps: int = 0
    spent: float = 0.0
    budget: float = 0.0
    n_fallbacks: int = 0
    status: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _auc(curve: list[tuple[float, float]], total: float) -> float:
    """Normalised area under a step curve of (spend, value), value already in 0..1.

    Trapezoid over spend, divided by the total budget, so an arm that spends less than its
    cap is not rewarded for the unspent remainder -- the area it did not earn stays empty.
    """
    if not curve or total <= 0:
        return 0.0
    pts = sorted(curve)
    area = 0.0
    prev_x, prev_y = 0.0, 0.0
    for x, y in pts:
        area += (x - prev_x) * prev_y
        prev_x, prev_y = x, y
    area += (total - prev_x) * prev_y  # it holds its final value to the cap
    return round(max(0.0, min(1.0, area / total)), 4)


def rediscovery_curve(
    molecules: list[tuple[str, float]],
    held_out: list[str],
    *,
    threshold: float = REDISCOVERY_SIMILARITY,
) -> tuple[list[tuple[float, float]], int]:
    """How much of the held-out set had been reached, as a function of spend.

    ``molecules`` is ``(smiles, spend_when_first_seen)``. Scaffold-level: a held-out molecule
    counts as reached once *any* proposal is within ``threshold`` of it, and each held-out
    molecule is counted at most once -- twenty analogues of one compound are one discovery.

    One similarity matrix for the whole comparison. The obvious nested-loop version is
    quadratic in similarity calls and would dominate the benchmark's runtime; here the
    all-pairs matrix is computed once and everything else is indexing.
    """
    if not held_out or not molecules:
        return [], 0

    proposed = [s for s, _ in molecules]
    spends = np.array([sp for _, sp in molecules], dtype=np.float64)

    hm, h_kept = matrix(held_out)
    pm, p_kept = matrix(proposed)
    if hm.size == 0 or pm.size == 0:
        return [], 0

    # Proposals may have been dropped by parsing, so realign spends to what survived.
    spend_by_smiles: dict[str, float] = {}
    for smi, sp in zip(proposed, spends, strict=True):
        spend_by_smiles[smi] = min(spend_by_smiles.get(smi, sp), sp)
    p_spends = np.array([spend_by_smiles[s] for s in p_kept], dtype=np.float64)

    sims = tanimoto(hm, pm)                      # (held_out, proposed)
    qualifies = sims >= threshold

    found_at: list[float] = []
    for row in range(qualifies.shape[0]):
        hits = np.flatnonzero(qualifies[row])
        if hits.size:
            found_at.append(float(p_spends[hits].min()))

    if not found_at:
        return [], 0

    found_at.sort()
    n = len(held_out)
    return [(sp, (i + 1) / n) for i, sp in enumerate(found_at)], len(found_at)


def score_curve(objective_curve: list[dict], total: float) -> float:
    """Best-so-far docking score as an area, normalised against the run's own best.

    Relative to the best any arm on this target achieved is what the aggregator does; within
    one run the normalisation is against its own maximum, so the number says "how early did
    it get where it got" rather than "how good was it".
    """
    pts = [(c.get("spent", 0.0), c.get("best")) for c in objective_curve if c.get("best") is not None]
    if not pts:
        return 0.0
    top = max(v for _, v in pts)
    if top <= 0:
        return 0.0
    return _auc([(x, v / top) for x, v in pts], total)


def evaluate(
    *,
    run_id: str,
    policy: str,
    target: str,
    seed: int,
    molecules: dict,
    objective_curve: list[dict],
    budget_total: float,
    spent: float,
    steps: int,
    status: str,
    held_out: list[str],
    n_held_out_scaffolds: int,
    seen: list[str],
    n_fallbacks: int = 0,
) -> RunMetrics:
    """Score one finished run.

    ``molecules`` is ``{smiles: {"origin", "spend", "scores", "props"}}`` -- the harness
    reconstructs it from the run directory so this function never touches a RunState and can
    be used on a run recorded months ago.
    """
    m = RunMetrics(
        run_id=run_id,
        target=target,
        policy=policy,
        seed=seed,
        n_held_out_scaffolds=n_held_out_scaffolds,
        n_molecules=len(molecules),
        steps=steps,
        spent=round(spent, 3),
        budget=budget_total,
        status=status,
        n_fallbacks=n_fallbacks,
    )

    # Only what the run *produced* counts as a discovery; a retrieved active is an input.
    proposed = [
        (s, float(d.get("spend") or 0.0))
        for s, d in molecules.items()
        if d.get("origin") not in ("retrieval",)
    ]

    curve, n_found = rediscovery_curve(proposed, held_out)
    m.n_rediscovered = n_found
    m.rediscovery = round(n_found / len(held_out), 4) if held_out else 0.0
    m.discovery_auc = _auc(curve, budget_total)

    scores = [
        float(d["scores"]["vina_score"])
        for d in molecules.values()
        if (d.get("scores") or {}).get("vina_score") is not None
    ]
    m.n_scored = len(scores)
    m.best_score = round(max(scores), 4) if scores else None
    m.score_auc = score_curve(objective_curve, budget_total)

    for c in sorted(objective_curve, key=lambda c: c.get("spent", 0.0)):
        if c.get("best") is not None and c["best"] >= HIT_THRESHOLD:
            m.cost_to_first_hit = round(float(c.get("spent", 0.0)), 2)
            break

    if proposed:
        smis = [s for s, _ in proposed]
        sims = max_similarity(smis, seen) if seen else dict.fromkeys(smis, 0.0)
        m.novelty = round(sum(1 for v in sims.values() if v < 0.6) / len(smis), 4)
        m.diversity = round(mean_pairwise_distance(smis), 4)
        from ..chem import murcko_scaffold

        m.n_scaffolds = len({murcko_scaffold(s) for s in smis} - {None})

        sa = [
            float((d.get("scores") or {}).get("sa_score"))
            for s, d in molecules.items()
            if (d.get("scores") or {}).get("sa_score") is not None
            and d.get("origin") not in ("retrieval",)
        ]
        if sa:
            m.sa_mean = round(sum(sa) / len(sa), 3)
        # `pains_alert`, singular -- the key the properties subsystem actually writes. The
        # plural spelling silently produced an empty column rather than an error, which is
        # the failure mode a metric should never have.
        pains = [
            (d.get("props") or {}).get("pains_alert")
            for d in molecules.values()
            if (d.get("props") or {}).get("pains_alert") is not None
        ]
        if pains:
            flagged = sum(1 for p in pains if p not in (None, "", "None", "False", False))
            m.pains_rate = round(flagged / len(pains), 4)

    if m.n_fallbacks and m.n_fallbacks >= max(1, steps // 2):
        m.notes.append(
            f"{m.n_fallbacks} of {steps} decisions fell back; this is not a clean run of {policy}"
        )
    if not held_out:
        m.notes.append("no holdout: rediscovery is meaningless for this row")
    if m.n_scored == 0:
        m.notes.append("nothing was scored")

    return m


__all__ = [
    "HIT_THRESHOLD",
    "REDISCOVERY_SIMILARITY",
    "RunMetrics",
    "evaluate",
    "rediscovery_curve",
    "score_curve",
]
