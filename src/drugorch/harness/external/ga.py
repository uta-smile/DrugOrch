"""GraphGA against a constraint set.

The operators are DrugOrch's own: :func:`drugorch.subsystems.generation.crossover` and
:func:`~drugorch.subsystems.generation.mutate`, unchanged. What differs from a target run is
the fitness. A target run scores children by docking them; here there is no protein, and a
child is scored by how much of the design spec it satisfies.

Two choices decide whether this works at all.

**Satisfaction leads, distance breaks ties.** :attr:`Score.fitness` counts satisfied
constraints first and uses normalised distance only to order molecules with the same count.
A fitness that was purely distance trades a satisfied constraint for a small gain on an
unsatisfied one, and the benchmark scores success as every constraint at once.

**Measurement is batched per round.** Each round measures its whole population in one call
to the benchmark's analyzer, which costs about a second for twenty-odd molecules against five
seconds per call of model-loading overhead if they went one at a time. The round structure
exists for that reason as much as for the genetics.

Elites are carried forward unchanged, so a round can never lose the best molecule found so
far, and the loop stops the moment a child satisfies everything: there is nothing above
success to climb toward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...subsystems.generation import crossover, mutate
from .panel import Measurement, PanelWorker
from .spec import DesignSpec, Score


@dataclass
class Candidate:
    smiles: str
    score: Score
    measurement: Measurement | None = None
    origin: str = "seed"
    similarity: float | None = None
    """Tanimoto to the anchor molecule, on optimization instances. None when unanchored."""

    @property
    def fitness(self) -> float:
        """Satisfaction first, then anchoring, then distance.

        The anchor term is a step rather than a gradient: a molecule that is still an edit
        of the lead outranks one that is not, and among molecules on the same side of the
        line it plays no part. A smooth penalty would trade similarity against constraints
        continuously, which is how a search talks itself from an edit into a replacement
        one small step at a time.
        """
        base = self.score.fitness
        if self.similarity is not None and self.similarity < ANCHOR_FLOOR:
            return base - 1000.0
        return base


@dataclass
class GAResult:
    """What a generation run produced."""

    best: Candidate | None = None
    population: list[Candidate] = field(default_factory=list)
    rounds: int = 0
    evaluated: int = 0
    solved: bool = False
    history: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        if self.best is None:
            return f"{self.rounds} rounds, {self.evaluated} molecules measured, none valid"
        s = self.best.score
        head = (
            f"{self.rounds} rounds, {self.evaluated} molecules measured. "
            f"Best satisfies {s.satisfied}/{s.total}"
        )
        if s.misses:
            head += f"; still missing: {'; '.join(s.misses[:4])}"
        return head


ANCHOR_FLOOR = 0.4
"""Tanimoto below which a molecule stops counting as an edit of the lead it came from.

Used only on optimization instances, where the question supplies a molecule and asks for it
to be improved. Measured on the first run of this benchmark: answers below 0.3 similarity to
the seed were 37 of 85, against 16 of 82 for the no-tool control, and the orchestrated arm
scored *below* its own control on that task while satisfying more constraints on average.
That is the signature of answering a different question well. 0.4 is one step above the
0.3 line that separated the two arms, chosen so a molecule has to stay recognisably the
lead rather than merely resemble it."""


def optimize(
    seeds: list[str],
    spec: DesignSpec,
    panel: PanelWorker,
    *,
    rounds: int = 6,
    population: int = 24,
    elites: int = 8,
    mutation_rate: float = 0.5,
    seed: int = 0,
    anchor: str | None = None,
) -> GAResult:
    """Evolve ``seeds`` toward ``spec``, measuring every generation.

    ``anchor`` is the lead molecule an optimization instance supplies. When given, a child
    that has drifted below :data:`ANCHOR_FLOOR` similarity to it is ranked below every
    child that has not, however well it scores. The benchmark does not measure similarity
    to the seed, so this is not scored directly; what it protects is the fragment
    constraints, which are read off the seed and are lost first when the search wanders.

    Returns the best molecule found and the population behind it. A seed that cannot be
    parsed is dropped rather than raising: seeds arrive from a language model and a fraction
    of them are not molecules.
    """
    rng = np.random.default_rng(seed)
    result = GAResult()

    def evaluate(smiles: list[str], origin: str) -> list[Candidate]:
        fresh = [s for s in dict.fromkeys(smiles) if s]
        if not fresh:
            return []
        measured = panel.measure(fresh)
        result.evaluated += len(fresh)
        out = []
        for s in fresh:
            m = measured.get(s)
            if m is None or not m.valid:
                continue
            out.append(Candidate(smiles=s, score=spec.score(m), measurement=m, origin=origin))
        if anchor and out:
            # One fingerprint pass per round rather than per molecule: the similarity is
            # only used for ordering, so it is attached to the candidate and read by
            # `fitness` rather than folded into the score the arm reports.
            from ...fingerprints import max_similarity

            sims = max_similarity([c.smiles for c in out], [anchor])
            for c in out:
                c.similarity = sims.get(c.smiles, 0.0)
        return out

    pop = evaluate(seeds, "seed")
    if not pop:
        return result

    pop.sort(key=lambda c: -c.fitness)
    result.best = pop[0]
    result.population = pop[:population]
    result.history.append(
        {"round": 0, "best": pop[0].fitness, "satisfied": pop[0].score.satisfied,
         "n": len(pop)}
    )
    if pop[0].score.solved:
        result.solved = True
        return result

    for r in range(1, rounds + 1):
        parents = result.population[: max(2, elites)]
        children: list[str] = []
        # Every operator can fail on a given parent, and a degenerate parent set can fail
        # on all of them: `mutate` returns None outright for a molecule no reaction
        # matches. Bounding the attempts rather than the failures is what keeps that from
        # spinning forever, which a round-number guard could not do on the first round or
        # once any one child existed.
        attempts = 0
        while len(children) < population and attempts < 4 * population:
            attempts += 1
            if len(parents) >= 2 and rng.random() > mutation_rate:
                a, b = rng.choice(len(parents), size=2, replace=False)
                op = crossover(parents[int(a)].smiles, parents[int(b)].smiles, rng)
            else:
                a = int(rng.integers(len(parents)))
                op = mutate(parents[a].smiles, rng)
            if op is None:
                continue
            children.append(op.smiles)

        if not children:
            break
        scored = evaluate(children, f"round {r}")
        merged = {c.smiles: c for c in result.population + scored}
        pop = sorted(merged.values(), key=lambda c: -c.fitness)
        result.population = pop[:population]
        result.best = pop[0]
        result.rounds = r
        result.history.append(
            {"round": r, "best": pop[0].fitness, "satisfied": pop[0].score.satisfied,
             "n": len(scored)}
        )
        if pop[0].score.solved:
            result.solved = True
            break

    return result


__all__ = ["Candidate", "GAResult", "optimize"]
