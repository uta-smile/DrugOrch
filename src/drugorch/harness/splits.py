"""Holding actives back, so a run can be asked to find something it was never shown.

Every arm is handed the ChEMBL actives by retrieval, which means the "answer key" is an
input, not an answer. Nothing is held out, so nothing can be rediscovered, so there is no
retrospective claim to make. This fixes that.

**The split is by scaffold, not by molecule.** A random split leaves close analogues of every
held-out compound in the training half, so "rediscovery" degenerates into "proposed something
one methyl away from a molecule it was given" -- which measures interpolation and reports it
as discovery. Holding out whole Bemis-Murcko families means a hit requires reaching a
chemotype the run has not seen.

A time split would be stronger still -- train on what was known in 2015, test on what was
published after -- but ChEMBL's activity records carry no document date or year through the
client, so it is not available without new API fields. Said plainly rather than implied.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..chem import canonical, murcko_scaffold

DEFAULT_HOLDOUT = 0.3


@dataclass(frozen=True, slots=True)
class Split:
    """One target's actives, divided once and reused by every arm."""

    target: str
    seen: tuple[str, ...]
    held_out: tuple[str, ...]
    held_out_scaffolds: frozenset[str]
    seen_scaffolds: frozenset[str]
    fraction: float
    seed: int
    unscaffoldable: int = 0
    """Molecules with no computable Murcko scaffold. Kept in `seen`: they cannot be part of
    a scaffold-level claim either way, and discarding them would quietly shrink the pool."""

    meta: dict = field(default_factory=dict)

    @property
    def n_held_out_scaffolds(self) -> int:
        return len(self.held_out_scaffolds)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "n_seen": len(self.seen),
            "n_held_out": len(self.held_out),
            "n_held_out_scaffolds": self.n_held_out_scaffolds,
            "n_seen_scaffolds": len(self.seen_scaffolds),
            "fraction": self.fraction,
            "seed": self.seed,
            "unscaffoldable": self.unscaffoldable,
            "fingerprint": self.fingerprint(),
            **self.meta,
        }

    def fingerprint(self) -> str:
        """A short hash of exactly which molecules were withheld.

        Two arms compared against different holdouts are not comparable, and the difference
        is invisible in any summary statistic. This makes it checkable: the benchmark asserts
        every cell for a (target, seed) carries the same value.
        """
        joined = "|".join(sorted(self.held_out))
        return hashlib.sha256(joined.encode()).hexdigest()[:16]


def scaffold_split(
    smiles: list[str],
    *,
    target: str = "",
    fraction: float = DEFAULT_HOLDOUT,
    seed: int = 42,
) -> Split:
    """Withhold whole scaffold families.

    Candidates are restricted to the better-populated scaffolds and then shuffled on the
    seed. Both halves of that matter: a holdout of one-off molecules is uninformative, and a
    holdout that does not change with the seed means extra seeds add no independent evidence.
    Reproducible for a given seed, materially different between seeds.
    """
    by_scaffold: dict[str, list[str]] = {}
    unscaffoldable: list[str] = []

    for raw in smiles:
        smi = canonical(raw)
        if smi is None:
            continue
        sc = murcko_scaffold(smi)
        if sc is None or sc == "":
            unscaffoldable.append(smi)
            continue
        by_scaffold.setdefault(sc, []).append(smi)

    # Two competing requirements. The holdout should contain well-populated chemotypes --
    # a holdout made of one-off molecules is both easy to miss by luck and uninformative
    # when missed -- but it must also genuinely differ between seeds, or running "three
    # seeds" means three runs against one withheld set and the extra seeds add no
    # independent evidence at all.
    #
    # Taking strictly largest-first satisfies the first and destroys the second: scaffold
    # sizes decide the order and the seed only breaks ties, so JAK2 produced a byte-identical
    # holdout for every seed. So: restrict to the better-populated half, then shuffle *that*
    # on the seed.
    sized = sorted(by_scaffold.items(), key=lambda kv: -len(kv[1]))
    eligible = [kv for kv in sized if len(kv[1]) >= 2] or sized
    # Keep enough candidates to fill the quota several times over, so the shuffle has room
    # to produce materially different holdouts.
    keep = max(4, int(len(eligible) * 0.75))
    eligible = eligible[:keep]

    def shuffle_key(item: tuple[str, list[str]]) -> str:
        return hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest()

    ordered = sorted(eligible, key=shuffle_key)
    total = sum(len(m) for m in by_scaffold.values()) + len(unscaffoldable)
    quota = int(round(total * fraction))

    held, held_scaffolds = [], set()
    for sc, members in ordered:
        if len(held) >= quota:
            break
        # Never hold out everything: a run with no seeds cannot generate, and an empty
        # `seen` set would be measuring a different experiment entirely.
        if len(held) + len(members) > total - 2:
            continue
        held.extend(members)
        held_scaffolds.add(sc)

    held_set = set(held)
    seen = [s for s in (x for m in by_scaffold.values() for x in m) if s not in held_set]
    seen.extend(unscaffoldable)

    return Split(
        target=target,
        seen=tuple(sorted(set(seen))),
        held_out=tuple(sorted(held_set)),
        held_out_scaffolds=frozenset(held_scaffolds),
        seen_scaffolds=frozenset(by_scaffold) - held_scaffolds,
        fraction=fraction,
        seed=seed,
        unscaffoldable=len(unscaffoldable),
    )


def build_split(
    target: str,
    *,
    fraction: float = DEFAULT_HOLDOUT,
    seed: int = 42,
    n_actives: int = 300,
    min_pchembl: float = 6.0,
    settings=None,
) -> Split | None:
    """Fetch a target's actives and split them. ``None`` if the target has no ChEMBL data."""
    from ..clients import chembl
    from ..clients.http import BankClient
    from ..clients.targets import readiness
    from ..config import get_settings

    st = settings or get_settings()
    st.ensure_dirs()
    client = BankClient(st.http_cache_dir, offline=st.offline)

    rdy = readiness(client, target)
    if not rdy.chembl_id:
        return None

    acts = chembl.actives(client, rdy.chembl_id, limit=n_actives, min_pchembl=min_pchembl)
    if len(acts) < 20:
        return None

    split = scaffold_split(
        [a.smiles for a in acts], target=target, fraction=fraction, seed=seed
    )
    potency = {a.smiles: a.pchembl for a in acts}
    object.__setattr__(
        split,
        "meta",
        {
            "chembl_target": rdy.chembl_id,
            "uniprot": rdy.uniprot,
            "gene": rdy.gene,
            "min_pchembl": min_pchembl,
            "best_held_out_pchembl": round(
                max((potency.get(s, 0.0) for s in split.held_out), default=0.0), 2
            ),
        },
    )
    return split


__all__ = ["DEFAULT_HOLDOUT", "Split", "build_split", "scaffold_split"]
