"""Retrieval over the benchmark's own molecule pool.

MolDesignBench ships 1.9 million MOSES molecules pre-tagged with the same ADMET predictions
its answer key uses, and exposes them to agents as the tool
``query_moses_by_fg_counts_prop_ranges``. This is DrugOrch's route to the same pool: a
property-window query answered from the partitioned parquet directly, which takes about a
tenth of a second over the full set.

Retrieval is the first move for a reason that the benchmark's structure makes plain. Most
instances constrain six or seven properties at once, and a molecule drawn from a pool that
already satisfies five of them is a far better starting point for a genetic algorithm than
anything proposed from a blank page. The generator then has to fix two constraints rather
than seven.

A query that matches nothing is relaxed rather than returned empty, loosest constraint first,
and what was dropped is reported back to the caller. An agent told "no molecule matched" will
usually conclude the instance is infeasible, which is the single most expensive mistake
available to it: 1,800 of the 2,000 instances are solvable.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .spec import DesignSpec

DEFAULT_POOL = Path("third_party/MolDesignBench/data/moses_admet_tagged/parts")

# Spec vocabulary to pool column. The pool carries thirteen of the twenty measured
# properties; the rest (MR, heavy_atoms, Fsp3, formal_charge, rings_aromatic) are simply not
# filterable here and are left to the generator and to measurement.
COLUMNS: dict[str, str] = {
    "MW": "MW",
    "logP": "logP",
    "HBD": "HBD",
    "HBA": "HBA",
    "TPSA": "TPSA",
    "BBBP": "BBBP",
    "Mutag": "Mutag",
    "HIA": "HIA",
    "QED": "QED",
    "logS": "logS",
    "logD": "logD",
    "rotB": "rotB",
    "rings_total": "rings",
    "hetero": "hetero",
    "topo_diameter": "diameter",
}

# The pool stores fragment counts under RDKit's function names; the benchmark's constraint
# vocabulary uses English. The map is taken from the evaluator's own table rather than
# guessed, for the reason set out in `constraints.FRAGMENT_FNS`: a substring match sends
# `sulfonamide` to `fr_amide` and nothing raises.
FR_NAMES: dict[str, str] = {
    "benzene ring": "fr_benzene",
    "carbonyl o": "fr_C_O",
    "amide": "fr_amide",
    "pyridine ring": "fr_pyridine",
    "aliphatic hydroxyl": "fr_Al_OH",
    "methoxy": "fr_methoxy",
    "carboxylic acid": "fr_COO",
    "ketone": "fr_ketone",
    "phenol": "fr_phenol",
    "ester": "fr_ester",
    "furan ring": "fr_furan",
    "sulfonamide": "fr_sulfonamd",
    "thiocarbonyl": "fr_C_S",
    "thioether": "fr_sulfide",
    "oxime": "fr_oxime",
    "thiazole ring": "fr_thiazole",
    "sulfone": "fr_sulfone",
    "urea": "fr_urea",
    "hydrazone": "fr_hdrzone",
    "thiophene ring": "fr_thiophene",
    "nitrile": "fr_nitrile",
    "hydroxylamine": "fr_N_O",
    "aldehyde": "fr_aldehyde",
    "amidine": "fr_amidine",
    "guanidine": "fr_guanido",
    "hydrazine": "fr_hdrzine",
    "oxazole ring": "fr_oxazole",
    "alkyl carbamate": "fr_alkyl_carbamate",
    "thiol": "fr_SH",
    "nitroso": "fr_nitroso",
    "azo": "fr_azo",
    "beta lactam": "fr_lactam",
    "isothiocyanate": "fr_isothiocyan",
    "benzodiazepine": "fr_benzodiazepine",
    "thiocyanate": "fr_thiocyan",
    "barbiturate": "fr_barbitur",
    "diazo": "fr_diazo",
    "isocyanate": "fr_isocyan",
    "aromatic nitrogen": "fr_Ar_N",
    "aniline": "fr_aniline",
    "aryl methyl sites for hydroxylation": "fr_aryl_methyl",
    "ether oxygens (including phenoxy)": "fr_ether",
    "halogen": "fr_halogen",
    "nitro": "fr_nitro",
    "imidazole ring": "fr_imidazole",
    "piperidine ring": "fr_piperdine",
    "piperzine ring": "fr_piperzine",
    "morpholine ring": "fr_morpholine",
    "primary amine": "fr_NH2",
    "secondary amine": "fr_NH1",
    "tertiary amine": "fr_NH0",
}


class PoolUnavailable(RuntimeError):
    """The molecule pool is missing or unreadable."""


@dataclass
class PoolHit:
    smiles: str
    properties: dict[str, float] = field(default_factory=dict)


@dataclass
class PoolResult:
    """What a query found, and what it had to give up to find it."""

    hits: list[PoolHit] = field(default_factory=list)
    relaxed: list[str] = field(default_factory=list)
    matched: int = 0
    """Rows satisfying the property windows. The fragment filter runs after this, over a
    capped prefix, so `scanned` is what the returned sample was actually drawn from."""

    scanned: int = 0

    @property
    def smiles(self) -> list[str]:
        return [h.smiles for h in self.hits]

    def summary(self) -> str:
        if not self.hits:
            return "no molecule in the 1.9M pool matched, even after relaxing every window"
        line = f"{self.matched} molecules matched; {len(self.hits)} returned"
        if self.relaxed:
            line += f". Relaxed to get there: {', '.join(self.relaxed)}"
        return line


@dataclass
class MoleculePool:
    """A handle on the parquet pool. The dataset object is opened once and reused."""

    path: Path = field(default_factory=lambda: DEFAULT_POOL)
    n_queries: int = 0
    _dataset: object | None = field(default=None, repr=False)

    def dataset(self):
        if self._dataset is None:
            try:
                import pyarrow.dataset as ds
            except ImportError as exc:  # pragma: no cover
                raise PoolUnavailable("pyarrow is not installed") from exc
            p = Path(self.path)
            if not p.is_dir():
                raise PoolUnavailable(
                    f"{p} is missing. It ships with MolDesignBench under "
                    "data/moses_admet_tagged."
                )
            self._dataset = ds.dataset(str(p), format="parquet")
        return self._dataset

    def search(
        self, spec: DesignSpec, *, limit: int = 24, seed: int = 0
    ) -> PoolResult:
        """Molecules satisfying as much of ``spec`` as the pool can satisfy.

        Windows are applied together first. If nothing matches, the widest constraint by
        normalised window width is dropped and the query repeats, because a wide window is
        the one a generator can most easily repair afterwards.
        """
        import pyarrow.dataset as ds

        filterable = [p for p in spec.properties if p.name in COLUMNS]
        # Widest window first: the one a generator can repair most cheaply is the one worth
        # giving up when the pool cannot satisfy everything.
        order = sorted(
            filterable,
            key=lambda p: -(abs((p.hi if p.hi is not None else 1e6) - (p.lo if p.lo is not None else -1e6))),
        )
        # Exact counts where the task asks for them. A generation instance requires exactly
        # one benzene ring, not at least one, so returning molecules with three spends a
        # decision and a measurement batch on candidates that cannot satisfy the arm's own
        # specification.
        wanted_frags = {
            FR_NAMES[f.name]: (f.count, f.at_least)
            for f in spec.fragments
            if f.name in FR_NAMES
        }

        dropped: list[str] = []
        active = list(order)
        self.n_queries += 1

        for _ in range(len(order) + 1):
            expr = None
            for p in active:
                col = ds.field(COLUMNS[p.name])
                if p.lo is not None:
                    expr = (col >= p.lo) if expr is None else (expr & (col >= p.lo))
                if p.hi is not None:
                    expr = (col <= p.hi) if expr is None else (expr & (col <= p.hi))

            cols = ["smiles", "fragment"] + sorted({COLUMNS[p.name] for p in filterable})
            scanner = self.dataset().scanner(columns=cols, filter=expr, batch_size=65_536)

            # Counted without materialising: `expr` is None on the last relaxation pass and
            # whenever nothing in the spec is filterable, and the pool holds 1.9 million
            # rows. Building those as Python dicts, in each of several worker threads at
            # once, is a multi-gigabyte spike in place of the tenth of a second this query
            # is supposed to take.
            matched = scanner.count_rows()
            rows: list[dict] = []
            if matched:
                # Collected in batches with an early stop: a sample of a few dozen is
                # wanted, not the whole match set.
                cap = max(limit * 8, 200)
                for batch in scanner.to_batches():
                    for row in batch.to_pylist():
                        if wanted_frags:
                            try:
                                counts = json.loads(row.get("fragment") or "{}")
                            except json.JSONDecodeError:
                                continue
                            if not all(
                                (int(counts.get(k, 0)) >= n)
                                if at_least
                                else (int(counts.get(k, 0)) == n)
                                for k, (n, at_least) in wanted_frags.items()
                            ):
                                continue
                        rows.append(row)
                    if len(rows) >= cap:
                        break

            if rows:
                rng = random.Random(seed)
                picked = rows if len(rows) <= limit else rng.sample(rows, limit)
                return PoolResult(
                    hits=[
                        PoolHit(
                            smiles=str(r["smiles"]),
                            properties={
                                k: float(v)
                                for k, v in r.items()
                                if k not in ("smiles", "fragment") and v is not None
                            },
                        )
                        for r in picked
                    ],
                    relaxed=dropped,
                    matched=matched,
                    scanned=len(rows),
                )

            if not active:
                break
            giving_up = active.pop(0)
            dropped.append(giving_up.describe())

        return PoolResult(relaxed=dropped)

    def describe(self) -> dict:
        return {
            "pool": "MOSES, ADMET-tagged, as shipped with MolDesignBench",
            "path": str(self.path),
            "queries": self.n_queries,
        }


_POOL: MoleculePool | None = None


def get_pool(path: Path | str | None = None) -> MoleculePool:
    global _POOL  # noqa: PLW0603 - one open dataset per process, by design
    if _POOL is None:
        _POOL = MoleculePool(path=Path(path) if path else DEFAULT_POOL)
    return _POOL


__all__ = [
    "COLUMNS",
    "DEFAULT_POOL",
    "FR_NAMES",
    "MoleculePool",
    "PoolHit",
    "PoolResult",
    "PoolUnavailable",
    "get_pool",
]
