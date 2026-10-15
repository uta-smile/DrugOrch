"""ADMET prediction as a routable subsystem.

Where the properties panel computes what a molecule *is* -- weight, logP, ring count, all
exact functions of the graph -- this predicts how it would *behave*: whether it is mutagenic,
whether it crosses into the brain, how long it survives a pass through the liver. Those are
model predictions with real error bars rather than descriptors, and the distinction is kept
visible: every value lands under an ``admet_`` prefix and nothing here overwrites a column
``properties`` owns.

It is a plugin rather than a built-in for a reason that is not architectural tidiness.
ADMET-AI cannot be installed into this project's environment -- it pins RDKit back to
2026.03.6, and RDKit's canonical SMILES are the keys the budget ledger charges on. So it runs
in ``.venv-mdb`` behind a worker process (see :mod:`drugorch.admet.worker`), and a machine
without that environment should lose this one subsystem with a readable message rather than
fail to start. That is the failure mode ``Registry.load_failures`` was built for.

Caching is lazy, per molecule, in the ordinary :class:`~drugorch.cache.ValueCache`. There is
no precomputed table and there should not be one: the molecules that matter in a run are the
ones the generator invents during it, and they do not exist until it does.
"""

from __future__ import annotations

import threading
import time

from drugorch.admet import PANEL, AdmetUnavailable, get_worker
from drugorch.subsystems.base import (
    BaseSubsystem,
    Capability,
    Cost,
    ParamSpec,
    ProgressFn,
    SubsystemResult,
)
from drugorch.versions import ADMET_V

VERSION = "v1"

PANEL_V = "v1"
"""Which columns the panel carries. Separate from :data:`ADMET_V`, which is the model.

Both are in the cache key: adding a column must not serve rows that predate it, and a model
update must not blend two models into one column.
"""

PRIMARY = "admet_ames"
"""The prop consulted to decide whether a molecule still needs predicting."""

# Liabilities worth surfacing without being asked, and which side of the line is bad. The
# thresholds are the conventional reading of each endpoint, not tuned to anything here.
FLAGS = (
    ("admet_ames", 0.5, "mutagenic (Ames)"),
    ("admet_herg", 0.5, "hERG blocker"),
    ("admet_dili", 0.5, "liver injury risk"),
    ("admet_clintox", 0.5, "clinical toxicity"),
)

CAPABILITY = Capability(
    key="admet",
    title="ADMET prediction (ADMET-AI)",
    summary=(
        "Predict absorption, distribution, metabolism, excretion and toxicity for molecules "
        "-- mutagenicity, hERG, liver injury, brain penetration, solubility, logD, clearance "
        "and CYP inhibition. These are model predictions, not measurements: useful for "
        "ranking and for ruling molecules out, not evidence that a compound is safe."
    ),
    requires=("molecules",),
    produces=tuple(PANEL),
    params=(
        ParamSpec(
            name="smiles",
            type="smiles_list",
            default=[],
            doc="Molecules to predict. Leave empty to predict everything not yet predicted.",
        ),
        ParamSpec(
            name="max_molecules",
            type="int",
            default=200,
            lo=1,
            hi=2000,
            doc="Cap on how many molecules a single call predicts.",
        ),
    ),
    unit_cost=0.1,
    per="molecule",
    typical_wall_s=2.0,
    version=VERSION,
)


class AdmetSubsystem(BaseSubsystem):
    capability = CAPABILITY

    def __init__(self, config: dict | None = None, cache=None):
        self.config = config or {}
        self.cache = cache
        self._venv = self.config.get("venv")

    def cache_key(self) -> str:
        return f"admet:{ADMET_V}:panel:{PANEL_V}"

    def _targets(self, params: dict, state) -> list[str]:
        given = [s for s in (params.get("smiles") or []) if s]
        pool = given or [
            s for s, rec in state.molecules.items() if rec.props.get(PRIMARY) is None
        ]
        cap = int(params.get("max_molecules") or 200)
        return pool[:cap]

    def preflight(self, params: dict, state) -> tuple[bool, str]:
        if not state.molecules:
            return False, "there are no molecules to predict on yet"
        if not self._targets(params, state):
            return False, "every molecule already has ADMET predictions"
        return True, ""

    def urgency(self, state, facts: dict) -> tuple[float, str]:
        """Informative and cheap, but it never moves the objective by itself.

        Scored below docking deliberately: a run that spends its budget characterising
        molecules it never measured against the target has learned a great deal about
        nothing.
        """
        n = len([s for s, rec in state.molecules.items() if rec.props.get(PRIMARY) is None])
        if n == 0:
            return 0.0, "every molecule has ADMET predictions"
        if n >= 20:
            return 0.5, f"{n} molecules unpredicted; one batch covers them all"
        return 0.25, f"only {n} molecules unpredicted; the batch barely pays for its startup"

    def run(
        self, params: dict, state, *, cancel: threading.Event, progress: ProgressFn
    ) -> SubsystemResult:
        return self._safe(lambda: self._run(params, state, cancel, progress))

    def _run(self, params: dict, state, cancel, progress) -> SubsystemResult:
        t0 = time.perf_counter()
        targets = self._targets(params, state)

        key = self.cache_key()
        cached = self.cache.get_many(key, targets) if self.cache else {}
        pending = [s for s in targets if s not in cached]

        worker = get_worker(self._venv)
        predicted: dict[str, dict] = {}
        unparseable: list[str] = []
        started_cold = not worker.running

        if pending and not cancel.is_set():
            progress(
                f"{len(cached)} from cache, predicting {len(pending)}"
                + (" (starting the ADMET environment)" if started_cold else ""),
                0.1,
            )
            try:
                raw = worker.predict(pending, columns=[c for c, _u in PANEL.values()])
            except AdmetUnavailable as exc:
                # Deliberately not degraded into defaults. A wrong ADMET value ranks exactly
                # like a right one, so a model that cannot answer has to be a call that
                # cannot answer.
                return SubsystemResult(
                    subsystem=self.capability.key,
                    ok=False,
                    explain={
                        "backend": "admet-ai",
                        "venv": str(worker.venv),
                        "rationale": (
                            "The isolated ADMET environment could not answer, so no "
                            "predictions were made. No defaults were substituted: a guessed "
                            "value would rank identically to a predicted one."
                        ),
                    },
                    cost=Cost(budget_units=0.0, wall_s=round(time.perf_counter() - t0, 3)),
                    note="ADMET unavailable",
                    error=str(exc),
                )
            for smi in pending:
                row = raw.get(smi)
                if not row:
                    unparseable.append(smi)
                    continue
                predicted[smi] = {prop: row.get(col) for prop, (col, _u) in PANEL.items()}

        if self.cache and predicted:
            wall = (time.perf_counter() - t0) / max(len(predicted), 1)
            self.cache.put_many(
                key, {s: (v, True, round(wall, 4)) for s, v in predicted.items()}
            )

        records: list[dict] = []
        for smi in targets:
            if smi in predicted:
                props = predicted[smi]
            elif smi in cached:
                value, ok = cached[smi]
                if not ok or not value:
                    continue
                props = value
            else:
                continue
            records.append({"smiles": smi, "props": dict(props)})

        flagged = self._flag_counts(records)
        return SubsystemResult(
            subsystem=self.capability.key,
            ok=bool(records),
            records=records,
            explain={
                "backend": "admet-ai",
                "model_version": ADMET_V,
                "panel_version": PANEL_V,
                "endpoints": len(PANEL),
                "predicted": len(predicted),
                "from_cache": len(cached),
                "unparseable": len(unparseable),
                "worker_started": started_cold,
                "liabilities": flagged,
                "units": {p: u for p, (_c, u) in PANEL.items()},
                "rationale": (
                    f"Predicted {len(PANEL)} ADMET endpoints for {len(records)} molecules "
                    f"({len(predicted)} computed, {len(cached)} already cached). "
                    + (
                        "Flagged: " + ", ".join(f"{n} {label}" for label, n in flagged.items())
                        if flagged
                        else "No molecule crossed a liability threshold."
                    )
                ),
                "caveat": (
                    "Predictions from models trained on public TDC datasets. Treat them as a "
                    "prioritisation signal, not as evidence of safety."
                ),
            },
            cost=Cost(
                # Charged on the molecules the run did not already know, as everywhere else:
                # a cache hit saves wall time, never budget.
                budget_units=self.capability.unit_cost * len(pending),
                wall_s=round(time.perf_counter() - t0, 3),
                n_unique_smiles=len(records),
                cache_hits=len(cached),
            ),
            note=(
                f"{len(records)} molecules, {len(PANEL)} endpoints"
                + (f", {sum(flagged.values())} liability flags" if flagged else "")
            ),
            error=None if records else "no predictions were produced",
        )

    @staticmethod
    def _flag_counts(records: list[dict]) -> dict[str, int]:
        out: dict[str, int] = {}
        for rec in records:
            for prop, threshold, label in FLAGS:
                v = rec["props"].get(prop)
                if isinstance(v, (int, float)) and v >= threshold:
                    out[label] = out.get(label, 0) + 1
        return out

    def describe(self) -> dict:
        return {
            "key": self.capability.key,
            "version": VERSION,
            "backend": "admet-ai",
            "model_version": ADMET_V,
            "panel_version": PANEL_V,
            "endpoints": {p: c for p, (c, _u) in PANEL.items()},
        }


def build(ctx):
    """The factory every subsystem is constructed through, plugin or built-in."""
    return AdmetSubsystem(config=ctx.config, cache=ctx.cache)
