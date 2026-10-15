"""The ADMET worker: a long-lived subprocess that answers prediction requests.

Four of MolDesignBench's property constraints -- mutagenicity, blood-brain penetration, logD
and aqueous solubility -- are ML predictions rather than descriptors, and they appear in 1,427
of its 2,000 instances. ADMET-AI supplies all four, but it cannot live in this project's
environment: it pins RDKit back several releases, and RDKit's canonical SMILES are the keys
the budget ledger charges on and the value cache is keyed by. A silent change there would
orphan every cached result without bumping a version string to say so. So it gets its own
virtualenv, ``.venv-mdb``, holding the exact stack MolDesignBench vendors -- admet_ai 1.2.0,
chemprop 1.6.1, numpy 1.26.4, RDKit 2025.03.6, torch 2.5.1.

That the environment is *theirs* and not merely isolated is the point. ``CalcNumHBA`` returns
4 under RDKit 2025.03.6 and 3 under 2026.09.1 for the same molecule, so a property computed
here and a property in their answer key are not the same quantity unless they are computed
under the same RDKit. See :data:`BENCH_COLUMNS`.

The shape of the bridge was decided by measurement:

=====================  ========
import admet_ai         4.7 s
model load              0.5 s
predict 3 molecules     0.09 s
predict 400             ~8 s
=====================  ========

Spawning a process per call would spend five seconds to do a tenth of a second of work,
making ADMET more expensive per batch than docking. So the process is started once and kept,
and requests go over stdin as JSON lines. Predictions are then cached lazily in the ordinary
:class:`~drugorch.cache.ValueCache` exactly as docking's are -- a generated molecule does not
exist until the GA invents it, so there is nothing to precompute and no batch pass that would
help.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..versions import ADMET_V

# The four endpoints the benchmark's constraint names map onto. ADMET-AI returns 104;
# these are the ones anything here asks for, named explicitly so a model update that
# renames a column fails loudly rather than silently returning nothing.
ENDPOINTS = {
    "Mutag": "AMES",
    "BBBP": "BBB_Martins",
    "logS": "Solubility_AqSolDB",
    "logD": "Lipophilicity_AstraZeneca",
}

# ADMET-AI returns 104 columns: 52 endpoints, each with a `_drugbank_approved_percentile`
# twin saying where the molecule sits among approved drugs. Carrying all of them into
# `molecules.csv` would bury the handful anyone reads, so a panel is chosen here.
#
# The exclusions matter more than the inclusions. ADMET-AI also reports `logP`,
# `molecular_weight`, `tpsa`, `hydrogen_bond_donors`, `QED` and `stereo_centers` -- all of
# which `properties.py` already computes with RDKit. Writing those would put two recipes in
# one column, which is precisely what `versions.py` exists to prevent, so none of them are
# here. `BRENK_alert` is excluded for the same reason against the toxicity plugin.
PANEL: dict[str, tuple[str, str]] = {
    # prop name              ADMET-AI column               unit / reading
    "admet_ames": ("AMES", "P(mutagenic)"),
    "admet_bbb": ("BBB_Martins", "P(crosses blood-brain barrier)"),
    "admet_logs": ("Solubility_AqSolDB", "log mol/L"),
    "admet_logd": ("Lipophilicity_AstraZeneca", "logD at pH 7.4"),
    "admet_herg": ("hERG", "P(hERG blocker)"),
    "admet_dili": ("DILI", "P(drug-induced liver injury)"),
    "admet_clintox": ("ClinTox", "P(clinical toxicity)"),
    "admet_ld50": ("LD50_Zhu", "log(1/(mol/kg))"),
    "admet_caco2": ("Caco2_Wang", "log cm/s"),
    "admet_hia": ("HIA_Hou", "P(human intestinal absorption)"),
    "admet_bioavailability": ("Bioavailability_Ma", "P(oral bioavailability > 20%)"),
    "admet_pgp": ("Pgp_Broccatelli", "P(P-glycoprotein substrate)"),
    "admet_ppbr": ("PPBR_AZ", "% bound to plasma protein"),
    "admet_vdss": ("VDss_Lombardo", "L/kg"),
    "admet_half_life": ("Half_Life_Obach", "hours"),
    "admet_clearance": ("Clearance_Hepatocyte_AZ", "uL/min/1e6 cells"),
    "admet_cyp3a4": ("CYP3A4_Veith", "P(CYP3A4 inhibitor)"),
    "admet_cyp2d6": ("CYP2D6_Veith", "P(CYP2D6 inhibitor)"),
    "admet_cyp2c9": ("CYP2C9_Veith", "P(CYP2C9 inhibitor)"),
}

BENCH_COLUMNS: dict[str, str] = {
    "MW": "molecular_weight",
    "logP": "logP",
    "HBA": "hydrogen_bond_acceptors",
    "HBD": "hydrogen_bond_donors",
    "TPSA": "tpsa",
    "QED": "QED",
    "BBBP": "BBB_Martins",
    "Mutag": "AMES",
    "logS": "Solubility_AqSolDB",
    "logD": "Lipophilicity_AstraZeneca",
}
"""MolDesignBench property names to the ADMET-AI columns the benchmark reads them from.

Only four of these -- Mutag, BBBP, logS, logD -- are model predictions. The other six are
plain RDKit descriptors that this project can compute itself, and they are listed here
anyway, for a reason that took a while to find.

``CalcNumHBA`` does not mean the same thing in every RDKit. On
``O=C(O)c1ccc(-n2nc(C(=O)c3c(Cl)cccc3Cl)c3ccccc32)cc1`` it returns 4 under RDKit 2025.03.6
and 3 under 2026.09.1 -- same function, same molecule, different number, no warning. The
benchmark's windows were generated under the older one, and HBA constrains about 81% of its
instances, so computing it locally would have quietly disagreed with the answer key on a
sixth of every molecule's constraints while looking entirely healthy.

Taking these from ADMET-AI costs nothing -- the call is already being made -- and pins them
to the RDKit the answer key was built with instead of to whichever one this project happens
to have installed. Descriptors the benchmark computes outside ADMET-AI (rotB, rings_total,
heavy_atoms, formal_charge, MR) are left local, and the full-dataset gate is what says that
is still safe.
"""

DEFAULT_VENV = ".venv-mdb"
STARTUP_TIMEOUT_S = 180.0
"""Generous: a warm start is ~5s, and a cold filesystem is far slower than that."""

REQUEST_TIMEOUT_S = 120.0


class AdmetUnavailable(RuntimeError):
    """The worker could not be started or stopped responding.

    Raised rather than returning defaults: a wrong ADMET value scores exactly like a right
    one, so a constraint that cannot be evaluated must be reported as unevaluated.
    """


def worker_source() -> str:
    """The script the isolated interpreter runs.

    Kept as a string rather than a file so there is one place to change and no risk of the
    two drifting. It speaks JSON lines: one request object in, one response object out.
    """
    return r'''
import contextlib, io, json, os, sys

# chemprop and lightning print progress bars to STDOUT, which is the response channel.
# Rather than filter them out downstream -- where a stray line that happens to parse as
# JSON would be indistinguishable from a real reply -- the real stdout is captured once
# here and every library write goes to a bin.
_REAL = sys.stdout
sys.stdout = io.StringIO()

def _emit(obj):
    _REAL.write(json.dumps(obj) + chr(10))
    _REAL.flush()

def main():
    os.environ.setdefault("TQDM_DISABLE", "1")
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            from admet_ai import ADMETModel
            # num_workers=0 is not a tuning choice. This script runs as a bare module with
            # no __main__ guard, and on Windows a DataLoader worker is a *spawned* process
            # that re-imports its parent -- which would re-run this file, start another
            # model, and fork again. MolDesignBench passes the same argument for a related
            # reason (daemonic processes cannot have children).
            #
            # The fingerprint Pool is switched off for a platform reason, not a disagreement
            # with their default. It engages at 100 molecules, and on Windows a Pool is
            # *spawned*: measured here, 200 molecules took 8.85s with it and 4.08s without.
            # On Linux, where Pool forks, their default is very likely the faster one.
            # Fingerprints are identical either way -- this is scheduling, not arithmetic.
            try:
                model = ADMETModel(num_workers=0, fingerprint_multiprocessing_min=10**9)
            except TypeError:
                try:
                    model = ADMETModel(num_workers=0)
                except TypeError:
                    model = ADMETModel()
    except Exception as exc:
        _emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return
    _emit({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        if req.get("stop"):
            return
        smiles = req.get("smiles") or []
        # The caller names the columns it wants. The model returns 98 and the panel uses 19,
        # so sending everything spends most of the response on numbers nobody reads:
        # measured at 400 molecules, the full frame cost ~4s of an 11.8s round trip.
        want = req.get("columns")
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                frame = model.predict(smiles=smiles)
            if want:
                frame = frame[[c for c in want if c in frame.columns]]
            # Oriented by row so one unparseable molecule cannot shift every other
            # molecule's answers by a position.
            out = {str(idx): {k: _num(v) for k, v in row.items()}
                   for idx, row in frame.to_dict(orient="index").items()}
            _emit({"ok": True, "predictions": out})
        except Exception as exc:
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f   # NaN -> None, so a missing value is never a 0.0

# Not boilerplate. admet_ai opens a multiprocessing Pool once a batch reaches
# `fingerprint_multiprocessing_min` (100) molecules, and Windows starts those children by
# *spawn*: the child re-imports this file to rebuild its state. Without this guard it
# re-runs main(), prints a second startup banner into the response pipe, and the reader
# picks up that banner as the answer to its prediction request -- which fails only on
# batches of 100 or more, and fails by returning the wrong object rather than by crashing.
if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
'''


@dataclass
class AdmetWorker:
    """A handle on the isolated interpreter. Started on first use, reused thereafter."""

    venv: Path = field(default_factory=lambda: Path(DEFAULT_VENV))
    _proc: subprocess.Popen | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _script: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # `venv` arrives from plugin_config, which is JSON, so it is routinely a string.
        # Without this the path arithmetic below raises TypeError instead of the readable
        # "here is how to build the environment" message this class exists to give.
        if not isinstance(self.venv, Path):
            self.venv = Path(self.venv)

    def python(self) -> Path:
        """The isolated interpreter, on either platform layout."""
        for rel in ("Scripts/python.exe", "bin/python"):
            p = self.venv / rel
            if p.is_file():
                return p
        raise AdmetUnavailable(
            f"no interpreter in {self.venv}. Create it with:\n"
            f"  uv venv {self.venv} --python 3.12\n"
            f"  VIRTUAL_ENV={self.venv} uv pip install admet-ai"
        )

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        """Launch the worker and block until the model reports ready."""
        if self.running:
            return
        py = self.python()

        script = Path(self.venv) / "_drugorch_admet_worker.py"
        script.write_text(worker_source(), encoding="utf-8")
        self._script = script

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # Lightning and chemprop are chatty on stderr; it is kept separate so a progress
        # bar can never be mistaken for a response on stdout.
        self._proc = subprocess.Popen(
            [str(py), str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
            bufsize=1,
        )

        line = self._readline(STARTUP_TIMEOUT_S)
        if not line:
            self.stop()
            raise AdmetUnavailable("the ADMET worker produced no startup banner")
        try:
            banner = json.loads(line)
        except json.JSONDecodeError as exc:
            self.stop()
            raise AdmetUnavailable(f"unreadable startup banner: {line[:200]}") from exc
        if not banner.get("ready"):
            self.stop()
            raise AdmetUnavailable(f"the ADMET model failed to load: {banner.get('error')}")

    def _readline(self, timeout_s: float) -> str:
        """Read one line with a deadline, so a hung worker cannot hang the run.

        The read happens on a helper thread because a blocking `readline` on a pipe cannot
        be interrupted; if it never returns the thread is abandoned as a daemon and the
        worker is killed.
        """
        assert self._proc is not None and self._proc.stdout is not None
        box: list[str] = []

        def _read():
            try:
                box.append(self._proc.stdout.readline())  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                box.append("")

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout_s)
        return box[0] if box else ""

    def predict(
        self, smiles: list[str], columns: list[str] | None = None
    ) -> dict[str, dict]:
        """Predictions for a batch, keyed by the SMILES as given.

        ``columns`` restricts the response to the endpoints actually wanted, which is most
        of the round-trip cost on a large batch. ``None`` returns everything the model has.
        """
        if not smiles:
            return {}
        with self._lock:
            if not self.running:
                self.start()
            assert self._proc is not None and self._proc.stdin is not None
            try:
                req = {"smiles": list(smiles)}
                if columns:
                    req["columns"] = list(columns)
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self.stop()
                raise AdmetUnavailable(f"the ADMET worker went away: {exc}") from exc

            line = self._readline(REQUEST_TIMEOUT_S)
            if not line:
                self.stop()
                raise AdmetUnavailable("the ADMET worker timed out")
            try:
                resp = json.loads(line)
            except json.JSONDecodeError as exc:
                self.stop()
                raise AdmetUnavailable(f"unreadable response: {line[:200]}") from exc

        if "ready" in resp and "ok" not in resp:
            # A startup banner where a prediction belongs means a second interpreter wrote
            # to this pipe. Said plainly, because "prediction failed" sends you looking at
            # the model instead of at the process tree.
            self.stop()
            raise AdmetUnavailable(
                "the ADMET worker emitted a second startup banner, so another process is "
                "writing to the response pipe -- the worker script lost its __main__ guard"
            )
        if not resp.get("ok"):
            raise AdmetUnavailable(str(resp.get("error") or "prediction failed"))
        return resp.get("predictions") or {}

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"stop": True}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - shutdown is best effort
            pass
        finally:
            if proc.poll() is None:
                proc.kill()

    def describe(self) -> dict:
        return {
            "backend": "admet-ai",
            "version": ADMET_V,
            "venv": str(self.venv),
            "running": self.running,
            "endpoints": dict(ENDPOINTS),
        }


_WORKER: AdmetWorker | None = None


def get_worker(venv: Path | str | None = None) -> AdmetWorker:
    """The process-wide worker. One import cost per session, not per call."""
    global _WORKER  # noqa: PLW0603 - one subprocess per process, by design
    if _WORKER is None:
        _WORKER = AdmetWorker(venv=Path(venv) if venv else Path(DEFAULT_VENV))
    return _WORKER


__all__ = [
    "DEFAULT_VENV",
    "ENDPOINTS",
    "AdmetUnavailable",
    "AdmetWorker",
    "get_worker",
    "worker_source",
]
