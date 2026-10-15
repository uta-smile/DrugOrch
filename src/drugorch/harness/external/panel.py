"""Measuring a molecule with the benchmark's own instruments.

Every property on the public board is measured by MolDesignBench's
``analyze_overall_molecule_properties`` and ``get_functional_groups``, running in
``.venv-mdb`` against the stack the benchmark vendors. Not a reimplementation of them, and
not DrugOrch's own RDKit panel.

The reason is that the two disagree, quietly. ``CalcNumHBA`` returns 4 under RDKit 2025.03.6
and 3 under 2026.09.1 for the same molecule, and a constraint whose window opens at 4 then
turns on which interpreter measured it rather than on the chemistry. ADMET-AI 1.2.0 and 2.0.1
differ by about the width of the benchmark's tolerance windows. A panel computed here and an
answer key computed there are not the same quantity, so the measurement is delegated to the
environment the key was built in.

This is also what the paper hands its own agents: ``analyze_overall_molecule_properties`` is
one of the seventeen tools in the w/ tool setting. Using it as the arm's measuring tool is
their protocol, not an advantage taken.

The process is started once and kept. ADMET-AI costs about five seconds to import and load
and a tenth of a second to answer, so a process per call would spend fifty times longer
starting than working.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_VENV = Path(".venv-mdb")
BENCH_REPO = Path("third_party/MolDesignBench")

STARTUP_TIMEOUT_S = 300.0
REQUEST_TIMEOUT_S = 300.0

# The denominators the benchmark normalises constraint distance by: the p5-p95 range of each
# property over 10,838,671 molecules, copied from `evaluate/evaluate_benchmark.py`. They are
# here because the fitness an arm optimises against should be the same shape as the distance
# it will be scored on; a fitness in raw units would spend its effort on whichever property
# happens to have the largest numbers (MW, always) instead of on the one furthest outside
# its window.
PERCENTILE_RANGES: dict[str, float] = {
    "MW": 303.32,
    "logP": 5.05,
    "logD": 4.43,
    "logS": 5.03,
    "HBD": 3.0,
    "HBA": 6.0,
    "TPSA": 93.66,
    "rotB": 7.0,
    "rings_total": 4.0,
    "QED": 0.65,
    "BBBP": 0.65,
    "Mutag": 0.70,
    "MR": 82.85,
    "heavy_atoms": 22.0,
    "formal_charge": 1.0,
}

FRAGMENT_RANGES: dict[str, float] = {
    "aromatic nitrogen": 4.0,
    "carbonyl o": 3.0,
    "amide": 2.0,
    "aniline": 2.0,
    "aryl methyl sites for hydroxylation": 2.0,
    "benzene ring": 3.0,
    "ether oxygens (including phenoxy)": 3.0,
    "halogen": 3.0,
    "methoxy": 2.0,
}


class PanelUnavailable(RuntimeError):
    """The measurement environment could not be started or stopped answering.

    Raised rather than falling back to a local RDKit, because a locally measured property
    and a benchmark-measured one are different quantities and mixing them would produce a
    score no one could interpret.
    """


def worker_source(repo: Path) -> str:
    """The script the isolated interpreter runs.

    Both tool classes write banners to stdout as they import (``Failed to patch pandas``,
    ``Could not load retrieval tools``), and stdout is the response channel. The real handle
    is captured before any import, exactly as the ADMET worker does it, so a library banner
    can never be read as a reply.
    """
    return (
        "import contextlib, io, json, os, sys\n"
        "_REAL = sys.stdout\n"
        "sys.stdout = io.StringIO()\n"
        f"sys.path.insert(0, {str(repo.resolve())!r})\n"
        """
def _emit(obj):
    _REAL.write(json.dumps(obj) + chr(10))
    _REAL.flush()

def main():
    os.environ.setdefault("TQDM_DISABLE", "1")
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            from moldesign.tools.analysis import MolPropAnalyzer, FunctionalGroups
            analyzer = MolPropAnalyzer()
            groups = FunctionalGroups()
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
        smiles = [s for s in (req.get("smiles") or []) if isinstance(s, str)]
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                props = {}
                if smiles:
                    raw = analyzer({"mol_smiles_list": smiles}, return_text=False)
                    # The tool returns a dict for one molecule and a list of dicts for
                    # several; both shapes are keyed by SMILES.
                    if isinstance(raw, dict):
                        props = raw
                    elif isinstance(raw, list):
                        for entry in raw:
                            if isinstance(entry, dict):
                                props.update(entry)
                    elif isinstance(raw, str):
                        _emit({"ok": False, "error": raw.splitlines()[0]})
                        continue
                frags = {}
                for smi in smiles:
                    fg = groups({"mol_smiles": smi}, return_text=False)
                    frags[smi] = fg if isinstance(fg, dict) else {}
            _emit({"ok": True, "properties": props, "fragments": frags})
        except Exception as exc:
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
"""
    )


@dataclass
class Measurement:
    """One molecule as the benchmark measures it."""

    smiles: str
    properties: dict[str, float] = field(default_factory=dict)
    fragments: dict[str, int] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return bool(self.properties)

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "properties": self.properties,
            "fragments": self.fragments,
        }


@dataclass
class PanelWorker:
    """A handle on the measurement interpreter. Started on first use, reused thereafter."""

    venv: Path = field(default_factory=lambda: DEFAULT_VENV)
    repo: Path = field(default_factory=lambda: BENCH_REPO)
    _proc: subprocess.Popen | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    n_calls: int = 0
    n_molecules: int = 0

    def __post_init__(self) -> None:
        self.venv = Path(self.venv)
        self.repo = Path(self.repo)

    def python(self) -> Path:
        for rel in ("Scripts/python.exe", "bin/python"):
            p = self.venv / rel
            if p.is_file():
                return p
        raise PanelUnavailable(
            f"no interpreter in {self.venv}. The public benchmark is measured in the "
            "environment its answer key was built in; create it with:\n"
            f"  uv venv {self.venv} --python 3.12\n"
            f"  VIRTUAL_ENV={self.venv} uv pip install -r {self.repo}/requirements.txt"
        )

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        if self.running:
            return
        if not (self.repo / "moldesign").is_dir():
            raise PanelUnavailable(f"{self.repo} does not hold the MolDesignBench sources")

        script = self.venv / "_drugorch_panel_worker.py"
        script.write_text(worker_source(self.repo), encoding="utf-8")

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        self._proc = subprocess.Popen(
            [str(self.python()), str(script)],
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
            raise PanelUnavailable("the measurement worker produced no startup banner")
        try:
            banner = json.loads(line)
        except json.JSONDecodeError as exc:
            self.stop()
            raise PanelUnavailable(f"unreadable startup banner: {line[:200]}") from exc
        if not banner.get("ready"):
            self.stop()
            raise PanelUnavailable(f"the measurement tools failed to load: {banner.get('error')}")

    def _readline(self, timeout_s: float) -> str:
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

    def measure(self, smiles: list[str]) -> dict[str, Measurement]:
        """Measure a batch, keyed by the SMILES as given.

        Invalid molecules come back with empty properties rather than raising: a generation
        round routinely proposes a few unparseable children, and one of them must not end
        the instance.
        """
        wanted = [s for s in dict.fromkeys(smiles) if isinstance(s, str) and s.strip()]
        if not wanted:
            return {}

        with self._lock:
            if not self.running:
                self.start()
            assert self._proc is not None and self._proc.stdin is not None
            try:
                self._proc.stdin.write(json.dumps({"smiles": wanted}) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self.stop()
                raise PanelUnavailable(f"the measurement worker went away: {exc}") from exc

            line = self._readline(REQUEST_TIMEOUT_S)
            if not line:
                self.stop()
                raise PanelUnavailable("the measurement worker timed out")
            try:
                resp = json.loads(line)
            except json.JSONDecodeError as exc:
                self.stop()
                raise PanelUnavailable(f"unreadable response: {line[:200]}") from exc

            self.n_calls += 1
            self.n_molecules += len(wanted)

        if "ready" in resp and "ok" not in resp:
            self.stop()
            raise PanelUnavailable(
                "the measurement worker emitted a second startup banner, so another "
                "process is writing to the response pipe"
            )
        if not resp.get("ok"):
            raise PanelUnavailable(str(resp.get("error") or "measurement failed"))

        props = resp.get("properties") or {}
        frags = resp.get("fragments") or {}
        return {
            s: Measurement(
                smiles=s,
                properties={k: v for k, v in (props.get(s) or {}).items() if v is not None},
                fragments=frags.get(s) or {},
            )
            for s in wanted
        }

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
            "measured_by": "MolDesignBench analyze_overall_molecule_properties + "
            "get_functional_groups",
            "venv": str(self.venv),
            "repo": str(self.repo),
            "calls": self.n_calls,
            "molecules": self.n_molecules,
        }


_WORKER: PanelWorker | None = None


def get_panel(venv: Path | str | None = None) -> PanelWorker:
    """The process-wide measurement worker. One import cost per session, not per call."""
    global _WORKER  # noqa: PLW0603 - one subprocess per process, by design
    if _WORKER is None:
        _WORKER = PanelWorker(venv=Path(venv) if venv else DEFAULT_VENV)
    return _WORKER


__all__ = [
    "BENCH_REPO",
    "DEFAULT_VENV",
    "FRAGMENT_RANGES",
    "PERCENTILE_RANGES",
    "Measurement",
    "PanelUnavailable",
    "PanelWorker",
    "get_panel",
    "worker_source",
]
