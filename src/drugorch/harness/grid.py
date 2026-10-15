"""The grid: every arm against every target at every seed, resumable.

Two properties this has to have, both learned the hard way by anyone who has run a benchmark
overnight:

**Resumable.** A cell whose run directory already carries a `summary.json` is rescored, not
re-run. Docking dominates the wall clock and an interrupted grid that restarts from zero is
a grid that never finishes.

**Failure-tolerant.** One dead cell -- a target whose receptor prep fails, an API outage --
is recorded and the grid continues. A benchmark that aborts on the first failure produces
nothing; one that drops the failure silently produces a lie. It does neither.

The split for a (target, seed) is built once and shared by every arm at that point, because
arms compared against different holdouts are not comparable.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings, get_settings
from .campaign import Cell, run_cell, score_run
from .metrics import RunMetrics
from .splits import Split, build_split


@dataclass(slots=True)
class GridResult:
    rows: list[RunMetrics] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    splits: dict[str, dict] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "failures": self.failures,
            "splits": self.splits,
            "started_at": self.started_at,
            "elapsed_s": round(time.time() - self.started_at, 1),
        }


def run_grid(
    targets: Iterable[str],
    policies: Iterable[str],
    seeds: Iterable[int],
    *,
    budget: float = 120.0,
    max_steps: int = 40,
    holdout: float = 0.3,
    settings: Settings | None = None,
    resume: bool = True,
    on_cell: Callable[[Cell, RunMetrics | None, str], None] | None = None,
) -> GridResult:
    """Run the whole grid and return every scored cell."""
    st = settings or get_settings()
    st.ensure_dirs()
    out = GridResult()

    targets, policies, seeds = list(targets), list(policies), list(seeds)
    split_cache: dict[tuple[str, int], Split | None] = {}

    for target in targets:
        for seed in seeds:
            key = (target, seed)
            if key not in split_cache:
                try:
                    split_cache[key] = build_split(
                        target, fraction=holdout, seed=seed, settings=st
                    )
                except Exception as exc:  # noqa: BLE001
                    out.failures.append(
                        {"target": target, "seed": seed, "stage": "split", "error": repr(exc)}
                    )
                    split_cache[key] = None
            split = split_cache[key]
            if split is not None:
                out.splits[f"{target}@{seed}"] = split.to_dict()

            for policy in policies:
                cell = Cell(
                    target=target,
                    policy=policy,
                    seed=seed,
                    budget=budget,
                    max_steps=max_steps,
                )
                run_dir = st.runs_dir / cell.run_id

                if resume and (run_dir / "summary.json").is_file():
                    try:
                        row = score_run(run_dir, cell, split)
                        row.notes.append("resumed: rescored, not re-run")
                        out.rows.append(row)
                        if on_cell:
                            on_cell(cell, row, "resumed")
                        continue
                    except Exception as exc:  # noqa: BLE001 - fall through and re-run
                        out.failures.append(
                            {**cell.to_dict(), "stage": "rescore", "error": repr(exc)}
                        )

                try:
                    row, _ = run_cell(cell, split, settings=st)
                    out.rows.append(row)
                    if on_cell:
                        on_cell(cell, row, "ran")
                except Exception as exc:  # noqa: BLE001 - one dead cell, not a dead grid
                    out.failures.append({**cell.to_dict(), "stage": "run", "error": repr(exc)})
                    if on_cell:
                        on_cell(cell, None, f"failed: {type(exc).__name__}")

    _check_split_consistency(out)
    return out


def _check_split_consistency(result: GridResult) -> None:
    """Every arm on one (target, seed) must have faced the same holdout.

    A mismatch is invisible in any summary number -- two arms would simply be answering
    different questions and the table would compare them anyway. Checked explicitly because
    nothing else would catch it.
    """
    seen: dict[str, str] = {}
    for key, s in result.splits.items():
        fp = s.get("fingerprint")
        if fp is None:
            continue
        prior = seen.get(key)
        if prior and prior != fp:
            result.failures.append(
                {"stage": "split_consistency", "key": key, "error": "holdout differs across arms"}
            )
        seen[key] = fp


def save(result: GridResult, path: Path) -> Path:
    """Write the grid as JSON plus a tidy CSV, one row per cell."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")

    if result.rows:
        csv_path = path.with_suffix(".csv")
        fields = [k for k in result.rows[0].to_dict() if k != "notes"]
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in result.rows:
                d = r.to_dict()
                d.pop("notes", None)
                w.writerow(d)
    return path


def load(path: Path) -> GridResult:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [RunMetrics(**{k: v for k, v in r.items()}) for r in raw.get("rows", [])]
    return GridResult(
        rows=rows,
        failures=raw.get("failures", []),
        splits=raw.get("splits", {}),
        started_at=raw.get("started_at", time.time()),
    )


__all__ = ["GridResult", "load", "run_grid", "save"]
