"""The benchmark harness: holding actives back, running the grid, ranking the arms.

A single run tells you what happened once. This turns that into evidence: the same arms, on
the same problems, at the same seeds and budgets, against a set of molecules none of them was
shown -- and a leaderboard that reports the comparison paired rather than averaged, because
targets differ from each other far more than policies differ on any one of them.
"""

from .campaign import Cell, run_cell, score_run
from .grid import GridResult, run_grid, save
from .metrics import RunMetrics, evaluate
from .report import ArmSummary, console_table, leaderboard, per_target
from .splits import Split, build_split, scaffold_split

__all__ = [
    "ArmSummary",
    "Cell",
    "GridResult",
    "RunMetrics",
    "Split",
    "build_split",
    "console_table",
    "evaluate",
    "leaderboard",
    "per_target",
    "run_cell",
    "run_grid",
    "save",
    "scaffold_split",
    "score_run",
]
