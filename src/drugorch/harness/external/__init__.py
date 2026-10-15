"""Adapters for benchmarks DrugOrch did not define.

The routing benchmark next door asks whether DrugOrch's orchestrator beats its own baselines.
This package asks the question a reader outside the project cares about: how the system does
on a benchmark somebody else designed, scored by that benchmark's own evaluator, against the
numbers its authors published.

The pieces, in the order a run uses them:

* :mod:`instances` loads MolDesignBench and holds the line between a question and its key.
* :mod:`spec` is the constraint set an arm believes it was asked for.
* :mod:`pool` retrieves from the 1.9M MOSES molecules the benchmark ships.
* :mod:`ga` evolves candidates toward a spec with DrugOrch's own GraphGA operators.
* :mod:`panel` measures molecules with the benchmark's instruments, in its environment.
* :mod:`agent` is the two arms: the orchestrated product, and a no-tool control.
* :mod:`mdb_eval` scores answers by running the benchmark's evaluator unmodified.
* :mod:`baselines` holds the published numbers, labelled as read rather than measured.
* :mod:`run` ties it together and writes the board.
"""

from .constraints import Constraints, Evaluation, evaluate, evaluate_many
from .instances import Instance, load_instances, prompt_view, stratified_sample
from .mdb_eval import ArmScore
from .run import ARMS, ExternalResult, run_external, save
from .spec import DesignSpec, Score

__all__ = [
    "ARMS",
    "ArmScore",
    "Constraints",
    "DesignSpec",
    "Evaluation",
    "ExternalResult",
    "Instance",
    "Score",
    "evaluate",
    "evaluate_many",
    "load_instances",
    "prompt_view",
    "run_external",
    "save",
    "stratified_sample",
]
