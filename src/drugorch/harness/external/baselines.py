"""The published state of the art on MolDesignBench.

These rows are copied from Table 3 of the paper, as reproduced in
``third_party/MolDesignBench/docs/leaderboard.md``. They were not re-run here, and they are
labelled as reported on every surface that shows them. The distinction matters: a number we
measured and a number we read out of a paper carry different warranties, and putting them in
one table without saying which is which is how a leaderboard stops being evidence.

Each row covers all 1,000 instances of its task. A DrugOrch row measured over a subset is
therefore not a like-for-like comparison on sample size, which is why every row on the board
carries its ``n`` and why the sampling is stratified: an even task split and the full set's
10% infeasible share, so the columns are at least measuring the same mixture.

Citation
--------
Jeong, Ko, Kim, Lee, Hormazabal, Lee, Han, Lim and Kim. *MolDesignBench: Evaluating
LLM-based Agent for Scenario-grounded Molecular Design.* COLM 2026.
"""

from __future__ import annotations

AUTHORS = (
    "Jeong, Yongjun",
    "Ko, Hanbum",
    "Kim, Ye Rin",
    "Lee, Chanhui",
    "Hormazabal, Rodrigo",
    "Lee, Jaewan",
    "Han, Sehui",
    "Lim, Sungbin",
    "Kim, Sungwoong",
)

BIBTEX = """\
@inproceedings{jeong2026moldesignbench,
  title     = {MolDesignBench: Evaluating LLM-based Agent for Scenario-grounded
               Molecular Design},
  author    = {Jeong, Yongjun and Ko, Hanbum and Kim, Ye Rin and Lee, Chanhui and
               Hormazabal, Rodrigo and Lee, Jaewan and Han, Sehui and Lim, Sungbin and
               Kim, Sungwoong},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}"""

SOURCE = {
    "paper": "MolDesignBench: Evaluating LLM-based Agent for Scenario-grounded "
    "Molecular Design",
    "authors": list(AUTHORS),
    "venue": "Conference on Language Modeling (COLM)",
    "venue_short": "COLM 2026",
    "year": 2026,
    "table": "Table 3",
    # Where this project read the numbers. The chain stops here: the figures are checked
    # against this file by `tests/test_external.py`, and that file is taken as the paper's
    # own reproduction of its table.
    "file": "third_party/MolDesignBench/docs/leaderboard.md",
    "code_url": "https://github.com/KU-AGI/MolDesignBench",
    "data_url": "https://huggingface.co/datasets/LG-AI-Research/MolDesignBench",
    "licence": "Code MIT, (c) 2026 Korea University and LG AI Research. "
    "Benchmark data CC BY-NC 4.0.",
    "bibtex": BIBTEX,
    "n_per_task": 1000,
    "measured_here": False,
}

# model, tools, success, csr, mcd, inf_acc, mcc, validity, tool_calls
_GENERATION = [
    ("GPT-5.4", False, 0.11, 0.75, 6.38, 0.27, 0.40, 0.96, None),
    ("GPT-5.4", True, 0.38, 0.82, 3.49, 0.48, 0.44, 0.92, 16.56),
    ("Claude-Opus-4.6", False, 0.04, 0.57, 7.31, 0.01, 0.09, 0.75, None),
    ("Claude-Opus-4.6", True, 0.20, 0.61, 4.87, 0.02, 0.13, 0.71, 12.69),
    ("GPT-OSS-120B", False, 0.06, 0.60, 8.75, 0.26, 0.41, 0.82, None),
    ("GPT-OSS-120B", True, 0.13, 0.32, 5.87, 0.27, 0.40, 0.38, 6.59),
    ("Qwen3-235B-A22B", False, 0.08, 0.45, 8.94, 0.51, 0.19, 0.62, None),
    ("Qwen3-235B-A22B", True, 0.19, 0.64, 7.35, 0.41, 0.32, 0.79, 8.29),
    ("Ether0-24B", False, 0.00, 0.15, 18.01, 0.00, 0.00, 0.25, None),
    ("Ether0-24B", True, 0.00, 0.12, 23.52, 0.00, 0.00, 0.25, 0.00),
    ("InternLM-S1-241B", False, 0.08, 0.66, 9.50, 0.46, 0.39, 0.93, None),
    ("InternLM-S1-241B", True, 0.09, 0.58, 9.73, 0.32, 0.28, 0.80, 8.80),
]

_OPTIMIZATION = [
    ("GPT-5.4", False, 0.17, 0.81, 2.99, 0.12, 0.33, 0.97, None),
    ("GPT-5.4", True, 0.43, 0.84, 1.62, 0.31, 0.48, 0.91, 16.20),
    ("Claude-Opus-4.6", False, 0.17, 0.73, 2.87, 0.00, 0.00, 0.87, None),
    ("Claude-Opus-4.6", True, 0.41, 0.84, 1.58, 0.00, 0.00, 0.91, 12.36),
    ("GPT-OSS-120B", False, 0.11, 0.67, 4.07, 0.11, 0.22, 0.84, None),
    ("GPT-OSS-120B", True, 0.23, 0.47, 2.64, 0.31, 0.43, 0.53, 4.97),
    ("Qwen3-235B-A22B", False, 0.14, 0.63, 3.96, 0.32, 0.30, 0.78, None),
    ("Qwen3-235B-A22B", True, 0.25, 0.76, 3.17, 0.10, 0.13, 0.90, 5.78),
    ("Ether0-24B", False, 0.02, 0.40, 8.90, 0.00, 0.00, 0.60, None),
    ("Ether0-24B", True, 0.00, 0.16, 8.53, 0.00, 0.00, 0.22, 0.00),
    ("InternLM-S1-241B", False, 0.13, 0.75, 4.23, 0.14, 0.21, 0.95, None),
    ("InternLM-S1-241B", True, 0.10, 0.64, 5.43, 0.11, 0.07, 0.84, 5.91),
]


def _rows(raw: list[tuple], task: str) -> list[dict]:
    out = []
    for model, tools, success, csr, mcd, inf_acc, mcc, validity, calls in raw:
        out.append(
            {
                "arm": model,
                "tools": bool(tools),
                "scope": task,
                "n": SOURCE["n_per_task"],
                "success": success,
                "csr": csr,
                "mcd": mcd,
                "inf_acc": inf_acc,
                "mcc": mcc,
                "validity": validity,
                "tool_calls": calls,
                "measured_here": False,
                "source": f"{SOURCE['venue_short']}, {SOURCE['table']}",
            }
        )
    return out


def published(task: str | None = None) -> list[dict]:
    """The published rows, for one task or both."""
    rows = _rows(_GENERATION, "generation") + _rows(_OPTIMIZATION, "optimization")
    return [r for r in rows if task is None or r["scope"] == task]


def best_published(task: str, metric: str = "success") -> dict | None:
    """The leading published row on one task, which is what a new row is read against."""
    rows = [r for r in published(task) if r.get(metric) is not None]
    if not rows:
        return None
    reverse = metric != "mcd"  # MCD is a distance: lower is better
    return sorted(rows, key=lambda r: r[metric], reverse=reverse)[0]


__all__ = ["SOURCE", "best_published", "published"]
