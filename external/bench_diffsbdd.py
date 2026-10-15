"""How much of the card DiffSBDD needs, and how fast it samples, at several batch sizes.

The question is not whether a 1 M-parameter model fits on a 16 GB card -- it obviously does
-- but how much room it leaves for everything else. Boltz-2 affinity inference measured
~5.1 GB on this machine and Vina wants CPU rather than VRAM, so what matters is whether a
generation step and a scoring step can be held open at once.

The validity column is the one worth reading twice. It is the fraction of *requested*
samples that came back as molecules RDKit would accept, which is the number a budget has to
be planned against -- not the fraction of returned molecules that are valid, which is 100%
by construction and is how this statistic is usually quoted.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from diffsbdd_runner import REPO, generate, load_model  # noqa: E402

PDB = Path(__file__).resolve().parent / "pockets" / "7LL4.pdb"
REF_LIGAND = "A:1201"
BATCHES = (1, 8, 32, 64)


def main() -> int:
    model, dev = load_model(REPO / "checkpoints" / "crossdocked_fullatom_cond.ckpt")
    if dev != "cuda":
        print("no CUDA device; VRAM figures will be absent")

    print(f"{'batch':>6} {'valid':>6} {'validity':>9} {'wall_s':>8} {'s/mol':>7} {'peak_GB':>8}")
    for n in BATCHES:
        if dev == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        res = generate(model, dev, PDB, REF_LIGAND, n_samples=n)
        per = res.wall_s / res.n_returned if res.n_returned else float("nan")
        peak = f"{res.peak_vram_gb:>8.3f}" if res.peak_vram_gb is not None else f"{'-':>8}"
        print(
            f"{n:>6} {res.n_returned:>6} {res.validity:>8.0%} "
            f"{res.wall_s:>8.1f} {per:>7.2f} {peak}",
            flush=True,  # the run takes minutes; buffered output shows nothing until the end
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
