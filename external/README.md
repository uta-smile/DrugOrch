# Pocket-conditioned generation: DiffSBDD on a Blackwell card

This directory holds the one thing DrugOrch could not do: propose molecules *from* a binding
site rather than judging molecules against one. The workbench's "Fit a pocket" panel is
structure-conditioned **selection** — chemistry proposes, the pocket disposes. DiffSBDD is a
structure-conditioned **prior**: it diffuses atoms directly into the site.

It runs. The numbers below were measured on this machine, not quoted.

## What is here

| path | what it is |
|---|---|
| `DiffSBDD/` | upstream checkout, MIT, unmodified — do not edit it |
| `diffsbdd_runner.py` | a callable interface, plus the three fixes needed to run it here |
| `torch_scatter_shim/` | pure-PyTorch `scatter_add`/`scatter_mean`, replacing a package that cannot be built here |
| `bench_diffsbdd.py` | VRAM and throughput at several batch sizes |
| `pockets/` | PDB files used as generation targets |

Everything under `external/DiffSBDD` is excluded from ruff (see the root `pyproject.toml`);
the runner and the shim are not, and are linted like the rest of the project.

## Reproducing the environment

```powershell
uv venv --python 3.12 .venv-sbdd
$env:VIRTUAL_ENV = ".venv-sbdd"
uv pip install torch --torch-backend=cu128
uv pip install -e external/torch_scatter_shim --no-deps
uv pip install "pytorch-lightning>=2.0" rdkit "biopython==1.81" networkx `
               openbabel-wheel wandb seaborn imageio scipy numpy
git clone --depth 1 https://github.com/arneschneuing/DiffSBDD.git external/DiffSBDD
curl -L -o external/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt `
     "https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt?download=1"
```

Then:

```powershell
.venv-sbdd\Scripts\python.exe external\diffsbdd_runner.py `
    --pdbfile external\pockets\7LL4.pdb --ref-ligand A:1201 --n-samples 20
```

`--ref-ligand` is `CHAIN:RESI` of something already bound in the pocket, or a path to an SDF.
It says *where* the site is; it does not condition the chemistry.

## Four things had to be fixed, none of them scientific

**`torch-scatter` cannot be installed here.** It ships CUDA kernels that must be compiled
against the exact torch build, there is no wheel for torch 2.11 + cu128 on Windows/CPython
3.12, and building needs MSVC, which is not present. DiffSBDD imports exactly two of its
functions and calls both with `dim=0` at all ~35 call sites, which torch does natively. The
shim provides those two and refuses anything else rather than silently differing.
`tests/test_scatter_shim.py` checks it against hand-computed values and against the authors'
own native implementation in `equivariant_diffusion/egnn_new.py` — including the case where
an empty segment must average to **zero and not NaN**, which would otherwise put NaNs into a
diffusion trajectory and yield molecules that look fine and are not.

**`biopython` 1.82 removed `three_to_one`.** Pinned to 1.81, the newest release that still
has it.

**The checkpoint is a Lightning 1.8 pickle and torch is 2.11.** Since torch 2.6 `torch.load`
defaults to `weights_only=True` and the file carries an `argparse.Namespace`. The runner
allowlists that one class. `weights_only=False` would also work and would permit arbitrary
code execution while unpickling; the narrow option is enough. Lightning 2.6 upgrades the 1.8
checkpoint on load without complaint.

**The molecule builder cannot work on Windows.** `make_mol_openbabel` opens a
`NamedTemporaryFile` and hands its *name* to three functions that open it again — fine on
POSIX, `PermissionError` on Windows, and it fails *after* the diffusion has run. Worse,
OpenBabel's `ReadFile`/`WriteFile` keep their streams open until the converter is collected,
so even a temporary directory cannot be torn down. The runner replaces the function with one
that converts in memory via `ReadString`/`WriteString`. The clone stays pristine, so `git
pull` does not conflict and nobody has to wonder whether the model was modified.

## Measured on an RTX 5070 Ti (16 GB, sm_120, driver 610.88)

Model: `crossdocked_fullatom_cond`, **1.01 M parameters**, 17.8 MB checkpoint — three orders
of magnitude smaller than the docking and affinity work it sits beside.

Sampling into the JAK2 pocket of 7LL4 (`bench_diffsbdd.py`):

| batch | valid | validity | wall | s / valid molecule | peak VRAM |
|---|---|---|---|---|---|
| 1 | 1 | 100% | 5.6 s | 5.56 s | 0.05 GB |
| 8 | 2 | 25% | 13.8 s | 6.91 s | 0.34 GB |
| 32 | 14 | 44% | 49.7 s | 3.55 s | 1.58 GB |
| 64 | 23 | 36% | 116.2 s | 5.05 s | 6.27 GB |

Two things to take from this.

**VRAM is driven by the batch, not by the model.** A 1 M-parameter network reaching 6.3 GB at
64 samples is not the weights — it is the pocket graph and every sample's trajectory held at
once. 64 is comfortable on this card; it would not leave room for Boltz-2 (~5.1 GB measured)
at the same time, so a cascade has to sequence them rather than co-resident them. A batch of
32 at 1.6 GB leaves plenty of room and costs the least per usable molecule.

**Validity is 25–44%, so the honest throughput is per *valid* molecule.** Asking for 64 and
getting 23 is the normal case, not a bad run. Plan budgets on the right-hand columns.

## What it actually produces

Twenty samples into the JAK2 pocket of 7LL4 (chain A, ligand Y5D), then docked with Vina
through DrugOrch's own workbench against GraphGA analogues of ruxolitinib and against
ruxolitinib itself — same pocket, same box, same exhaustiveness:

| source | n | best (kcal/mol) | median | mean QED | mean SA | PAINS |
|---|---|---|---|---|---|---|
| DiffSBDD | 9 | **−10.37** | −7.00 | 0.405 | 4.02 | 2 |
| GraphGA (from ruxolitinib) | 9 | −9.20 | **−8.69** | **0.741** | 3.80 | 1 |
| ruxolitinib (reference) | 1 | −8.63 | −8.63 | **0.800** | 3.40 | 0 |

Read that carefully, because the headline number is the least interesting part of it.

DiffSBDD produced the single best-docking molecule in the comparison, beating an approved
JAK2 inhibitor by 1.7 kcal/mol. It also produced a set whose **median** is 1.7 kcal/mol
*worse* than the GA's, whose drug-likeness is roughly half, and which trips twice as many
PAINS alerts. Its raw validity was 50% at this pocket (10 of 20) against 80% on the repo's
own example.

That is the known failure mode of structure-based generative models stated in one table: they
explore more widely than a mutation operator can, and most of what they return is chemically
poor, and a scoring function rewards some of it anyway. A single best score from a generator
like this is not evidence of anything on its own — which is exactly why the `similarity`
subsystem's safety gate exists, and why PoseBusters-style geometry checks are the obvious
next thing to add.

## What this is not, yet

Nothing here is wired into DrugOrch. It is a measured, working, reproducible install in its
own environment, deliberately kept at arm's length until there is a reason to commit to it:
the `.venv-sbdd` environment is isolated exactly as `.venv-boltz` is, and the project's own
code does not import any of it.

Wiring it in as a subsystem would mean deciding how a 1 M-parameter prior with 50% validity
and poor median chemistry earns a place in a budgeted run — which is a question the benchmark
harness can answer properly, and guesswork cannot.
