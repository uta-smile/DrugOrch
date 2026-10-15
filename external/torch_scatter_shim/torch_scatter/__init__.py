"""A pure-PyTorch stand-in for the two ``torch_scatter`` functions DiffSBDD uses.

**Why this exists.** ``torch-scatter`` ships CUDA kernels that must be compiled against the
exact torch build in use. There is no prebuilt wheel for torch 2.11 + cu128 on Windows and
CPython 3.12, and building one needs MSVC, the CUDA toolkit, and a compile targeting sm_120 --
a large dependency to take on for two functions, and one more thing to break on the next
torch upgrade.

DiffSBDD imports exactly ``scatter_add`` and ``scatter_mean``, and calls both with ``dim=0``
at every one of its ~35 call sites. That is the simplest case of a scatter reduction and
``Tensor.scatter_add_`` does it natively. So this provides those two, with ``torch_scatter``'s
signature, and nothing else -- an honest shim rather than a reimplementation of the library.

**Correctness.** The implementation follows the one the DiffSBDD authors already wrote for
their own EGNN layer (``equivariant_diffusion/egnn_new.py::unsorted_segment_sum``), which is
the same operation and is used on the model's hot path. Three semantics have to match
``torch_scatter`` exactly or a diffusion trajectory drifts silently rather than failing:

* **empty segments sum to zero**, which ``scatter_add_`` into a zero-filled tensor gives;
* **empty segments average to zero, not NaN** -- ``torch_scatter`` defines ``0/0`` as ``0``
  here, so the count is clamped to a minimum of one before dividing;
* **``dim_size=None`` means ``index.max() + 1``**, so an output is never shorter than the
  indices addressing it.

``tests/test_scatter_shim.py`` checks all three against hand-computed values, including the
empty-segment cases, which are the ones a smoke test would miss.

**Scope.** ``dim`` must be 0 and ``out`` is not supported; both raise rather than silently
doing something else. DiffSBDD never needs either, and a shim that quietly accepted an
argument it ignores would be worse than one that refuses.
"""

from __future__ import annotations

import torch

__all__ = ["scatter_add", "scatter_mean"]
__version__ = "0.0.0+drugorch-shim"


def _check(dim: int, out: torch.Tensor | None) -> None:
    if dim != 0:
        raise NotImplementedError(
            f"this torch_scatter shim only implements dim=0, got dim={dim}. "
            "Install the real torch-scatter if another dimension is needed."
        )
    if out is not None:
        raise NotImplementedError("this torch_scatter shim does not implement `out=`")


def _expand(index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Broadcast a 1-D segment index across ``src``'s trailing dimensions.

    ``scatter_add_`` wants an index the same shape as the source; ``torch_scatter`` takes one
    value per row. Done with ``expand`` rather than ``repeat`` so no memory is copied.
    """
    if index.dim() != 1:
        raise NotImplementedError("this shim expects a 1-D index, one entry per row")
    view = (-1,) + (1,) * (src.dim() - 1)
    return index.view(view).expand_as(src)


def _size(src: torch.Tensor, index: torch.Tensor, dim_size: int | None) -> tuple[int, ...]:
    n = int(dim_size) if dim_size is not None else int(index.max().item()) + 1 if index.numel() else 0
    return (n,) + tuple(src.shape[1:])


def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Sum the rows of ``src`` into the segments named by ``index``."""
    _check(dim, out)
    result = src.new_zeros(_size(src, index, dim_size))
    if src.numel():
        result.scatter_add_(0, _expand(index, src), src)
    return result


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Average the rows of ``src`` over the segments named by ``index``.

    A segment with no members averages to zero rather than NaN, matching
    ``torch_scatter``. Getting this wrong would not raise -- it would put NaNs into a
    diffusion trajectory and produce molecules that are quietly garbage.
    """
    _check(dim, out)
    total = scatter_add(src, index, dim_size=dim_size)
    counts = src.new_zeros(total.shape[0])
    if index.numel():
        counts.scatter_add_(0, index, src.new_ones(index.shape[0]))
    counts = counts.clamp(min=1).view((-1,) + (1,) * (total.dim() - 1))
    return total / counts
