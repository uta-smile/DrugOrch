"""The pure-PyTorch stand-in for torch-scatter, checked against hand-computed values.

DiffSBDD runs a diffusion trajectory through ``scatter_add`` and ``scatter_mean`` tens of
times per sampling step. If either disagreed with ``torch_scatter`` the model would not
fail -- it would produce coordinates that are subtly wrong, and the molecules coming out
would look plausible and be garbage. A smoke test that only checked shapes would pass
against every one of the mistakes that matter, so these check values, and in particular the
empty-segment cases where the two obvious implementations differ.

Skipped when the shim is not on the path: it lives in the DiffSBDD environment, not the
project one, so this is only meaningful where that environment is present.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SHIM = Path(__file__).resolve().parents[1] / "external" / "torch_scatter_shim"

pytest.importorskip("torch", reason="torch is only installed in the model environments")

if SHIM.is_dir() and str(SHIM) not in sys.path:
    sys.path.insert(0, str(SHIM))

if importlib.util.find_spec("torch_scatter") is None:  # pragma: no cover
    pytest.skip("the torch_scatter shim is not present", allow_module_level=True)

import torch  # noqa: E402
from torch_scatter import scatter_add, scatter_mean  # noqa: E402


def test_add_sums_each_segment():
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = torch.tensor([0, 1, 0])

    out = scatter_add(src, index, dim=0)

    assert torch.equal(out, torch.tensor([[6.0, 8.0], [3.0, 4.0]]))


def test_mean_averages_each_segment():
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = torch.tensor([0, 1, 0])

    out = scatter_mean(src, index, dim=0)

    assert torch.allclose(out, torch.tensor([[3.0, 4.0], [3.0, 4.0]]))


def test_an_empty_segment_sums_to_zero_not_nothing():
    """Segment 1 has no members. It must appear in the output as a zero row."""
    src = torch.tensor([[2.0], [4.0]])
    index = torch.tensor([0, 2])

    out = scatter_add(src, index, dim=0, dim_size=4)

    assert torch.equal(out, torch.tensor([[2.0], [0.0], [4.0], [0.0]]))


def test_an_empty_segment_averages_to_zero_not_nan():
    """The case that would poison a diffusion trajectory without ever raising.

    The natural implementation divides a zero sum by a zero count and yields NaN.
    ``torch_scatter`` defines this as 0, and a NaN here would propagate through every
    later denoising step and come out as a molecule that looks fine and is not.
    """
    src = torch.tensor([[2.0], [4.0]])
    index = torch.tensor([0, 2])

    out = scatter_mean(src, index, dim=0, dim_size=4)

    assert not torch.isnan(out).any(), "an empty segment produced NaN"
    assert torch.equal(out, torch.tensor([[2.0], [0.0], [4.0], [0.0]]))


def test_dim_size_defaults_to_the_largest_index():
    src = torch.ones(3, 2)
    index = torch.tensor([0, 0, 5])

    assert scatter_add(src, index, dim=0).shape == (6, 2)


def test_dim_size_may_be_larger_than_the_indices_need():
    src = torch.ones(2, 3)
    index = torch.tensor([0, 1])

    assert scatter_add(src, index, dim=0, dim_size=7).shape == (7, 3)


def test_three_dimensional_sources_are_handled():
    """DiffSBDD scatters coordinate tensors, not only flat feature rows."""
    src = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    index = torch.tensor([0, 0, 1, 1])

    out = scatter_add(src, index, dim=0)

    assert torch.equal(out, torch.tensor([[3.0, 5.0, 7.0], [15.0, 17.0, 19.0]]))


def test_an_unsupported_dimension_raises_rather_than_quietly_differing():
    """A shim that accepted an argument it ignores would be worse than one that refuses."""
    src = torch.ones(2, 2)
    index = torch.tensor([0, 1])

    with pytest.raises(NotImplementedError, match="dim=0"):
        scatter_add(src, index, dim=1)
    with pytest.raises(NotImplementedError, match="out="):
        scatter_add(src, index, dim=0, out=torch.zeros(2, 2))


def test_it_agrees_with_the_authors_own_native_implementation():
    """DiffSBDD's EGNN already contains this operation, written by its authors.

    Checking against it rather than only against hand-computed values ties the shim to the
    behaviour the model was trained with.
    """

    def unsorted_segment_sum(data, segment_ids, num_segments):
        result = data.new_full((num_segments, data.size(1)), 0)
        ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, ids, data)
        return result

    torch.manual_seed(0)
    src = torch.randn(40, 5)
    index = torch.randint(0, 7, (40,))

    assert torch.allclose(scatter_add(src, index, dim=0, dim_size=7),
                          unsorted_segment_sum(src, index, 7))
