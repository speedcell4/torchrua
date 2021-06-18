import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

RTOL = 1e-5
ATOL = 1e-5


def assert_equal(x: Tensor, y: Tensor) -> None:
    assert torch.equal(x, y), f'{x} != {y}'


def assert_close(x: Tensor, y: Tensor) -> None:
    assert torch.allclose(x, y, rtol=RTOL, atol=ATOL), f'{x} != {y}'


def assert_packed_equal(x: PackedSequence, y: PackedSequence) -> None:
    assert_equal(x.data, y.data)
    assert_close(x.batch_sizes, y.batch_sizes)

    if x.sorted_indices is None:
        assert y.sorted_indices is None
    else:
        assert_close(x.sorted_indices, y.sorted_indices)

    if x.unsorted_indices is None:
        assert y.unsorted_indices is None
    else:
        assert_close(x.unsorted_indices, y.unsorted_indices)


def assert_packed_close(x: PackedSequence, y: PackedSequence) -> None:
    assert_close(x.data, y.data)
    assert_close(x.batch_sizes, y.batch_sizes)

    if x.sorted_indices is None:
        assert y.sorted_indices is None
    else:
        assert_close(x.sorted_indices, y.sorted_indices)

    if x.unsorted_indices is None:
        assert y.unsorted_indices is None
    else:
        assert_close(x.unsorted_indices, y.unsorted_indices)
