import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.testing import assert_close

from torchrua.catting import CattedSequence

__all__ = [
    'assert_close',
    'assert_grad_close',
    'assert_catted_sequence_close',
    'assert_packed_sequence_close',
]


def assert_grad_close(actual: Tensor, expected: Tensor, inputs, **kwargs) -> None:
    grad = torch.rand_like(actual)

    actual_grads = torch.autograd.grad(actual, inputs, grad, retain_graph=True, allow_unused=False)
    expected_grads = torch.autograd.grad(expected, inputs, grad, retain_graph=True, allow_unused=False)

    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert_close(actual=actual_grad, expected=expected_grad, **kwargs)


def assert_catted_sequence_close(actual: CattedSequence, expected: CattedSequence, **kwargs) -> None:
    assert_close(actual=actual.data, expected=expected.data, **kwargs)
    assert_close(actual=actual.token_sizes, expected=expected.token_sizes, **kwargs)


def assert_packed_sequence_close(actual: PackedSequence, expected: PackedSequence, **kwargs) -> None:
    assert_close(actual=actual.data, expected=expected.data, **kwargs)
    assert_close(actual=actual.batch_sizes, expected=expected.batch_sizes, **kwargs)

    if actual.sorted_indices is None:
        assert expected.sorted_indices is None
    else:
        assert_close(actual=actual.sorted_indices, expected=expected.sorted_indices, **kwargs)

    if actual.unsorted_indices is None:
        assert expected.unsorted_indices is None
    else:
        assert_close(actual=actual.unsorted_indices, expected=expected.unsorted_indices, **kwargs)
