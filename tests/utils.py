from typing import List, Tuple, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.testing import assert_close

from torchrua.catting import CattedSequence

__all__ = [
    'assert_equal', 'assert_close', 'assert_grad_close',
    'assert_catted_sequence_equal', 'assert_catted_sequence_close',
    'assert_packed_sequence_equal', 'assert_packed_sequence_close',
]


def assert_equal(actual: Tensor, expected: Tensor) -> None:
    assert torch.equal(actual, expected)


def assert_grad_close(
        actual: Tensor, expected: Tensor,
        inputs: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
        allow_unused: bool = False,
        check_device: bool = True, check_dtype: bool = True, check_stride: bool = True) -> None:
    kwargs = dict(check_device=check_device, check_dtype=check_dtype, check_stride=check_stride)

    grad = torch.rand_like(actual)

    actual_grads = torch.autograd.grad(
        actual, inputs, grad,
        create_graph=False,
        allow_unused=allow_unused,
    )

    expected_grads = torch.autograd.grad(
        expected, inputs, grad,
        create_graph=False,
        allow_unused=allow_unused,
    )

    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert_close(actual=actual_grad, expected=expected_grad, **kwargs)


def assert_catted_sequence_close(
        actual: CattedSequence, expected: CattedSequence,
        check_device: bool = True, check_dtype: bool = True, check_stride: bool = True) -> None:
    kwargs = dict(check_device=check_device, check_dtype=check_dtype, check_stride=check_stride)

    assert_close(actual=actual.data, expected=expected.data, **kwargs)
    assert_equal(actual=actual.token_sizes, expected=expected.token_sizes)


def assert_catted_sequence_equal(actual: CattedSequence, expected: CattedSequence) -> None:
    assert_equal(actual=actual.data, expected=expected.data)
    assert_equal(actual=actual.token_sizes, expected=expected.token_sizes)


def assert_packed_sequence_close(
        actual: PackedSequence, expected: PackedSequence,
        check_device: bool = True, check_dtype: bool = True, check_stride: bool = True) -> None:
    kwargs = dict(check_device=check_device, check_dtype=check_dtype, check_stride=check_stride)

    assert_close(actual=actual.data, expected=expected.data, **kwargs)
    assert_equal(actual=actual.batch_sizes, expected=expected.batch_sizes)

    if actual.sorted_indices is None:
        assert expected.sorted_indices is None
    else:
        assert_equal(actual=actual.sorted_indices, expected=expected.sorted_indices)

    if actual.unsorted_indices is None:
        assert expected.unsorted_indices is None
    else:
        assert_equal(actual=actual.unsorted_indices, expected=expected.unsorted_indices)


def assert_packed_sequence_equal(actual: PackedSequence, expected: PackedSequence) -> None:
    assert_equal(actual=actual.data, expected=expected.data)
    assert_equal(actual=actual.batch_sizes, expected=expected.batch_sizes)

    if actual.sorted_indices is None:
        assert expected.sorted_indices is None
    else:
        assert_equal(actual=actual.sorted_indices, expected=expected.sorted_indices)

    if actual.unsorted_indices is None:
        assert expected.unsorted_indices is None
    else:
        assert_equal(actual=actual.unsorted_indices, expected=expected.unsorted_indices)
