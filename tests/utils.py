from typing import List, Tuple, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.testing import assert_equal, assert_close

__all__ = [
    'assert_equal',
    'assert_close',
    'assert_packed_sequence_equal',
    'assert_packed_sequence_close',
    'assert_grad_close',
]


def assert_packed_sequence_equal(
        actual: PackedSequence,
        expected: PackedSequence,
        check_device: bool = True, check_dtype: bool = True, check_stride: bool = True) -> None:
    kwargs = dict(check_device=check_device, check_dtype=check_dtype, check_stride=check_stride)

    assert_equal(actual.data, expected.data, **kwargs)
    assert_close(actual.batch_sizes, expected.batch_sizes, **kwargs)

    if actual.sorted_indices is None:
        assert expected.sorted_indices is None
    else:
        assert_close(actual.sorted_indices, expected.sorted_indices, **kwargs)

    if actual.unsorted_indices is None:
        assert expected.unsorted_indices is None
    else:
        assert_close(actual.unsorted_indices, expected.unsorted_indices, **kwargs)


def assert_packed_sequence_close(
        actual: PackedSequence,
        expected: PackedSequence,
        check_device: bool = True, check_dtype: bool = True, check_stride: bool = True) -> None:
    kwargs = dict(check_device=check_device, check_dtype=check_dtype, check_stride=check_stride)

    assert_close(actual.data, expected.data, **kwargs)
    assert_close(actual.batch_sizes, expected.batch_sizes, **kwargs)

    if actual.sorted_indices is None:
        assert expected.sorted_indices is None
    else:
        assert_close(actual.sorted_indices, expected.sorted_indices, **kwargs)

    if actual.unsorted_indices is None:
        assert expected.unsorted_indices is None
    else:
        assert_close(actual.unsorted_indices, expected.unsorted_indices, **kwargs)


def assert_grad_close(
        actual: Tensor,
        expected: Tensor,
        inputs: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
        check_device: bool = True, check_dtype: bool = True, check_stride: bool = True) -> None:
    kwargs = dict(check_device=check_device, check_dtype=check_dtype, check_stride=check_stride)

    grad = torch.rand_like(actual)

    actual_grads = torch.autograd.grad(
        actual, inputs, grad,
        create_graph=False,
    )

    expected_grads = torch.autograd.grad(
        expected, inputs, grad,
        create_graph=False,
    )

    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert_close(actual_grad, expected_grad, **kwargs)


def assert_packed_grad_close(
        actual: PackedSequence,
        expected: PackedSequence,
        inputs: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
        check_device: bool = True, check_dtype: bool = True, check_stride: bool = True) -> None:
    kwargs = dict(check_device=check_device, check_dtype=check_dtype, check_stride=check_stride)

    grad = torch.rand_like(actual.data)

    actual_grads = torch.autograd.grad(
        actual.data, inputs, grad,
        create_graph=False,
    )

    expected_grads = torch.autograd.grad(
        expected.data, inputs, grad,
        create_graph=False,
    )

    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert_close(actual_grad, expected_grad, **kwargs)
