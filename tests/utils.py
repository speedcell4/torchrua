from typing import List, Tuple, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.testing import assert_equal, assert_close


def assert_packed_equal(actual: PackedSequence, expected: PackedSequence) -> None:
    assert_equal(actual.data, expected.data)
    assert_close(actual.batch_sizes, expected.batch_sizes)

    if actual.sorted_indices is None:
        assert expected.sorted_indices is None
    else:
        assert_close(actual.sorted_indices, expected.sorted_indices)

    if actual.unsorted_indices is None:
        assert expected.unsorted_indices is None
    else:
        assert_close(actual.unsorted_indices, expected.unsorted_indices)


def assert_packed_close(actual: PackedSequence, expected: PackedSequence) -> None:
    assert_close(actual.data, expected.data)
    assert_close(actual.batch_sizes, expected.batch_sizes)

    if actual.sorted_indices is not None:
        assert expected.sorted_indices is not None
        assert_close(actual.sorted_indices, expected.sorted_indices)

    if actual.unsorted_indices is not None:
        assert expected.unsorted_indices is not None
        assert_close(actual.unsorted_indices, expected.unsorted_indices)


def assert_grad_close(actual: Tensor, expected: Tensor, inputs: Union[List[Tensor], Tuple[Tensor, ...]]) -> None:
    grad = torch.rand_like(actual)

    actual_grad = torch.autograd.grad(
        actual, inputs, grad,
        create_graph=False,
    )

    expected_grad = torch.autograd.grad(
        expected, inputs, grad,
        create_graph=False,
    )

    for a_grad, e_grad in zip(actual_grad, expected_grad):
        assert_close(a_grad, e_grad)


def assert_packed_grad_close(actual: PackedSequence, expected: PackedSequence,
                             inputs: Union[List[Tensor], Tuple[Tensor, ...]]) -> None:
    grad = torch.rand_like(actual.data)

    actual_grad = torch.autograd.grad(
        actual.data, inputs, grad,
        create_graph=False,
    )

    expected_grad = torch.autograd.grad(
        expected.data, inputs, grad,
        create_graph=False,
    )

    for a_grad, e_grad in zip(actual_grad, expected_grad):
        assert_close(a_grad, e_grad)
