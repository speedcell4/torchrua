from typing import List, Tuple, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

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
    data_x, token_sizes_x = pad_packed_sequence(x)
    data_y, token_sizes_y = pad_packed_sequence(y)

    assert_close(data_x, data_y)
    assert_equal(token_sizes_x, token_sizes_y)


def assert_grad_close(prediction: Tensor, target: Tensor, inputs: Union[List[Tensor], Tuple[Tensor, ...]]) -> None:
    grad = torch.rand_like(prediction)

    prediction = torch.autograd.grad(
        prediction, inputs, grad,
        create_graph=False,
    )

    target = torch.autograd.grad(
        target, inputs, grad,
        create_graph=False,
    )

    for grad_p, grad_t in zip(prediction, target):
        assert_close(grad_p, grad_t)


def assert_packed_grad_close(prediction: PackedSequence, target: PackedSequence,
                             inputs: Union[List[Tensor], Tuple[Tensor, ...]]) -> None:
    prediction, _ = pad_packed_sequence(prediction)
    target, _ = pad_packed_sequence(target)

    grad = torch.rand_like(prediction)

    prediction = torch.autograd.grad(
        prediction, inputs, grad,
        create_graph=False,
    )

    target = torch.autograd.grad(
        target, inputs, grad,
        create_graph=False,
    )

    for grad_p, grad_t in zip(prediction, target):
        assert_close(grad_p, grad_t)
