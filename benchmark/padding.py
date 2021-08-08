from typing import Type

import torch
from torch.nn.utils import rnn as tgt

from benchmark.generators import draw_token_size_lists, draw_embedding_dims, draw_devices
from benchmark.utils import TimerSuit, timeit
from torchrua import padding as rua


@timeit
def pad_sequence(token_sizes: Type[draw_token_size_lists],
                 dim: Type[draw_embedding_dims],
                 device: Type[draw_devices],
                 batch_first: bool = True, *,
                 timer: TimerSuit):
    token_sizes = token_sizes()
    dim = dim()
    device = device()

    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    with timer.rua_forward:
        actual = rua.pad_sequence(inputs, batch_first=batch_first)

    with timer.naive_forward:
        excepted = tgt.pad_sequence(inputs, batch_first=batch_first)

    with timer.rua_backward:
        _ = torch.autograd.grad(
            actual, inputs, torch.ones_like(actual),
            create_graph=False,
        )

    with timer.naive_backward:
        _ = torch.autograd.grad(
            excepted, inputs, torch.ones_like(excepted),
            create_graph=False,
        )


@timeit
def pad_packed_sequence(token_sizes: Type[draw_token_size_lists],
                        dim: Type[draw_embedding_dims],
                        device: Type[draw_devices],
                        batch_first: bool = True, *,
                        timer: TimerSuit):
    token_sizes = token_sizes()
    dim = dim()
    device = device()

    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    inputs = tgt.pack_sequence(inputs, enforce_sorted=False)

    with timer.rua_forward:
        actual, _ = rua.pad_packed_sequence(inputs, batch_first=batch_first)

    with timer.naive_forward:
        excepted, _ = tgt.pad_packed_sequence(inputs, batch_first=batch_first)

    with timer.rua_backward:
        _ = torch.autograd.grad(
            actual, inputs.data, torch.ones_like(actual),
            create_graph=False,
        )

    with timer.naive_backward:
        _ = torch.autograd.grad(
            excepted, inputs.data, torch.ones_like(excepted),
            create_graph=False,
        )
