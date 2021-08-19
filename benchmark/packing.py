from typing import Type

import torch
from torch.nn.utils import rnn as tgt

from benchmark.generators import draw_token_size_lists, draw_embedding_dims, draw_devices
from benchmark.utils import TimerSuit, timeit
from torchrua import packing as rua


@timeit
def pack_sequence(token_sizes: Type[draw_token_size_lists],
                  dim: Type[draw_embedding_dims],
                  device: Type[draw_devices], *,
                  timer: TimerSuit):
    token_sizes = token_sizes()
    dim = dim()
    device = device()

    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    with timer.rua_forward:
        actual = rua.pack_sequence(inputs)

    with timer.naive_forward:
        excepted = tgt.pack_sequence(inputs, enforce_sorted=False)

    with timer.rua_backward:
        _ = torch.autograd.grad(
            actual.data, inputs, torch.ones_like(actual.data),
            create_graph=False,
        )

    with timer.naive_backward:
        _ = torch.autograd.grad(
            excepted.data, inputs, torch.ones_like(excepted.data),
            create_graph=False,
        )


@timeit
def pack_padded_sequence(token_sizes: Type[draw_token_size_lists],
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
    inputs = tgt.pad_sequence(inputs, batch_first=batch_first)
    token_sizes = torch.tensor(token_sizes, device=torch.device('cpu'))

    with timer.rua_forward:
        actual = rua.pack_padded_sequence(inputs, token_sizes, batch_first=batch_first)

    with timer.naive_forward:
        excepted = tgt.pack_padded_sequence(inputs, token_sizes, batch_first=batch_first, enforce_sorted=False)

    with timer.rua_backward:
        _ = torch.autograd.grad(
            actual.data, inputs, torch.ones_like(actual.data),
            create_graph=False,
        )

    with timer.naive_backward:
        _ = torch.autograd.grad(
            excepted.data, inputs, torch.ones_like(excepted.data),
            create_graph=False,
        )
