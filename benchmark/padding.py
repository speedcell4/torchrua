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

    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    with timer.rua_forward:
        prediction = rua.pad_sequence(sequences, batch_first=batch_first)

    with timer.naive_forward:
        target = tgt.pad_sequence(sequences, batch_first=batch_first)

    with timer.rua_backward:
        _ = torch.autograd.grad(
            prediction, sequences, torch.ones_like(prediction),
            create_graph=False,
        )

    with timer.naive_backward:
        _ = torch.autograd.grad(
            target, sequences, torch.ones_like(target),
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

    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    sequences = tgt.pack_sequence(sequences, enforce_sorted=False)

    with timer.rua_forward:
        prediction, _ = rua.pad_packed_sequence(sequences, batch_first=batch_first)

    with timer.naive_forward:
        target, _ = tgt.pad_packed_sequence(sequences, batch_first=batch_first)

    with timer.rua_backward:
        _ = torch.autograd.grad(
            prediction, sequences.data, torch.ones_like(prediction),
            create_graph=False,
        )

    with timer.naive_backward:
        _ = torch.autograd.grad(
            target, sequences.data, torch.ones_like(target),
            create_graph=False,
        )
