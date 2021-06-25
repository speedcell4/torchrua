from typing import Type

import torch

from benchmark.generators import draw_token_size_lists, draw_embedding_dims, draw_devices
from benchmark.utils import TimerSuit, timeit
from torchrua import packing as rua
from torch.nn.utils import rnn as tgt


@timeit
def pack_sequence(token_sizes: Type[draw_token_size_lists],
                  dim: Type[draw_embedding_dims],
                  device: Type[draw_devices], *,
                  timer: TimerSuit):
    token_sizes = token_sizes()
    dim = dim()
    device = device()

    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    with timer.rua_forward:
        prediction = rua.pack_sequence(sequences)

    with timer.naive_forward:
        target = tgt.pack_sequence(sequences, enforce_sorted=False)

    with timer.rua_backward:
        _ = torch.autograd.grad(
            prediction.data, sequences, torch.ones_like(prediction.data),
            create_graph=False,
        )

    with timer.naive_backward:
        _ = torch.autograd.grad(
            target.data, sequences, torch.ones_like(target.data),
            create_graph=False,
        )
