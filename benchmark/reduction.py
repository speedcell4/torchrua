from typing import Type

import torch
from torch.nn.utils import rnn as tgt

from benchmark.generators import draw_token_size_lists, draw_embedding_dims, draw_devices
from benchmark.utils import TimerSuit, timeit
from torchrua import reduction as rua
from torchrua.padding import pad_packed_sequence


@timeit
def tree_reduce_packed_sequence(device: Type[draw_devices],
                                token_sizes: Type[draw_token_size_lists],
                                dim: Type[draw_embedding_dims], *,
                                timer: TimerSuit):
    device = device()
    token_sizes = token_sizes()
    dim = dim()

    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    sequences = tgt.pack_sequence(sequences, enforce_sorted=False)

    with timer.rua_compile:
        reduction_indices = rua.tree_reduce_packed_indices(batch_sizes=sequences.batch_sizes.to(device=device))

    with timer.rua_forward:
        prediction = rua.tree_reduce_packed_sequence(torch.add)(sequences.data, reduction_indices=reduction_indices)
        prediction = prediction[sequences.unsorted_indices]

    with timer.naive_forward:
        target, _ = pad_packed_sequence(sequences, batch_first=False, padding_value=0)
        target = target.sum(dim=0)

    with timer.rua_backward:
        _, = torch.autograd.grad(
            prediction, sequences.data, torch.ones_like(prediction),
            create_graph=False,
        )

    with timer.naive_backward:
        _, = torch.autograd.grad(
            target, sequences.data, torch.ones_like(target),
            create_graph=False,
        )
