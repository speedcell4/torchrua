from typing import Type

import torch
from torch.nn.utils.rnn import pad_sequence, pack_sequence

from benchmark.generators import draw_token_size_lists, draw_embedding_dims, draw_devices
from benchmark.utils import TimerSuit, timeit
from torchrua import tree_reduction_indices, tree_reduce_packed_sequence


@timeit
def tree_reduce(device: Type[draw_devices],
                token_sizes: Type[draw_token_size_lists],
                dim: Type[draw_embedding_dims], *,
                timer: TimerSuit):
    device = device()
    token_sizes = token_sizes()
    dim = dim()

    sequences = [
        torch.randn((token_size, dim), requires_grad=True, device=device)
        for token_size in token_sizes
    ]

    packed_sequence = pack_sequence(sequences, enforce_sorted=False)

    with timer.rua_compile:
        reduction_indices = tree_reduction_indices(batch_sizes=packed_sequence.batch_sizes.to(device=device))

    with timer.rua_forward:
        rua = tree_reduce_packed_sequence(torch.add)(packed_sequence.data, reduction_indices=reduction_indices)
        rua = rua[packed_sequence.unsorted_indices]

    with timer.naive_forward:
        naive = pad_sequence(sequences, batch_first=False).sum(dim=0)

    with timer.rua_backward:
        _, = torch.autograd.grad(
            rua, packed_sequence.data, torch.ones_like(rua),
            create_graph=False, allow_unused=True, only_inputs=True,
        )

    with timer.naive_backward:
        _, = torch.autograd.grad(
            naive, packed_sequence.data, torch.ones_like(naive),
            create_graph=False, allow_unused=True, only_inputs=True,
        )
