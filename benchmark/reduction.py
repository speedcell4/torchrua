from typing import Type

import torch
from einops import rearrange
from torch import nn

from benchmark.generators import draw_devices, draw_batch_size_lists, draw_embedding_dims
from benchmark.packing import pack_sequence
from benchmark.utils import timeit, TimerSuit
from torchrua import cat_sequence
from torchrua import reduction as rua


@timeit
def reduce_catted_sequences(device: Type[draw_devices],
                            batch_sizes: Type[draw_batch_size_lists],
                            dim: Type[draw_embedding_dims], *,
                            timer: TimerSuit):
    device = device()
    batch_sizes = batch_sizes()
    in_dim = dim()
    hidden_dim = dim()

    sequences = [
        [
            torch.randn((token_size, in_dim), requires_grad=True, device=device)
            for token_size in draw_batch_size_lists()
        ]
        for _ in batch_sizes
    ]
    flatten_sequences = [token for sequence in sequences for token in sequence]
    catted_sequences = [cat_sequence(sequence, device=device) for sequence in sequences]
    packed_sequences = [pack_sequence(sequence, device=device) for sequence in sequences]

    rnn = nn.LSTM(
        input_size=in_dim,
        hidden_size=hidden_dim,
        bidirectional=True, bias=True,
    ).to(device=device)

    with timer.rua_compile:
        reduction_pack = rua.reduce_catted_sequences(catted_sequences, device=device)

    with timer.rua_forward:
        _, (prediction, _) = rnn(reduction_pack)
        prediction = rearrange(prediction, 'd n x -> n (d x)')

    with timer.naive_forward:
        target = []
        for pack in packed_sequences:
            _, (t, _) = rnn(pack)
            target.append(rearrange(t, 'd n x -> n (d x)'))
        target = pack_sequence(target).data

    with timer.rua_backward:
        torch.autograd.grad(
            prediction, flatten_sequences, torch.ones_like(prediction),
            create_graph=False,
        )

    with timer.naive_backward:
        torch.autograd.grad(
            target, flatten_sequences, torch.ones_like(target),
            create_graph=False,
        )
