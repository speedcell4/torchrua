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
    inputs = [token for sequence in sequences for token in sequence]
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
        _, (actual, _) = rnn(reduction_pack)
        actual = rearrange(actual, 'd n x -> n (d x)')

    with timer.naive_forward:
        excepted = []
        for pack in packed_sequences:
            _, (t, _) = rnn(pack)
            excepted.append(rearrange(t, 'd n x -> n (d x)'))
        excepted = pack_sequence(excepted).data

    with timer.rua_backward:
        torch.autograd.grad(
            actual, inputs, torch.ones_like(actual),
            create_graph=False,
        )

    with timer.naive_backward:
        torch.autograd.grad(
            excepted, inputs, torch.ones_like(excepted),
            create_graph=False,
        )
