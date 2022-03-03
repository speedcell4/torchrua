from typing import Type

import torch

from benchmark.generators import draw_token_size_lists, draw_embedding_dims, draw_devices
from benchmark.utils import TimerSuit, timeit
from torchrua import cat_sequence, reduce_catted_indices, pad_catted_sequence
from torchrua import pack_sequence, reduce_packed_indices
from torchrua import pad_sequence, reduce_padded_indices, pad_packed_sequence
from torchrua import reduce_sequence


@timeit
def reduce_packed_sequence(token_sizes: Type[draw_token_size_lists],
                           dim: Type[draw_embedding_dims],
                           device: Type[draw_devices], *,
                           timer: TimerSuit):
    token_sizes = token_sizes()
    dim = dim()
    device = device()

    inputs = pack_sequence([
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ], device=device)

    with timer.rua_compile:
        indices = reduce_packed_indices(batch_sizes=inputs.batch_sizes)

    with timer.rua_forward:
        actual = reduce_sequence(torch.add)(inputs.data, indices)

    with timer.naive_forward:
        excepted, _ = pad_packed_sequence(inputs, batch_first=False)
        excepted = excepted.sum(dim=0)

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


@timeit
def reduce_padded_sequence(token_sizes: Type[draw_token_size_lists],
                           dim: Type[draw_embedding_dims],
                           device: Type[draw_devices], *,
                           timer: TimerSuit):
    token_sizes = token_sizes()
    dim = dim()
    device = device()

    sequences = pad_sequence([
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ], device=device, batch_first=False)
    token_sizes = torch.tensor(token_sizes, device=device)

    with timer.rua_compile:
        indices = reduce_padded_indices(token_sizes=token_sizes, batch_first=False)

    with timer.rua_forward:
        prediction = reduce_sequence(torch.add)(sequences, indices)

    with timer.naive_forward:
        target = sequences.sum(dim=0)

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
def reduce_catted_sequence(token_sizes: Type[draw_token_size_lists],
                           dim: Type[draw_embedding_dims],
                           device: Type[draw_devices], *,
                           timer: TimerSuit):
    token_sizes = token_sizes()
    dim = dim()
    device = device()

    sequence = cat_sequence([
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ], device=device)

    with timer.rua_compile:
        indices = reduce_catted_indices(token_sizes=sequence.token_sizes)

    with timer.rua_forward:
        prediction = reduce_sequence(torch.add)(sequence, indices)

    with timer.naive_forward:
        target, _ = pad_catted_sequence(sequence, batch_first=False)
        target = target.sum(dim=0)

    with timer.rua_backward:
        _ = torch.autograd.grad(
            prediction, sequence, torch.ones_like(prediction),
            create_graph=False,
        )

    with timer.naive_backward:
        _ = torch.autograd.grad(
            target, sequence, torch.ones_like(target),
            create_graph=False,
        )
