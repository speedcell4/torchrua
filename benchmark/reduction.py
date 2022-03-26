from typing import Type

import torch

from benchmark.generators import draw_token_size_lists, draw_embedding_dims, draw_devices
from benchmark.utils import TimerSuit, timeit
from torchrua import cat_sequence, reduce_catted_indices, pad_catted_sequence
from torchrua import pack_sequence, reduce_packed_indices
from torchrua import pad_sequence, reduce_padded_indices, pad_packed_sequence
from torchrua import reduce_sequence
from torchrua.reduction import reduce_packed_indices2, reduce_sequence2, reduce_padded_indices2, reduce_catted_indices2


@timeit
def reduce_packed_sequence(token_sizes: Type[draw_token_size_lists],
                           dim: Type[draw_embedding_dims],
                           device: Type[draw_devices], *,
                           timer: TimerSuit):
    token_sizes = token_sizes()
    dim = dim()
    device = device()

    sequence = data, batch_sizes, _, _ = pack_sequence([
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ], device=device)

    with timer.rua_compile:
        indices = reduce_packed_indices(batch_sizes=batch_sizes)

    with timer.rua_forward:
        actual = reduce_sequence(torch.add)(data, indices)

    with timer.naive_forward:
        excepted, _ = pad_packed_sequence(sequence, batch_first=False)
        excepted = excepted.sum(dim=0)

    with timer.rua_backward:
        _ = torch.autograd.grad(
            actual, data, torch.ones_like(actual),
            create_graph=False,
        )

    with timer.naive_backward:
        _ = torch.autograd.grad(
            excepted, data, torch.ones_like(excepted),
            create_graph=False,
        )


@timeit
def reduce_packed_sequence2(token_sizes: Type[draw_token_size_lists],
                            dim: Type[draw_embedding_dims],
                            device: Type[draw_devices], *,
                            timer: TimerSuit):
    token_sizes = token_sizes()
    dim = dim()
    device = device()

    sequence = data, batch_sizes, _, _ = pack_sequence([
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ], device=device)

    with timer.rua_compile:
        indices = reduce_packed_indices2(batch_sizes=batch_sizes)

    with timer.rua_forward:
        actual = reduce_sequence2(data, indices, op=torch.add)

    with timer.naive_forward:
        excepted, _ = pad_packed_sequence(sequence, batch_first=False)
        excepted = excepted.sum(dim=0)

    with timer.rua_backward:
        _ = torch.autograd.grad(
            actual, data, torch.ones_like(actual),
            create_graph=False,
        )

    with timer.naive_backward:
        _ = torch.autograd.grad(
            excepted, data, torch.ones_like(excepted),
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

    data, token_sizes = pad_sequence([
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ], device=device, batch_first=False)

    with timer.rua_compile:
        indices = reduce_padded_indices(token_sizes=token_sizes, batch_first=False)

    with timer.rua_forward:
        prediction = reduce_sequence(torch.add)(data, indices)

    with timer.naive_forward:
        target = data.sum(dim=0)

    with timer.rua_backward:
        _ = torch.autograd.grad(
            prediction, data, torch.ones_like(prediction),
            create_graph=False,
        )

    with timer.naive_backward:
        _ = torch.autograd.grad(
            target, data, torch.ones_like(target),
            create_graph=False,
        )


@timeit
def reduce_padded_sequence2(token_sizes: Type[draw_token_size_lists],
                            dim: Type[draw_embedding_dims],
                            device: Type[draw_devices], *,
                            timer: TimerSuit):
    token_sizes = token_sizes()
    dim = dim()
    device = device()

    data, token_sizes = pad_sequence([
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ], device=device, batch_first=False)

    with timer.rua_compile:
        indices = reduce_padded_indices2(token_sizes=token_sizes, batch_first=False)

    with timer.rua_forward:
        prediction = reduce_sequence2(data, indices, op=torch.add)

    with timer.naive_forward:
        target = data.sum(dim=0)

    with timer.rua_backward:
        _ = torch.autograd.grad(
            prediction, data, torch.ones_like(prediction),
            create_graph=False,
        )

    with timer.naive_backward:
        _ = torch.autograd.grad(
            target, data, torch.ones_like(target),
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

    sequence = data, token_sizes = cat_sequence([
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ], device=device)

    with timer.rua_compile:
        indices = reduce_catted_indices(token_sizes=token_sizes)

    with timer.rua_forward:
        prediction = reduce_sequence(torch.add)(data, indices)

    with timer.naive_forward:
        target, _ = pad_catted_sequence(sequence, batch_first=False)
        target = target.sum(dim=0)

    with timer.rua_backward:
        _ = torch.autograd.grad(
            prediction, sequence.data, torch.ones_like(prediction),
            create_graph=False,
        )

    with timer.naive_backward:
        _ = torch.autograd.grad(
            target, sequence.data, torch.ones_like(target),
            create_graph=False,
        )


@timeit
def reduce_catted_sequence2(token_sizes: Type[draw_token_size_lists],
                            dim: Type[draw_embedding_dims],
                            device: Type[draw_devices], *,
                            timer: TimerSuit):
    token_sizes = token_sizes()
    dim = dim()
    device = device()

    sequence = data, token_sizes = cat_sequence([
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ], device=device)

    with timer.rua_compile:
        indices = reduce_catted_indices2(token_sizes=token_sizes)

    with timer.rua_forward:
        prediction = reduce_sequence2(data, indices, op=torch.add)

    with timer.naive_forward:
        target, _ = pad_catted_sequence(sequence, batch_first=False)
        target = target.sum(dim=0)

    with timer.rua_backward:
        _ = torch.autograd.grad(
            prediction, data, torch.ones_like(prediction),
            create_graph=False,
        )

    with timer.naive_backward:
        _ = torch.autograd.grad(
            target, data, torch.ones_like(target),
            create_graph=False,
        )
