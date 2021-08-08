from typing import Type

import torch
import torch_scatter

from benchmark.generators import draw_devices, draw_embedding_dims, draw_token_sizes
from benchmark.utils import timeit, TimerSuit
from torchrua import scatter as rua


@timeit
def scatter_add(device: Type[draw_devices],
                token_sizes: Type[draw_token_sizes],
                dim: Type[draw_embedding_dims], *,
                timer: TimerSuit):
    device = device()
    token_size, num = token_sizes(), token_sizes()
    if num > token_size:
        token_size, num = num, token_size
    in_dim = dim()

    inputs = torch.randn((token_size, in_dim), requires_grad=True, device=device)
    index1 = torch.randint(0, num, (token_size,), device=device)
    index2 = index1[:, None].expand_as(inputs)

    with timer.rua_forward:
        actual = rua.scatter_add(tensor=inputs, index=index1)

    with timer.naive_forward:
        excepted = torch.scatter_add(
            torch.zeros((num, in_dim), device=device),
            src=inputs, index=index2, dim=0,
        )

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


@timeit
def scatter_max(device: Type[draw_devices],
                token_sizes: Type[draw_token_sizes],
                dim: Type[draw_embedding_dims], *,
                timer: TimerSuit):
    device = device()
    token_size, num = token_sizes(), token_sizes()
    if num > token_size:
        token_size, num = num, token_size
    in_dim = dim()

    inputs = torch.randn((token_size, in_dim), requires_grad=True, device=device)
    index1 = torch.randint(0, num, (token_size,), device=device)
    index2 = index1[:, None].expand_as(inputs)

    with timer.rua_forward:
        actual = rua.scatter_max(tensor=inputs, index=index1)

    with timer.naive_forward:
        excepted, _ = torch_scatter.scatter_max(src=inputs, index=index2, dim=0)

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


@timeit
def scatter_logsumexp(device: Type[draw_devices],
                      token_sizes: Type[draw_token_sizes],
                      dim: Type[draw_embedding_dims], *,
                      timer: TimerSuit):
    device = device()
    token_size, num = token_sizes(), token_sizes()
    if num > token_size:
        token_size, num = num, token_size
    in_dim = dim()

    inputs = torch.randn((token_size, in_dim), requires_grad=True, device=device)
    index1 = torch.randint(0, num, (token_size,), device=device)
    index2 = index1[:, None].expand_as(inputs)

    with timer.rua_forward:
        actual = rua.scatter_logsumexp(tensor=inputs, index=index1)

    with timer.naive_forward:
        excepted = torch_scatter.scatter_logsumexp(src=inputs, index=index2, dim=0)

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


@timeit
def scatter_softmax(device: Type[draw_devices],
                    token_sizes: Type[draw_token_sizes],
                    dim: Type[draw_embedding_dims], *,
                    timer: TimerSuit):
    device = device()
    token_size, num = token_sizes(), token_sizes()
    if num > token_size:
        token_size, num = num, token_size
    in_dim = dim()

    inputs = torch.randn((token_size, in_dim), requires_grad=True, device=device)
    index1 = torch.randint(0, num, (token_size,), device=device)
    index2 = index1[:, None].expand_as(inputs)

    with timer.rua_forward:
        actual = rua.scatter_softmax(tensor=inputs, index=index1)

    with timer.naive_forward:
        excepted = torch_scatter.scatter_softmax(src=inputs, index=index2, dim=0)

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
