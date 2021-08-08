from typing import Type

import torch
import torch_scatter

from benchmark.generators import draw_devices, draw_embedding_dims, draw_token_sizes
from benchmark.utils import timeit, TimerSuit
from torchrua.scatter import scatter_add as rua_scatter_add


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

    tensor = torch.randn((token_size, in_dim), requires_grad=True, device=device)
    index = torch.randint(0, num, (token_size,), device=device)

    with timer.rua_forward:
        prediction = rua_scatter_add(tensor=tensor, index=index)

    with timer.naive_forward:
        target = torch_scatter.scatter_add(src=tensor, index=index, dim=0)

    with timer.rua_backward:
        torch.autograd.grad(
            prediction, tensor, torch.ones_like(prediction),
            create_graph=False,
        )

    with timer.naive_backward:
        torch.autograd.grad(
            target, tensor, torch.ones_like(target),
            create_graph=False,
        )
