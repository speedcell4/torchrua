from typing import Type

import torch
from einops import rearrange
from torch import Tensor

from benchmark.generators import draw_batch_sizes
from benchmark.generators import draw_token_sizes, draw_devices, draw_embedding_dims
from benchmark.utils import timeit, TimerSuit
from torchrua import attention_indices, scatter_multi_head_attention
from torchrua import pad_sequence, cat_sequence, token_sizes_to_mask


def multi_head_attention(q: Tensor, k: Tensor, v: Tensor, tau: float, mask: Tensor) -> Tensor:
    q = rearrange(q, '... q h x -> ... h q x')
    k = rearrange(k, '... k h x -> ... h x k')
    v = rearrange(v, '... k h x -> ... h k x')

    scores = q @ k * tau
    scores, mask = torch.broadcast_tensors(scores, mask[..., None, None, :])
    scores = torch.masked_fill(scores, mask=mask, value=-float('inf'))

    attention = torch.softmax(scores, dim=-1)
    return attention @ v


@timeit
def scatter_attention(device: Type[draw_devices],
                      batch_sizes: Type[draw_batch_sizes],
                      token_sizes: Type[draw_token_sizes],
                      dim: Type[draw_embedding_dims], *,
                      timer: TimerSuit):
    device = device()
    batch_size = batch_sizes()
    in_dim = dim()

    q_token_sizes = [token_sizes() for _ in range(batch_size)]
    k_token_sizes = [token_sizes() for _ in range(batch_size)]

    q = [
        torch.randn((token_size, 2, in_dim), requires_grad=True, device=device)
        for token_size in q_token_sizes
    ]
    k = [
        torch.randn((token_size, 2, in_dim), requires_grad=True, device=device)
        for token_size in k_token_sizes
    ]
    v = [
        torch.randn((token_size, 2, in_dim), requires_grad=True, device=device)
        for token_size in k_token_sizes
    ]

    padded_q = pad_sequence(q, batch_first=True)
    padded_k = pad_sequence(k, batch_first=True)
    padded_v = pad_sequence(v, batch_first=True)

    catted_q, q_token_sizes = cat_sequence(q)
    catted_k, k_token_sizes = cat_sequence(k)
    catted_v, v_token_sizes = cat_sequence(v)

    mask = token_sizes_to_mask(k_token_sizes, batch_first=True)

    with timer.rua_compile:
        q_ptr, k_ptr = attention_indices(q_token_sizes=q_token_sizes, k_token_sizes=k_token_sizes)

    with timer.rua_forward:
        prediction = scatter_multi_head_attention(
            catted_q[q_ptr], catted_k[k_ptr], catted_v[k_ptr],
            tau=1., q_ptr=q_ptr,
        )

    with timer.naive_forward:
        target = multi_head_attention(
            padded_q, padded_k, padded_v,
            tau=1., mask=mask,
        )

    with timer.rua_backward:
        torch.autograd.grad(
            prediction, (catted_q, catted_k, catted_v), torch.ones_like(prediction),
            create_graph=False,
        )

    with timer.naive_backward:
        torch.autograd.grad(
            target, (padded_q, padded_k, padded_v), torch.ones_like(target),
            create_graph=False,
        )
