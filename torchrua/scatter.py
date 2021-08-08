from typing import Tuple

import torch
from torch import Tensor
from torch.types import Device

from torchrua import accumulate_sizes

__all__ = [
    'scatter_index_to_ptr',
    'scatter_add',
    'scatter_sub',
    'scatter_mean',
    'scatter_max',
    'scatter_min',
    'scatter_logsumexp',
    'scatter_softmax',
    'scatter_log_softmax',
    'scatter_multi_head_attention',
]


@torch.no_grad()
def scatter_index_to_ptr(index: Tensor,
                         dtype: torch.dtype = torch.long,
                         device: Device = None) -> Tuple[Tensor, Tensor]:
    if device is None:
        device = index.device

    index = index.to(dtype=dtype, device=device)
    _, sorted_indices = torch.sort(index, dim=0, stable=True, descending=False)

    token_sizes = torch.zeros((index.max().item() + 1,), dtype=dtype, device=device)
    token_sizes.scatter_add_(dim=0, index=index, src=torch.ones_like(index))

    return sorted_indices, accumulate_sizes(sizes=token_sizes)


def scatter_add(tensor: Tensor, index: Tensor) -> Tensor:
    indices, offsets = scatter_index_to_ptr(index=index, device=tensor.device)
    ret, _, _, _ = torch.embedding_bag(
        weight=tensor.view((tensor.size()[0], -1)),
        indices=indices, offsets=offsets, mode=0,
    )
    return ret.view((ret.size()[0], *tensor.size()[1:]))


def scatter_sub(tensor: Tensor, index: Tensor) -> Tensor:
    indices, offsets = scatter_index_to_ptr(index=index, device=tensor.device)
    ret, _, _, _ = torch.embedding_bag(
        weight=tensor.neg().view((tensor.size()[0], -1)),
        indices=indices, offsets=offsets, mode=0,
    )
    return ret.view((ret.size()[0], *tensor.size()[1:]))


def scatter_mean(tensor: Tensor, index: Tensor) -> Tensor:
    indices, offsets = scatter_index_to_ptr(index=index, device=tensor.device)
    ret, _, _, _ = torch.embedding_bag(
        weight=tensor.view((tensor.size()[0], -1)),
        indices=indices, offsets=offsets, mode=1,
    )
    return ret.view((ret.size()[0], *tensor.size()[1:]))


def scatter_max(tensor: Tensor, index: Tensor) -> Tensor:
    indices, offsets = scatter_index_to_ptr(index=index, device=tensor.device)
    ret, _, _, _ = torch.embedding_bag(
        weight=tensor.view((tensor.size()[0], -1)),
        indices=indices, offsets=offsets, mode=2,
    )
    return ret.view((ret.size()[0], *tensor.size()[1:]))


def scatter_min(tensor: Tensor, index: Tensor) -> Tensor:
    indices, offsets = scatter_index_to_ptr(index=index, device=tensor.device)
    ret, _, _, _ = torch.embedding_bag(
        weight=tensor.neg().view((tensor.size()[0], -1)),
        indices=indices, offsets=offsets, mode=2,
    )
    return ret.neg().view((ret.size()[0], *tensor.size()[1:]))


def scatter_logsumexp(tensor: Tensor, index: Tensor) -> Tensor:
    indices, offsets = scatter_index_to_ptr(index=index, device=tensor.device)
    tensor_view = tensor.view((tensor.size()[0], -1))

    with torch.no_grad():
        m, _, _, _ = torch.embedding_bag(
            weight=tensor_view,
            indices=indices, offsets=offsets, mode=2,
        )

    s, _, _, _ = torch.embedding_bag(
        weight=(tensor_view - m[index]).exp(),
        indices=indices, offsets=offsets, mode=0,
    )
    ret = torch.masked_fill(s, s == 0, 1.).log() + m
    return ret.view((ret.size()[0], *tensor.size()[1:]))


def scatter_log_softmax(tensor: Tensor, index: Tensor) -> Tensor:
    return tensor - scatter_logsumexp(tensor=tensor, index=index)[index]


def scatter_softmax(tensor: Tensor, index: Tensor) -> Tensor:
    return (tensor - scatter_logsumexp(tensor=tensor, index=index)[index]).exp()


def scatter_multi_head_attention(query: Tensor, key: Tensor, value: Tensor, tau: float, index: Tensor) -> Tensor:
    indices, offsets = scatter_index_to_ptr(index=index, device=query.device)
    score_view = (query[..., None, :] @ key[..., :, None] * tau).view((query.size()[0], -1))

    with torch.no_grad():
        m, _, _, _ = torch.embedding_bag(
            weight=score_view,
            indices=indices, offsets=offsets, mode=2,
        )

    s, _, _, _ = torch.embedding_bag(
        weight=(score_view - m[index]).exp(),
        indices=indices, offsets=offsets, mode=0,
    )
    log_partition = torch.masked_fill(s, s == 0, 1.).log() + m

    attention = (score_view - log_partition[index]).exp()
    ret, _, _, _ = torch.embedding_bag(
        weight=(attention[..., None] * value).view((attention.size()[0], -1)),
        indices=indices, offsets=offsets, mode=0,
    )

    return ret.view((ret.size()[0], *value.size()[1:]))
