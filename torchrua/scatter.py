from typing import Tuple

import torch
from torch import Tensor
from torch.types import Device

from torchrua.utils import accumulate_sizes

__all__ = [
    'scatter_index_to_ptr',
    'scatter_add',
    'scatter_mean',
    'scatter_max',
    'scatter_min',
    'scatter_logsumexp',
    'scatter_softmax',
    'scatter_log_softmax',
]


@torch.no_grad()
def scatter_index_to_ptr(index: Tensor,
                         dtype: torch.dtype = torch.long,
                         device: Device = None) -> Tuple[Tensor, Tensor]:
    if device is None:
        device = index.device

    index = index.to(dtype=dtype, device=device)
    sorted_indices = torch.argsort(index, dim=0, descending=False)

    token_ptr = torch.arange(index.max().item() + 1, dtype=dtype, device=device)
    tb_mask = token_ptr[:, None] == index[None, :]
    token_sizes = tb_mask.long().sum(dim=-1)

    return sorted_indices, accumulate_sizes(sizes=token_sizes)


def scatter_add(tensor: Tensor, index: Tensor) -> Tensor:
    indices, offsets = scatter_index_to_ptr(index=index, device=tensor.device)
    ret, _, _, _ = torch.embedding_bag(
        weight=tensor.view((tensor.size()[0], -1)),
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
