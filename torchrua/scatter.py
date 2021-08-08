from typing import Tuple

import torch
from torch import Tensor
from torch.types import Device

from torchrua import accumulate_sizes

__all__ = [
    'scatter_index_to_ptr',
    'scatter_add', 'scatter_mean', 'scatter_max', 'scatter_min',
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
