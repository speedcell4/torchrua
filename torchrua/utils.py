from typing import Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import invert_permutation
from torch.types import Device

__all__ = [
    'accumulate_sizes',
    'resize_sizes',
    'sizes_to_sorting_indices',
]


@torch.no_grad()
def accumulate_sizes(sizes: Tensor) -> Tensor:
    return F.pad(sizes.cumsum(dim=0), pad=[1, -1])


@torch.no_grad()
def resize_sizes(sizes: Tensor, n: int) -> Tensor:
    if n <= sizes.size()[0]:
        assert sizes[0] == sizes[-n]
        return sizes[-n:]
    return F.pad(sizes, [n - sizes.size()[0], 0], value=sizes[0])


@torch.no_grad()
def sizes_to_sorting_indices(sizes: Tensor, device: Device = None) -> Tuple[Tensor, Tensor, Tensor]:
    if device is None:
        device = sizes.device

    sorted_sizes, sorted_indices = sizes.cpu().sort(dim=0, descending=True)
    sorted_sizes = sorted_sizes.to(device=device)
    sorted_indices = sorted_indices.to(device=device)
    unsorted_indices = invert_permutation(sorted_indices)

    return sorted_sizes, sorted_indices, unsorted_indices
