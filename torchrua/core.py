from typing import Tuple, Optional, NamedTuple

import torch
from torch import Tensor
from torch.types import Device

__all__ = [
    'CattedSequence',
    'accumulate_sizes', 'transpose_sizes',
    'major_sizes_to_size',
    'major_sizes_to_ptr',
    'minor_sizes_to_ptr',
    'sizes_to_sorting',
    'invert_permutation',
]


class CattedSequence(NamedTuple):
    data: Tensor
    token_sizes: Tensor

    def to(self, dtype: torch.dtype = None, device: Device = None, *args, **kwargs) -> 'CattedSequence':
        return CattedSequence(
            data=self.data.to(device=device, dtype=dtype, *args, **kwargs),
            token_sizes=self.token_sizes.to(device=device, dtype=dtype, *args, **kwargs),
        )


@torch.no_grad()
def accumulate_sizes(sizes: Tensor) -> Tensor:
    acc_sizes = sizes.cumsum(dim=0).roll(shifts=1, dims=0)
    acc_sizes[0] = 0
    return acc_sizes


@torch.no_grad()
def transpose_sizes(sizes: Tensor) -> Tensor:
    n, _ = major_sizes_to_size(sizes=sizes)
    index = torch.arange(n, device=sizes.device)
    return (index[:, None] < sizes[None, :]).long().sum(dim=-1)


@torch.no_grad()
def major_sizes_to_size(sizes: Tensor) -> Tuple[int, int]:
    return sizes.max().item(), sizes.size()[0]


@torch.no_grad()
def major_sizes_to_ptr(sizes: Tensor) -> Tuple[Tensor, Tensor]:
    minor_ptr = torch.repeat_interleave(repeats=sizes)

    major_ptr = torch.repeat_interleave(accumulate_sizes(sizes), repeats=sizes)
    major_ptr = torch.arange(major_ptr.size()[0], device=major_ptr.device) - major_ptr

    return major_ptr, minor_ptr


@torch.no_grad()
def minor_sizes_to_ptr(sizes: Tensor,
                       minor_ptr: Optional[Tensor] = None,
                       major_ptr: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    t, b = major_sizes_to_size(sizes=sizes)

    if minor_ptr is None:
        minor_ptr = torch.arange(t, device=sizes.device)
    if major_ptr is None:
        major_ptr = torch.arange(b, device=sizes.device)

    assert minor_ptr.size() == (t,), f'{minor_ptr.size()} != ({t},)'
    assert major_ptr.size() == (b,), f'{major_ptr.size()} != ({b},)'

    mask = minor_ptr[:, None] < sizes[None, :]

    minor_ptr = torch.masked_select(minor_ptr[:, None], mask=mask)
    major_ptr = torch.masked_select(major_ptr[None, :], mask=mask)
    major_sizes = mask.long().sum(dim=1)

    return minor_ptr, major_ptr, major_sizes


@torch.no_grad()
def sizes_to_sorting(sizes: Tensor, device: Device = None) -> Tuple[Tensor, Tensor, Tensor]:
    if device is None:
        device = sizes.device

    sizes, sorted_indices = sizes.cpu().sort(dim=0, descending=True)
    sizes = sizes.to(device=device)
    sorted_indices = sorted_indices.to(device=device)
    unsorted_indices = invert_permutation(sorted_indices)

    return sizes, sorted_indices, unsorted_indices


@torch.no_grad()
def invert_permutation(tensor: Tensor, device: Device = None, dtype: torch.dtype = None) -> Tensor:
    if dtype is None:
        dtype = tensor.dtype
    if device is None:
        device = tensor.device

    index = torch.arange(tensor.size()[0], dtype=dtype, device=device)
    permutation = torch.empty_like(index)
    permutation[tensor] = index
    return permutation
