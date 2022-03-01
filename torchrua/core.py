from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.types import Device

__all__ = [
    'accumulate_sizes', 'transpose_sizes',
    'major_sizes_to_ptr',
    'minor_sizes_to_ptr',
    'sizes_to_sorting',
    'invert_permutation',
]


@torch.no_grad()
def accumulate_sizes(sizes: Tensor) -> Tensor:
    acc_sizes = sizes.cumsum(dim=0).roll(shifts=1, dims=0)
    acc_sizes[0] = 0
    return acc_sizes


@torch.no_grad()
def transpose_sizes(sizes: Tensor) -> Tensor:
    index = torch.arange(sizes.max().item(), device=sizes.device)
    return (index[:, None] < sizes[None, :]).long().sum(dim=-1)


@torch.no_grad()
def major_sizes_to_ptr(sizes: Tensor) -> Tuple[Tensor, Tensor]:
    minor_ptr = torch.repeat_interleave(repeats=sizes)

    major_ptr = torch.repeat_interleave(accumulate_sizes(sizes), repeats=sizes)
    major_ptr = torch.arange(major_ptr.size()[0], device=major_ptr.device) - major_ptr

    return major_ptr, minor_ptr


@torch.no_grad()
def minor_sizes_to_ptr(token_sizes: Tensor,
                       token_ptr: Optional[Tensor] = None,
                       batch_ptr: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    t = token_sizes.max().item()
    b = token_sizes.size()[0]

    if token_ptr is None:
        token_ptr = torch.arange(t, device=token_sizes.device)
    assert token_ptr.size() == (t,)

    if batch_ptr is None:
        batch_ptr = torch.arange(b, device=token_sizes.device)
    assert batch_ptr.size() == (b,)

    tb_mask = token_ptr[:, None] < token_sizes[None, :]

    token_ptr = torch.masked_select(token_ptr[:, None], mask=tb_mask)
    batch_ptr = torch.masked_select(batch_ptr[None, :], mask=tb_mask)
    sorted_batch_sizes = tb_mask.long().sum(dim=1)

    return token_ptr, batch_ptr, sorted_batch_sizes


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

    ret = torch.empty(tensor.size()[0], dtype=dtype, device=device)
    index = torch.arange(tensor.size()[0], dtype=dtype, device=device)
    ret[tensor] = index
    return ret
