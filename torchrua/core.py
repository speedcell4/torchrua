from typing import NamedTuple, Optional, Tuple

import torch
from torch import Tensor


class CattedSequence(NamedTuple):
    data: Tensor
    token_sizes: Tensor

    def to(self, dtype: torch.dtype = None, device: torch.device = None, **kwargs) -> 'CattedSequence':
        return CattedSequence(
            data=self.data.to(dtype=dtype, device=device, **kwargs),
            token_sizes=self.token_sizes.to(dtype=dtype, device=device, **kwargs),
        )


def get_dtype(*tensors: Optional[Tensor], dtype: torch.dtype = None) -> torch.dtype:
    if dtype is not None:
        return dtype

    for tensor in tensors:
        if tensor is not None:
            return tensor.dtype

    raise RuntimeError(f'tensors are all None')


def get_device(*tensors: Optional[Tensor], device: torch.device = None) -> torch.device:
    if device is not None:
        return device

    for tensor in tensors:
        if tensor is not None:
            return tensor.device

    raise RuntimeError(f'tensors are all None')


def major_sizes_to_shapes(sizes: Tensor) -> Tuple[int, int]:
    return sizes.max().item(), sizes.size()[0]


def arange_like(tensor: Tensor, dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    device = get_device(tensor, device=device)
    size, *_ = tensor.size()

    return torch.arange(size, dtype=dtype, device=device)


def repeat_interleave(tensor: Tensor = None, *, repeats: Tensor) -> Tensor:
    if tensor is None:
        tensor = arange_like(repeats, dtype=torch.long)

    return torch.repeat_interleave(tensor, repeats=repeats)


def major_sizes_to_ptr(sizes: Tensor) -> Tuple[Tensor, Tensor]:
    minor_ptr = repeat_interleave(repeats=sizes)

    major_ptr = repeat_interleave(accumulate_sizes(sizes=sizes), repeats=sizes)
    major_ptr = arange_like(major_ptr) - major_ptr

    return major_ptr, minor_ptr


@torch.no_grad()
def minor_sizes_to_ptr(sizes: Tensor, minor_ptr: Optional[Tensor] = None, major_ptr: Optional[Tensor] = None):
    t, b = major_sizes_to_shapes(sizes=sizes)

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


def major_masked_select(sizes: Tensor, device: torch.device = None):
    if device is None:
        device = sizes.device

    sizes = sizes.to(device=device)
    a, b = major_sizes_to_shapes(sizes=sizes)

    major_ptr = torch.arange(a, dtype=torch.long, device=device)
    minor_ptr = torch.arange(b, dtype=torch.long, device=device)

    mask = major_ptr[None, :] < sizes[:, None]
    major_ptr = torch.masked_select(major_ptr[None, :], mask=mask)
    minor_ptr = torch.masked_select(minor_ptr[:, None], mask=mask)
    sizes = mask.long().sum(dim=0)

    return major_ptr, minor_ptr, sizes


def minor_masked_select(sizes: Tensor, device: torch.device = None):
    if device is None:
        device = sizes.device

    sizes = sizes.to(device=device)
    a, b = major_sizes_to_shapes(sizes=sizes)

    major_ptr = torch.arange(a, dtype=torch.long, device=device)
    minor_ptr = torch.arange(b, dtype=torch.long, device=device)

    mask = major_ptr[:, None] < sizes[None, :]
    major_ptr = torch.masked_select(major_ptr[:, None], mask=mask)
    minor_ptr = torch.masked_select(minor_ptr[None, :], mask=mask)
    sizes = mask.long().sum(dim=1)

    return major_ptr, minor_ptr, sizes


def accumulate_sizes(sizes: Tensor) -> Tensor:
    sizes = sizes.cumsum(dim=0).roll(shifts=1, dims=0)
    sizes[0] = 0
    return sizes


@torch.no_grad()
def transpose_sizes(sizes: Tensor) -> Tensor:
    n, _ = major_sizes_to_shapes(sizes=sizes)
    index = torch.arange(n, device=sizes.device)
    return (index[:, None] < sizes[None, :]).long().sum(dim=-1)


@torch.no_grad()
def sizes_to_sorting(sizes: Tensor, device: torch.device = None) -> Tuple[Tensor, Tensor, Tensor]:
    if device is None:
        device = sizes.device

    sizes, sorted_indices = sizes.cpu().sort(dim=0, descending=True)
    sizes = sizes.to(device=device)
    sorted_indices = sorted_indices.to(device=device)
    unsorted_indices = invert_permutation(sorted_indices)

    return sizes, sorted_indices, unsorted_indices


def invert_permutation(tensor: Tensor) -> Tensor:
    index = arange_like(tensor)
    permutation = torch.empty_like(index)
    permutation[tensor] = index
    return permutation
