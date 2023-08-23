from typing import Optional
from typing import Tuple

import torch
from torch import Tensor


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


def broadcast_devices(*tensors: Optional[Tensor], device: torch.device = None):
    device = get_device(*tensors, device=device)
    tensors = (tensor if tensor is None else tensor.to(device=device) for tensor in tensors)
    return *tensors, device


def major_sizes_to_size(sizes: Tensor) -> Tuple[int, int]:
    return sizes.max().item(), sizes.size()[0]


def arange_like(tensor: Tensor, dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    device = get_device(tensor, device=device)
    size, *_ = tensor.size()

    return torch.arange(size, dtype=dtype, device=device)


def repeat_interleave(tensor: Tensor = None, *, repeats: Tensor) -> Tensor:
    if tensor is None:
        tensor = arange_like(repeats, dtype=torch.long)

    return torch.repeat_interleave(tensor, repeats=repeats)


def major_sizes_to_ptr(sizes: Tensor) -> Tuple[(Tensor), (Tensor)]:
    minor_ptr = repeat_interleave(repeats=sizes)

    major_ptr = repeat_interleave(accumulate_sizes(sizes=sizes), repeats=sizes)
    major_ptr = arange_like(major_ptr) - major_ptr

    return major_ptr, minor_ptr


def accumulate_sizes(sizes: Tensor) -> Tensor:
    sizes = sizes.cumsum(dim=0).roll(shifts=1, dims=0)
    sizes[0] = 0
    return sizes


def invert_permutation(tensor: Tensor) -> Tensor:
    index = arange_like(tensor)
    permutation = torch.empty_like(index)
    permutation[tensor] = index
    return permutation
