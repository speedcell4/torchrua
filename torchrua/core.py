from typing import Tuple

import torch
from torch import Tensor


def major_sizes_to_ptr(sizes: Tensor) -> Tuple[Tensor, Tensor]:
    minor_ptr = torch.repeat_interleave(repeats=sizes)

    major_ptr = torch.repeat_interleave(get_offsets(sizes=sizes), repeats=sizes)
    major_ptr = torch.arange(major_ptr.size()[0], device=major_ptr.device) - major_ptr

    return major_ptr, minor_ptr


def get_offsets(sizes: Tensor) -> Tensor:
    sizes = sizes.cumsum(dim=0).roll(shifts=1, dims=0)
    sizes[0] = 0
    return sizes


def invert_permutation(tensor: Tensor) -> Tensor:
    index = torch.arange(tensor.size()[0], device=tensor.device)
    permutation = torch.empty_like(index)
    permutation[tensor] = index
    return permutation


def _self(self, *_, **__):
    return self
