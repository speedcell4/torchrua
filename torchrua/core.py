from typing import Tuple

import torch
from torch import Tensor


@torch.no_grad()
def sizes_to_ptr(sizes: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor]:
    major_ptr = torch.repeat_interleave(sizes)
    index = torch.arange(major_ptr.size()[0], device=major_ptr.device)
    acc_sizes = sizes.cumsum(dim=0)

    if not reverse:
        acc_sizes[-1] = 0
        minor_ptr = index - acc_sizes[major_ptr - 1]
    else:
        minor_ptr = acc_sizes[major_ptr] - index - 1
    return major_ptr, minor_ptr


@torch.no_grad()
def transpose_sizes(sizes: Tensor) -> Tensor:
    sorted_sizes, _ = torch.sort(sizes, descending=True)

    rolled_sizes = torch.roll(sorted_sizes, -1, dims=0)
    rolled_sizes[-1] = 0

    return torch.repeat_interleave(sorted_sizes - rolled_sizes).add(1).flip(dims=[0])
