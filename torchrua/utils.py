from typing import List

import torch


def with_shape(shape: torch.Size, dim: int, value: int) -> List[int]:
    shape = list(shape)
    shape[dim] = value
    return shape


def broadcast_shapes(*sizes: torch.Size, dim: int):
    shape = torch.broadcast_shapes(*[with_shape(size, dim=dim, value=1) for size in sizes])
    return [with_shape(shape, dim=dim, value=size[dim]) for size in sizes]
