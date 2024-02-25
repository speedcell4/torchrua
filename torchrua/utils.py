from typing import List

import torch
from torch import Tensor


def with_shape(shape: torch.Size, dim: int, value: int) -> List[int]:
    shape = list(shape)
    shape[dim] = value
    return shape


def broadcast_shapes(*sizes: torch.Size, dim: int):
    shape = torch.broadcast_shapes(*[with_shape(size, dim=dim, value=1) for size in sizes])
    return [with_shape(shape, dim=dim, value=size[dim]) for size in sizes]


def broadcast_tensors(*tensors: Tensor, dim: int):
    shapes = broadcast_shapes(*[tensor.size() for tensor in tensors], dim=dim)
    return [tensor.expand(shape) for tensor, shape in zip(tensors, shapes)]


def gather(tensor: Tensor, index: Tensor, dim: int) -> Tensor:
    tensor, index = broadcast_tensors(tensor, index, dim=dim)
    return tensor.gather(dim=dim, index=index)
