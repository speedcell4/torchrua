from typing import Any, List, Tuple

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


def to_self(self: Any, *_, **__) -> Any:
    return self


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
