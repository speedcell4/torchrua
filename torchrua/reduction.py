from functools import wraps
from typing import Union, List, NamedTuple, Callable, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.types import Device

from torchrua.catting import cat_packed_indices
from torchrua.core import major_sizes_to_ptr, accumulate_sizes, invert_permutation

__all__ = [
    'ReductionIndices',
    'reduce_packed_indices',
    'reduce_padded_indices',
    'reduce_catted_indices',
    'reduce_sequence',
]


class ReductionIndices(NamedTuple):
    xs: List[Tensor]
    ys: List[Tensor]
    zs: List[Tensor]
    src: Union[Tuple[Union[Tuple[Tensor, Tensor], Tensor], Tensor]]
    dst: Tensor
    num_steps: int


@torch.no_grad()
def reduce_indices(token_sizes: Tensor, device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)

    op_sizes = token_sizes * 2 - 1
    op_ptr, _ = major_sizes_to_ptr(sizes=op_sizes)

    acc_token_sizes2 = op_sizes.cumsum(dim=0)
    num_steps = acc_token_sizes2[-1].item()
    dst = acc_token_sizes2 - 1
    acc_token_sizes2 = F.pad(acc_token_sizes2, [1, -1])

    offsets = acc_token_sizes2.clone()
    mask = torch.ones_like(op_ptr, dtype=torch.bool)

    num_bits = torch.iinfo(token_sizes.dtype).bits
    sizes = 1 << torch.arange(num_bits, device=device)
    acc_sizes = accumulate_sizes(sizes=sizes)
    opt_sizes = (op_sizes[None, :] - acc_sizes[:, None]).clamp_min_(0).min(sizes[:, None])
    opt_sizes = opt_sizes[1:opt_sizes.any(dim=1).long().sum()]

    xs, ys, zs = [], [], []
    for index in range(opt_sizes.size()[0] - 1, -1, -1):
        opt_size = opt_sizes[index]
        token_ptr, batch_ptr = major_sizes_to_ptr(sizes=torch.div(opt_size, 2, rounding_mode='trunc'))
        ptr = offsets[batch_ptr] + token_ptr

        x = ptr + token_ptr
        z = ptr + opt_size[batch_ptr]
        xs.append(x)
        ys.append(x + 1)
        zs.append(z)

        mask[z] = False
        offsets += opt_size

    return xs, ys, zs, op_ptr[mask], acc_token_sizes2, dst, num_steps


@torch.no_grad()
def reduce_catted_indices(token_sizes: Tensor) -> ReductionIndices:
    _, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)

    xs, ys, zs, token_ptr, acc_token_sizes, dst, num_steps = reduce_indices(token_sizes=token_sizes)
    src = acc_token_sizes[batch_ptr] + token_ptr

    return ReductionIndices(
        xs=xs, ys=ys, zs=zs,
        src=(..., src), dst=dst, num_steps=num_steps,
    )


@torch.no_grad()
def reduce_packed_indices(batch_sizes: Tensor, unsorted_indices: Tensor = None) -> ReductionIndices:
    indices, token_sizes = cat_packed_indices(batch_sizes=batch_sizes, unsorted_indices=unsorted_indices)

    xs, ys, zs, (_, src), dst, num_steps = reduce_catted_indices(token_sizes=token_sizes)
    src = src[invert_permutation(indices)]

    return ReductionIndices(
        xs=xs, ys=ys, zs=zs,
        src=(..., src), dst=dst, num_steps=num_steps,
    )


@torch.no_grad()
def reduce_padded_indices(token_sizes: Tensor, batch_first: bool = False) -> ReductionIndices:
    xs, ys, zs, (_, src2), dst, num_steps = reduce_catted_indices(token_sizes=token_sizes)

    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)
    src1 = (batch_ptr, token_ptr) if batch_first else (token_ptr, batch_ptr)

    return ReductionIndices(
        xs=xs, ys=ys, zs=zs,
        src=(src1, src2), dst=dst, num_steps=num_steps,
    )


def reduce_sequence(fn: Callable[[Tensor, Tensor], Tensor]):
    @wraps(fn)
    def _reduce_sequence(tensor: Tensor, indices: ReductionIndices) -> Tensor:
        xs, ys, zs, (src1, src2), dst, num_steps = indices

        if isinstance(src1, tuple):
            _, _, *sizes = tensor.size()
        else:
            _, *sizes = tensor.size()
        data = torch.zeros(
            (num_steps, *sizes), requires_grad=False,
            dtype=tensor.dtype, device=tensor.device,
        )

        data[src2] = tensor[src1]
        for x, y, z in zip(xs, ys, zs):
            data[z] = fn(data[x], data[y])

        return data[dst]

    return _reduce_sequence
