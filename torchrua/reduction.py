from functools import wraps
from typing import Union, List, NamedTuple, Callable, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.types import Device

from torchrua.catting import cat_packed_indices, CattedSequence, cat_sequence
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
    tgt: Tensor
    num_steps: int


@torch.no_grad()
def reduce_indices(token_sizes: Tensor, device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)
    token_sizes = (token_sizes << 1) - 1

    src, _ = major_sizes_to_ptr(sizes=token_sizes)

    acc_token_sizes = token_sizes.cumsum(dim=0)
    num_steps = acc_token_sizes[-1].item()
    tgt = acc_token_sizes - 1
    acc_token_sizes = F.pad(acc_token_sizes, [1, -1])

    offsets = acc_token_sizes.clone()
    mask = torch.ones_like(src, dtype=torch.bool)

    num_bits = torch.iinfo(token_sizes.dtype).bits
    sizes = 1 << torch.arange(num_bits - 1, device=device)
    acc_sizes = accumulate_sizes(sizes=sizes)
    opt_sizes = (token_sizes[None, :] - acc_sizes[:, None]).clamp_min_(0).min(sizes[:, None])
    opt_sizes = opt_sizes[1:opt_sizes.any(dim=1).long().sum()]

    xs, ys, zs = [], [], []
    for index in range(opt_sizes.size()[0] - 1, -1, -1):
        opt_size = opt_sizes[index]
        token_ptr, batch_ptr = major_sizes_to_ptr(sizes=opt_size >> 1)
        ptr = offsets[batch_ptr] + token_ptr

        x = ptr + token_ptr
        z = ptr + opt_size[batch_ptr]
        xs.append(x)
        ys.append(x + 1)
        zs.append(z)

        mask[z] = False
        offsets += opt_size

    return xs, ys, zs, src[mask], acc_token_sizes, tgt, num_steps


@torch.no_grad()
def reduce_catted_indices(token_sizes: Tensor) -> ReductionIndices:
    _, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)

    xs, ys, zs, token_ptr, acc_token_sizes, tgt, num_steps = reduce_indices(token_sizes=token_sizes)
    src = acc_token_sizes[batch_ptr] + token_ptr

    return ReductionIndices(
        xs=xs, ys=ys, zs=zs,
        src=(..., src), tgt=tgt, num_steps=num_steps,
    )


@torch.no_grad()
def reduce_packed_indices(batch_sizes: Tensor, unsorted_indices: Tensor = None) -> ReductionIndices:
    indices, token_sizes = cat_packed_indices(batch_sizes=batch_sizes, unsorted_indices=unsorted_indices)

    xs, ys, zs, (_, src), tgt, num_steps = reduce_catted_indices(token_sizes=token_sizes)
    src = src[invert_permutation(indices)]

    return ReductionIndices(
        xs=xs, ys=ys, zs=zs,
        src=(..., src), tgt=tgt, num_steps=num_steps,
    )


@torch.no_grad()
def reduce_padded_indices(token_sizes: Tensor, batch_first: bool = False) -> ReductionIndices:
    xs, ys, zs, (_, src2), tgt, num_steps = reduce_catted_indices(token_sizes=token_sizes)

    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)
    src1 = (batch_ptr, token_ptr) if batch_first else (token_ptr, batch_ptr)

    return ReductionIndices(
        xs=xs, ys=ys, zs=zs,
        src=(src1, src2), tgt=tgt, num_steps=num_steps,
    )


def reduce_sequence(fn: Callable[[Tensor, Tensor], Tensor]):
    @wraps(fn)
    def _reduce_sequence(tensor: Tensor, indices: ReductionIndices) -> Tensor:
        xs, ys, zs, (src1, src2), tgt, num_steps = indices

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

        return data[tgt]

    return _reduce_sequence


@torch.no_grad()
def token_sizes_to_reduction_sizes(token_sizes: Tensor, device: Device = None):
    if device is None:
        device = token_sizes.device

    num_bits = torch.iinfo(token_sizes.dtype).bits
    token_sizes = token_sizes.to(device=device) << 1

    sizes = 1 << torch.arange(num_bits - 1, device=device)[:, None]
    sizes = torch.clamp_min(token_sizes - sizes, 0).min(sizes)
    sizes = torch.flipud(sizes)

    return token_sizes, sizes[sizes.any(dim=-1)]


@torch.no_grad()
def token_sizes_to_reduction_ptr(token_sizes: Tensor, device: Device = None):
    token_sizes, sizes = token_sizes_to_reduction_sizes(token_sizes, device=device)

    n, *_ = token_sizes.size()

    sizes0 = sizes >> 1
    sizes1 = F.pad(sizes0, [0, 0, 1, -1])
    sizes2 = sizes - sizes1

    token_ptr, batch_ptr = major_sizes_to_ptr(sizes0.view(-1))
    tgt = sizes.view(-1).cumsum(dim=0)[batch_ptr + n - 1] + token_ptr

    token_ptr, batch_ptr = major_sizes_to_ptr(sizes2.view(-1))
    src = (accumulate_sizes(sizes.view(-1)) + sizes1.view(-1))[batch_ptr] + token_ptr
    token_ptr = F.pad(sizes2.cumsum(dim=0), [0, 0, 1, -1]).view(-1)[batch_ptr] + token_ptr

    return src, tgt, batch_ptr % n, token_ptr, sizes.sum(dim=1)


@torch.no_grad()
def reduce_catted_indices2(token_sizes: Tensor, device: Device = None):
    if device is None:
        device = token_sizes.device

    src1, tgt, batch_ptr, token_ptr, sizes = token_sizes_to_reduction_ptr(token_sizes, device=device)
    src2 = accumulate_sizes(token_sizes)[batch_ptr] + token_ptr
    num_steps = sizes.sum().detach().item()

    return num_steps, sizes[:-1], src1, src2, tgt


def reduce_catted_sequence2(sequence: CattedSequence, op: Callable[[Tensor, Tensor], Tensor] = torch.add) -> Tensor:
    data, token_sizes = sequence

    n, = token_sizes.size()
    num_steps, sizes, src1, src2, tgt = reduce_catted_indices2(token_sizes=token_sizes, device=data.device)

    tensor = torch.empty(
        (num_steps, *data.size()[1:]),
        device=data.device, dtype=data.dtype, requires_grad=False,
    )
    tensor[src1] = data[src2]

    x1, y1 = 0, 0
    x2, y2 = 0, 0
    for size in sizes.detach().tolist():
        x1, y1 = y1, y1 + (size >> 1)
        x2, y2 = y2, y2 + (size >> 0)
        tensor[tgt[x1:y1]] = op(tensor[x2 + 0:y2:2], tensor[x2 + 1:y2:2])

    return tensor[-n:]


if __name__ == '__main__':
    s = cat_sequence([
        torch.arange(5, dtype=torch.float32),
        torch.arange(2, dtype=torch.float32),
        torch.arange(3, dtype=torch.float32),
    ])
    print(reduce_catted_sequence2(s))
