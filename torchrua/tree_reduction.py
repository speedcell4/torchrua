from functools import wraps
from typing import Union, List, NamedTuple, Callable, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import invert_permutation

from torchrua.core import major_sizes_to_ptr, token_sizes_to_ptr
from torchrua.utils import accumulate_sizes

__all__ = [
    'TreeReduceIndices',
    'tree_reduce_packed_indices',
    'tree_reduce_padded_indices',
    'tree_reduce_catted_indices',
    'tree_reduce_sequence',
]


class TreeReduceIndices(NamedTuple):
    xs: List[Tensor]
    ys: List[Tensor]
    zs: List[Tensor]
    src: Tuple[Union[Tensor, Tuple[Tensor, Tensor]], Tensor]
    dst: Tensor
    num_steps: int


@torch.no_grad()
def tree_reduce_indices(token_sizes1: Tensor):
    token_sizes2 = token_sizes1 * 2 - 1
    token_ptr2, _ = major_sizes_to_ptr(sizes=token_sizes2)

    acc_token_sizes2 = token_sizes2.cumsum(dim=0)
    num_steps = acc_token_sizes2[-1].item()
    dst = acc_token_sizes2 - 1
    acc_token_sizes2 = F.pad(acc_token_sizes2, [1, -1])

    offsets = acc_token_sizes2.clone()
    mask = torch.ones_like(token_ptr2, dtype=torch.bool)

    sizes = 2 ** torch.arange(torch.iinfo(token_sizes1.dtype).bits - 1, device=token_sizes1.device)
    acc_sizes = accumulate_sizes(sizes=sizes)
    opt_sizes = (token_sizes2[:, None] - acc_sizes[None, :]).clamp_min(0).min(sizes)
    opt_sizes = opt_sizes[:, 1:opt_sizes.any(dim=0).long().sum()]

    xs, ys, zs = [], [], []
    for index in range(opt_sizes.size()[1] - 1, -1, -1):
        opt_size = opt_sizes[:, index]
        token_ptr, batch_ptr, _ = token_sizes_to_ptr(
            token_sizes=torch.div(opt_size, 2, rounding_mode='trunc'),
        )
        ptr = offsets[batch_ptr] + token_ptr

        x = ptr + token_ptr
        z = ptr + opt_size[batch_ptr]
        xs.append(x)
        ys.append(x + 1)
        zs.append(z)

        mask[z] = False
        offsets += opt_size

    return xs, ys, zs, token_ptr2[mask], acc_token_sizes2, dst, num_steps


@torch.no_grad()
def tree_reduce_packed_indices(batch_sizes: Tensor) -> TreeReduceIndices:
    batch_ptr1, token_ptr1, token_sizes1 = token_sizes_to_ptr(token_sizes=batch_sizes)
    acc_batch_sizes1 = accumulate_sizes(sizes=batch_sizes)

    xs, ys, zs, token_ptr2, acc_token_sizes2, dst, num_steps = tree_reduce_indices(token_sizes1=token_sizes1)
    src1 = acc_batch_sizes1[token_ptr1] + batch_ptr1
    src2 = acc_token_sizes2[batch_ptr1] + token_ptr2
    src2 = src2[invert_permutation(src1)]

    return TreeReduceIndices(
        xs=xs, ys=ys, zs=zs,
        src=(..., src2), dst=dst, num_steps=num_steps,
    )


@torch.no_grad()
def tree_reduce_padded_indices(token_sizes: Tensor, batch_first: bool = False) -> TreeReduceIndices:
    token_ptr1, batch_ptr1 = major_sizes_to_ptr(sizes=token_sizes)

    xs, ys, zs, token_ptr2, acc_token_sizes2, dst, num_steps = tree_reduce_indices(token_sizes1=token_sizes)
    src1 = (batch_ptr1, token_ptr1) if batch_first else (token_ptr1, batch_ptr1)
    src2 = acc_token_sizes2[batch_ptr1] + token_ptr2

    return TreeReduceIndices(
        xs=xs, ys=ys, zs=zs,
        src=(src1, src2), dst=dst, num_steps=num_steps,
    )


@torch.no_grad()
def tree_reduce_catted_indices(token_sizes: Tensor) -> TreeReduceIndices:
    token_ptr1, batch_ptr1 = major_sizes_to_ptr(sizes=token_sizes)
    acc_token_sizes1 = accumulate_sizes(sizes=token_sizes)

    xs, ys, zs, token_ptr2, acc_token_sizes2, dst, num_steps = tree_reduce_indices(token_sizes1=token_sizes)
    src1 = acc_token_sizes1[batch_ptr1] + token_ptr1
    src2 = acc_token_sizes2[batch_ptr1] + token_ptr2

    return TreeReduceIndices(
        xs=xs, ys=ys, zs=zs,
        src=(src1, src2), dst=dst, num_steps=num_steps,
    )


def tree_reduce_sequence(fn: Callable[[Tensor, Tensor], Tensor]):
    @wraps(fn)
    def _tree_reduce_sequence(tensor: Tensor, indices: TreeReduceIndices) -> Tensor:
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

    return _tree_reduce_sequence
