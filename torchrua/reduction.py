from typing import Union, NamedTuple, Callable, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.catting import CattedSequence
from torchrua.core import major_sizes_to_ptr, accumulate_sizes, invert_permutation, transpose_sizes
from torchrua.padding import PaddedSequence

__all__ = [
    'ReductionIndices', 'reduce_sequence',
    'token_sizes_to_reduction_sizes', 'token_sizes_to_reduction_ptr',
    'reduce_packed_indices', 'reduce_packed_sequence',
    'reduce_padded_indices', 'reduce_padded_sequence',
    'reduce_catted_indices', 'reduce_catted_sequence',
]


class ReductionIndices(NamedTuple):
    batch_size: int
    cache_size: int
    sizes: Tensor
    src: Union[Tensor, Tuple[Tensor, Tensor, Tensor]]
    tgt: Tensor


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
    if device is None:
        device = token_sizes.device

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
def reduce_catted_indices(token_sizes: Tensor, device: Device = None):
    if device is None:
        device = token_sizes.device

    src, tgt, batch_ptr, token_ptr, sizes = token_sizes_to_reduction_ptr(token_sizes, device=device)
    index = accumulate_sizes(token_sizes)[batch_ptr] + token_ptr
    cache_size = sizes.sum().detach().item()
    batch_size, = token_sizes.size()

    return ReductionIndices(
        batch_size=batch_size, cache_size=cache_size,
        sizes=sizes[:-1], src=src[invert_permutation(index)], tgt=tgt,
    )


@torch.no_grad()
def reduce_packed_indices(batch_sizes: Tensor, unsorted_indices: Tensor = None, device: Device = None):
    if device is None:
        if unsorted_indices is not None:
            device = unsorted_indices.device
        else:
            device = batch_sizes.device

    batch_sizes = batch_sizes.to(device=device)
    token_sizes = transpose_sizes(sizes=batch_sizes)
    if unsorted_indices is not None:
        token_sizes = token_sizes[unsorted_indices]

    src, tgt, batch_ptr, token_ptr, sizes = token_sizes_to_reduction_ptr(token_sizes, device=device)

    if unsorted_indices is not None:
        batch_ptr = unsorted_indices[batch_ptr]
    index = accumulate_sizes(batch_sizes)[token_ptr] + batch_ptr
    cache_size = sizes.sum().detach().item()
    batch_size = batch_sizes[0].item()

    return ReductionIndices(
        batch_size=batch_size, cache_size=cache_size,
        sizes=sizes[:-1], src=src[invert_permutation(index)], tgt=tgt,
    )


@torch.no_grad()
def reduce_padded_indices(token_sizes: Tensor, batch_first: bool, device: Device = None):
    if device is None:
        device = token_sizes.device

    src, tgt, batch_ptr, token_ptr, sizes = token_sizes_to_reduction_ptr(token_sizes, device=device)
    cache_size = sizes.sum().detach().item()
    batch_size, = token_sizes.size()

    if batch_first:
        return ReductionIndices(
            batch_size=batch_size, cache_size=cache_size,
            sizes=sizes[:-1], src=(src, batch_ptr, token_ptr), tgt=tgt,
        )
    else:
        return ReductionIndices(
            batch_size=batch_size, cache_size=cache_size,
            sizes=sizes[:-1], src=(src, token_ptr, batch_ptr), tgt=tgt,
        )


def reduce_sequence(data: Tensor, indices: ReductionIndices, op: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    batch_size, cache_size, sizes, src, tgt = indices

    if torch.is_tensor(src):
        tensor = torch.empty(
            (cache_size, *data.size()[1:]),
            device=data.device, dtype=data.dtype, requires_grad=False,
        )
        tensor[src] = data
    else:
        src, ptr1, ptr2 = src
        tensor = torch.empty(
            (cache_size, *data.size()[2:]),
            device=data.device, dtype=data.dtype, requires_grad=False,
        )
        tensor[src] = data[ptr1, ptr2]

    x1, y1 = 0, 0
    x2, y2 = 0, 0
    for size in sizes.detach().tolist():
        x1, y1 = y1, y1 + (size >> 1)
        x2, y2 = y2, y2 + (size >> 0)
        tensor[tgt[x1:y1]] = op(tensor[x2 + 0:y2:2], tensor[x2 + 1:y2:2])

    return tensor[-batch_size:]


def reduce_catted_sequence(op: Callable[[Tensor, Tensor], Tensor]):
    def wrap(sequence: CattedSequence, indices: ReductionIndices = None) -> Tensor:
        data, token_sizes = sequence

        if indices is None:
            indices = reduce_catted_indices(token_sizes=token_sizes, device=data.device)

        return reduce_sequence(data=data, indices=indices, op=op)

    return wrap


def reduce_packed_sequence(op: Callable[[Tensor, Tensor], Tensor]):
    def wrap(sequence: PackedSequence, indices: ReductionIndices = None) -> Tensor:
        data, batch_sizes, sorted_indices, unsorted_indices = sequence

        if indices is None:
            indices = reduce_packed_indices(
                batch_sizes=batch_sizes,
                unsorted_indices=unsorted_indices, device=data.device,
            )

        return reduce_sequence(data=data, indices=indices, op=op)

    return wrap


def reduce_padded_sequence(op: Callable[[Tensor, Tensor], Tensor]):
    def wrap(sequence: PaddedSequence, batch_first: bool = True, indices: ReductionIndices = None) -> Tensor:
        data, token_sizes = sequence

        if indices is None:
            indices = reduce_padded_indices(
                token_sizes=token_sizes,
                batch_first=batch_first, device=data.device,
            )

        return reduce_sequence(data=data, indices=indices, op=op)

    return wrap
