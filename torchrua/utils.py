from typing import Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence, invert_permutation

__all__ = [
    'accumulate_sizes', 'resize_sizes',
    'batch_sizes_to_mask', 'token_sizes_to_mask', 'packed_sequence_to_mask',
    'batch_sizes_to_token_sizes', 'packed_sequence_to_token_sizes',
    'token_sizes_to_batch_sizes',
    'lengths_to_sorting_indices',
]


@torch.no_grad()
def accumulate_sizes(sizes: Tensor) -> Tensor:
    return F.pad(sizes.cumsum(dim=0), pad=[1, -1])


@torch.no_grad()
def resize_sizes(sizes: Tensor, n: int) -> Tensor:
    if n <= sizes.size()[0]:
        assert sizes[0] == sizes[-n]
        return sizes[-n:]
    return F.pad(sizes, [n - sizes.size()[0], 0], value=sizes[0])


@torch.no_grad()
def batch_sizes_to_mask(
        batch_sizes: Tensor, unsorted_indices: Tensor = None,
        batch_first: bool = True, total_length: int = None,
        dtype: torch.dtype = torch.bool, device: torch.device = None) -> Tensor:
    batch_size = batch_sizes[0].item()

    batch_ptr = torch.arange(batch_size, dtype=batch_sizes.dtype, device=batch_sizes.device)
    if total_length is not None:
        batch_sizes = resize_sizes(batch_sizes, n=total_length)

    if unsorted_indices is None:
        unsorted_indices = ...

    if batch_first:
        mask = batch_ptr[unsorted_indices, None] < batch_sizes[None, :]
    else:
        mask = batch_ptr[None, unsorted_indices] < batch_sizes[:, None]

    if device is None:
        device = batch_sizes.device

    return mask.to(dtype=dtype, device=device)


@torch.no_grad()
def batch_sizes_to_token_sizes(batch_sizes: Tensor, unsorted_indices: Tensor = None,
                               dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    return batch_sizes_to_mask(
        batch_sizes, unsorted_indices=unsorted_indices,
        batch_first=True, dtype=dtype, device=device,
    ).sum(dim=1)


@torch.no_grad()
def token_sizes_to_mask(lengths: Tensor, batch_first: bool = True, total_length: int = None,
                        dtype: torch.dtype = torch.bool, device: torch.device = None) -> Tensor:
    if total_length is None:
        total_length = lengths.max().item()

    token_ptr = torch.arange(total_length, dtype=lengths.dtype, device=lengths.device)

    if batch_first:
        mask = token_ptr[None, :] < lengths[:, None]
    else:
        mask = token_ptr[:, None] < lengths[None, :]

    if device is None:
        device = lengths.device

    return mask.to(dtype=dtype, device=device)


@torch.no_grad()
def token_sizes_to_batch_sizes(lengths: Tensor, total_length: int = None,
                               dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    return token_sizes_to_mask(
        lengths, batch_first=False, total_length=total_length, dtype=dtype, device=device,
    ).sum(dim=1)


@torch.no_grad()
def lengths_to_sorting_indices(lengths: Tensor, dtype: torch.dtype = torch.long,
                               device: torch.device = None) -> Tuple[Tensor, Tensor]:
    sorted_indices = lengths.argsort(dim=0, descending=True)
    unsorted_indices = invert_permutation(sorted_indices)

    if device is None:
        device = lengths.device
    return sorted_indices.to(dtype=dtype, device=device), unsorted_indices.to(dtype=dtype, device=device)


@torch.no_grad()
def packed_sequence_to_mask(pack: PackedSequence, unsort: bool,
                            batch_first: bool = True, total_length: int = None,
                            dtype: torch.dtype = torch.bool, device: torch.device = None) -> Tensor:
    return batch_sizes_to_mask(
        batch_sizes=pack.batch_sizes,
        unsorted_indices=pack.unsorted_indices if unsort else None,
        batch_first=batch_first, total_length=total_length,
        dtype=dtype, device=device,
    )


@torch.no_grad()
def packed_sequence_to_token_sizes(pack: PackedSequence, unsort: bool,
                                   dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    return batch_sizes_to_token_sizes(
        batch_sizes=pack.batch_sizes,
        unsorted_indices=pack.unsorted_indices if unsort else None,
        dtype=dtype, device=device,
    )
