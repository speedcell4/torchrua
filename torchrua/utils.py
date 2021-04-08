from typing import Union, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence, invert_permutation

__all__ = [
    'get_device',
    'resize_batch_sizes', 'accumulate_batch_sizes',
    'batch_sizes_to_mask', 'lengths_to_mask', 'packed_sequence_to_mask',
    'batch_sizes_to_lengths', 'packed_sequence_to_lengths',
    'lengths_to_batch_sizes',
    'lengths_to_sorting_indices',
]


@torch.no_grad()
def get_device(seq: Union[Tensor, PackedSequence], device: torch.device = None) -> torch.device:
    if device is not None:
        return device
    if torch.is_tensor(seq):
        return seq.device
    return seq.data.device


@torch.no_grad()
def accumulate_batch_sizes(batch_sizes: Tensor, device: torch.device = None) -> Tensor:
    batch_sizes = batch_sizes.to(device=device).cumsum(dim=0).roll(1, dims=[0])
    batch_sizes[0] = 0

    return batch_sizes


def resize_batch_sizes(batch_sizes: Tensor, total_length: int) -> Tensor:
    num_tokens = batch_sizes.size(0)
    if total_length <= num_tokens:
        assert batch_sizes[0] == batch_sizes[-total_length]
        return batch_sizes[-total_length:]
    return F.pad(batch_sizes, [total_length - num_tokens, 0], value=batch_sizes[0])


@torch.no_grad()
def batch_sizes_to_mask(
        batch_sizes: Tensor, unsorted_indices: Tensor = None,
        batch_first: bool = True, total_length: int = None,
        dtype: torch.dtype = torch.bool, device: torch.device = None) -> Tensor:
    batch_size = batch_sizes[0].item()

    batch_ptr = torch.arange(batch_size, dtype=batch_sizes.dtype, device=batch_sizes.device)
    if total_length is not None:
        batch_sizes = resize_batch_sizes(batch_sizes, total_length=total_length)

    if unsorted_indices is None:
        unsorted_indices = ...

    if batch_first:
        mask = batch_ptr[unsorted_indices, None] < batch_sizes[None, :]
    else:
        mask = batch_ptr[None, unsorted_indices] < batch_sizes[:, None]

    device = get_device(batch_sizes, device=device)

    return mask.to(dtype=dtype, device=device)


@torch.no_grad()
def batch_sizes_to_lengths(batch_sizes: Tensor, unsorted_indices: Tensor = None,
                           dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    return batch_sizes_to_mask(
        batch_sizes, unsorted_indices=unsorted_indices,
        batch_first=True, dtype=dtype, device=device,
    ).sum(dim=1)


@torch.no_grad()
def lengths_to_mask(lengths: Tensor, batch_first: bool = True, total_length: int = None,
                    dtype: torch.dtype = torch.bool, device: torch.device = None) -> Tensor:
    if total_length is None:
        total_length = lengths.max().item()

    token_ptr = torch.arange(total_length, dtype=lengths.dtype, device=lengths.device)

    if batch_first:
        mask = token_ptr[None, :] < lengths[:, None]
    else:
        mask = token_ptr[:, None] < lengths[None, :]

    device = get_device(lengths, device=device)

    return mask.to(dtype=dtype, device=device)


@torch.no_grad()
def lengths_to_batch_sizes(lengths: Tensor, total_length: int = None,
                           dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    return lengths_to_mask(
        lengths, batch_first=False, total_length=total_length, dtype=dtype, device=device,
    ).sum(dim=1)


@torch.no_grad()
def lengths_to_sorting_indices(lengths: Tensor, dtype: torch.dtype = torch.long,
                               device: torch.device = None) -> Tuple[Tensor, Tensor]:
    sorted_indices = lengths.argsort(dim=0, descending=True)
    unsorted_indices = invert_permutation(sorted_indices)

    device = get_device(lengths, device=device)
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
def packed_sequence_to_lengths(pack: PackedSequence, unsort: bool,
                               dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    return batch_sizes_to_lengths(
        batch_sizes=pack.batch_sizes,
        unsorted_indices=pack.unsorted_indices if unsort else None,
        dtype=dtype, device=device,
    )
