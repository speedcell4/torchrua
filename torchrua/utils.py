from typing import Union, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, invert_permutation

__all__ = [
    'Seq',
    'get_dtype', 'get_device',
    'get_batch_size', 'get_total_length',
    'get_batch_sizes', 'accumulate_batch_sizes',
    'batch_sizes_to_mask', 'batch_sizes_to_lengths',
    'lengths_to_mask', 'lengths_to_batch_sizes', 'lengths_to_sorting_indices',
    'packed_sequence_to_mask', 'packed_sequence_to_lengths',
    'resize_batch_sizes',
]

Seq = Union[Tensor, PackedSequence]


@torch.no_grad()
def get_dtype(seq: Seq, dtype: torch.dtype = None) -> torch.dtype:
    if dtype is not None:
        return dtype
    if torch.is_tensor(seq):
        return seq.dtype
    return seq.data.dtype


@torch.no_grad()
def get_device(seq: Seq, device: torch.device = None) -> torch.device:
    if device is not None:
        return device
    if torch.is_tensor(seq):
        return seq.device
    return seq.data.device


@torch.no_grad()
def get_batch_size(seq: Seq) -> int:
    if torch.is_tensor(seq):
        return seq[0].item()
    return seq.batch_sizes[0].item()


@torch.no_grad()
def get_total_length(seq: Seq, total_length: int = None) -> int:
    if total_length is not None:
        return total_length
    if torch.is_tensor(seq):
        return seq.size(0)
    return seq.batch_sizes.size(0)


@torch.no_grad()
def get_batch_sizes(seq: Seq, total_length: int = None, device: torch.device = None) -> Tensor:
    device = get_device(seq, device=device)

    if torch.is_tensor(seq):
        batch_sizes = seq
    else:
        batch_sizes = seq.batch_sizes

    if total_length is not None:
        if total_length < batch_sizes.size(0):
            assert batch_sizes[0].item() == batch_sizes[-total_length].item(), \
                f'some sequences contain only less than {total_length} elements, truncating is not allowed.'
            batch_sizes = batch_sizes[-total_length:]
        elif total_length > batch_sizes.size(0):
            padding = torch.full(
                (total_length - batch_sizes.size(0),), fill_value=batch_sizes[0],
                dtype=batch_sizes.dtype, device=batch_sizes.device,
            )
            batch_sizes = torch.cat([padding, batch_sizes], dim=0)
    return batch_sizes.to(device=device)


@torch.no_grad()
def accumulate_batch_sizes(batch_sizes: Tensor, device: torch.device = None) -> Tensor:
    batch_sizes = batch_sizes.to(device=device).cumsum(dim=0).roll(1, dims=[0])
    batch_sizes[0] = 0

    return batch_sizes


@torch.no_grad()
def batch_sizes_to_mask(
        batch_sizes: Tensor, unsorted_indices: Tensor = None,
        batch_first: bool = True, total_length: int = None,
        dtype: torch.dtype = torch.bool, device: torch.device = None) -> Tensor:
    batch_size = get_batch_size(batch_sizes)

    batch_ptr = torch.arange(batch_size, dtype=batch_sizes.dtype, device=batch_sizes.device)
    batch_sizes = get_batch_sizes(batch_sizes, total_length=total_length, device=batch_sizes.device)

    if unsorted_indices is None:
        unsorted_indices = ...

    if batch_first:
        mask = batch_ptr[unsorted_indices, None] < batch_sizes[None, :]
    else:
        mask = batch_ptr[None, unsorted_indices] < batch_sizes[:, None]

    device = get_device(batch_sizes, device=device)
    dtype = get_dtype(batch_sizes, dtype=dtype)

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
    dtype = get_dtype(lengths, dtype=dtype)

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
    dtype = get_dtype(lengths, dtype=dtype)
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


def resize_batch_sizes(batch_sizes: Tensor, total_length: int) -> Tensor:
    num_tokens = batch_sizes.size(0)
    if total_length <= num_tokens:
        return batch_sizes[:total_length]
    return torch.cat([
        batch_sizes,
        torch.zeros(
            (total_length, num_tokens),
            dtype=batch_sizes.dtype, device=batch_sizes.device,
        )
    ])
