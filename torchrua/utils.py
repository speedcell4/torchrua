from typing import Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, invert_permutation

__all__ = [
    'fetch_dtype', 'fetch_device',
    'fetch_batch_size', 'fetch_total_length',
    'fetch_batch_sizes', 'fetch_accumulated_batch_sizes',
    'batch_sizes_to_mask', 'batch_sizes_to_lengths',
    'lengths_to_mask', 'lengths_to_batch_sizes', 'lengths_to_sorted_indices',
    'packed_sequence_to_mask', 'packed_sequence_to_lengths',
]


@torch.no_grad()
def fetch_dtype(x: Union[Tensor, PackedSequence], dtype: torch.dtype = None) -> torch.dtype:
    if dtype is not None:
        return dtype
    if torch.is_tensor(x):
        return x.dtype
    if isinstance(x, PackedSequence):
        return x.data.dtype
    raise TypeError(f'unsupported type {type(x)}')


@torch.no_grad()
def fetch_device(x: Union[Tensor, PackedSequence], device: torch.device = None) -> torch.device:
    if device is not None:
        return device
    if torch.is_tensor(x):
        return x.device
    if isinstance(x, PackedSequence):
        return x.data.device
    raise TypeError(f'unsupported type {type(x)}')


@torch.no_grad()
def fetch_batch_size(x: Union[Tensor, PackedSequence]) -> int:
    batch_sizes = x
    if not torch.is_tensor(x):
        batch_sizes = x.batch_sizes
    return batch_sizes[0].item()


@torch.no_grad()
def fetch_total_length(x: Union[Tensor, PackedSequence], total_length: int = None) -> int:
    batch_sizes = x
    if not torch.is_tensor(x):
        batch_sizes = x.batch_sizes
    if total_length is not None:
        return total_length
    return batch_sizes.size(0)


@torch.no_grad()
def fetch_batch_sizes(x: Union[Tensor, PackedSequence],
                      total_length: int = None, device: torch.device = None) -> Tensor:
    device = fetch_device(x, device=device)

    batch_sizes = x
    if not torch.is_tensor(x):
        batch_sizes = x.batch_sizes

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
def fetch_accumulated_batch_sizes(x: Union[Tensor, PackedSequence], device: torch.device = None) -> Tensor:
    batch_sizes = x
    if not torch.is_tensor(x):
        batch_sizes = x.batch_sizes
    batch_sizes: Tensor = batch_sizes.cumsum(dim=0).roll(1, dims=[0])
    batch_sizes[0] = 0
    return batch_sizes.to(device=device)


@torch.no_grad()
def batch_sizes_to_mask(batch_sizes: Tensor, batch_first: bool = True, total_length: int = None,
                        dtype: torch.dtype = torch.bool, device: torch.device = None) -> Tensor:
    device = fetch_device(batch_sizes, device=device)
    dtype = fetch_dtype(batch_sizes, dtype=dtype)
    batch_size = fetch_batch_size(batch_sizes)

    batch_sizes = fetch_batch_sizes(batch_sizes, total_length=total_length, device=device)
    indices = torch.ones(
        (batch_size, batch_size),
        dtype=dtype, device=device,
    )

    if batch_first:
        return indices.triu(0)[:, batch_sizes - 1]
    else:
        return indices.tril(0)[batch_sizes - 1, :]


@torch.no_grad()
def batch_sizes_to_lengths(batch_sizes: Tensor, batch_first: bool = True,
                           dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    return batch_sizes_to_mask(batch_sizes, batch_first=batch_first, dtype=dtype, device=device).sum(dim=1)


@torch.no_grad()
def lengths_to_mask(lengths: Tensor, batch_first: bool = True, total_length: int = None,
                    dtype: torch.dtype = torch.bool, device: torch.device = None) -> Tensor:
    device = fetch_device(lengths, device=device)
    dtype = fetch_dtype(lengths, dtype=dtype)

    if total_length is None:
        total_length = lengths.max().item()

    indices = torch.ones(
        (total_length, total_length),
        dtype=dtype, device=device,
    )
    if batch_first:
        return indices.tril(0)[lengths - 1, :]
    else:
        return indices.triu(0)[:, lengths - 1]


@torch.no_grad()
def lengths_to_batch_sizes(lengths: Tensor, dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    return lengths_to_mask(lengths, batch_first=True, dtype=dtype, device=device).sum(dim=0)


@torch.no_grad()
def lengths_to_sorted_indices(lengths: Tensor, dtype: torch.dtype = torch.long, device: torch.device = None):
    device = fetch_device(lengths, device=device)
    dtype = fetch_dtype(lengths, dtype=dtype)

    sorted_indices = lengths.argsort(dim=0, descending=True)
    unsorted_indices = invert_permutation(sorted_indices)
    return sorted_indices.to(dtype=dtype, device=device), unsorted_indices.to(dtype=dtype, device=device)


@torch.no_grad()
def packed_sequence_to_mask(pack: PackedSequence, unsort: bool, batch_first: bool = True, total_length: int = None,
                            dtype: torch.dtype = torch.bool, device: torch.device = None) -> Tensor:
    mask = batch_sizes_to_mask(
        batch_sizes=pack.batch_sizes, batch_first=batch_first, total_length=total_length,
        dtype=dtype, device=device or pack.data.device,
    )
    if unsort and pack.unsorted_indices is not None:
        if batch_first:
            mask = mask[pack.unsorted_indices, :]
        else:
            mask = mask[:, pack.unsorted_indices]
    return mask


@torch.no_grad()
def packed_sequence_to_lengths(pack: PackedSequence, unsort: bool,
                               dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    lengths = batch_sizes_to_lengths(
        batch_sizes=pack.batch_sizes, dtype=dtype,
        device=device or pack.data.device,
    )
    if unsort and pack.unsorted_indices is not None:
        lengths = lengths[pack.unsorted_indices]
    return lengths
