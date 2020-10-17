from typing import Union

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence


@torch.no_grad()
def packed_sequence_to_mask(pack: PackedSequence, unsort: bool, total_length: int = None,
                            dtype: torch.dtype = torch.bool, device: torch.device = None) -> Tensor:
    mask = batch_sizes_to_mask(
        batch_sizes=pack.batch_sizes, total_length=total_length,
        dtype=dtype, device=device or pack.data.device,
    )
    if unsort and pack.unsorted_indices is not None:
        mask = mask[pack.unsorted_indices]
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


@torch.no_grad()
def batch_sizes_to_mask(batch_sizes: Tensor, total_length: int = None,
                        dtype: torch.dtype = torch.bool, device: torch.device = None) -> Tensor:
    if total_length is not None:
        if total_length > batch_sizes.size(0):
            batch_sizes = F.pad(batch_sizes, [0, total_length - batch_sizes.size(0)], value=0)
        elif total_length < batch_sizes.size(0):
            batch_sizes = batch_sizes[:total_length]

    batch_size = batch_sizes[0].item()
    return torch.ones(
        (batch_size + 1, batch_size + 1),
        dtype=dtype, device=device or batch_sizes.device,
    ).triu(0)[1:, batch_sizes]


@torch.no_grad()
def batch_sizes_to_lengths(batch_sizes: Tensor, dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    mask = batch_sizes_to_mask(batch_sizes, dtype=torch.bool, device=device)
    return mask_to_lengths(mask, dtype=dtype, device=device)


@torch.no_grad()
def mask_to_lengths(mask: Tensor, dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    return mask.to(dtype=dtype, device=device or mask.device).sum(dim=1)


@torch.no_grad()
def mask_to_batch_sizes(mask: Tensor, dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    return mask.to(dtype=dtype, device=device or mask.device).sum(dim=0)


@torch.no_grad()
def lengths_to_mask(lengths: Tensor, total_length: int = None,
                    dtype: torch.dtype = torch.bool, device: torch.device = None) -> Tensor:
    if total_length is None:
        total_length = lengths.max().item()

    return torch.ones((total_length, total_length), dtype=dtype, device=device or lengths.device).tril(0)[lengths - 1]


@torch.no_grad()
def lengths_to_batch_sizes(lengths: Tensor, dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    mask = lengths_to_mask(lengths=lengths, dtype=torch.bool, device=device)
    return mask_to_batch_sizes(mask=mask, dtype=dtype, device=device)


@torch.no_grad()
def fetch_batch_size(pack: PackedSequence) -> int:
    return pack.batch_sizes[0].item()


@torch.no_grad()
def fetch_total_length(pack: PackedSequence) -> int:
    return pack.batch_sizes.size(0)


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
def fetch_batch_sizes(x: Union[Tensor,PackedSequence], total_length: int = None) -> Tensor:
    batch_sizes = x
    if not torch.is_tensor(x):
        batch_sizes = x.batch_sizes

    if total_length is not None:
        if total_length < batch_sizes.size(0):
            assert batch_sizes[0].item() == batch_sizes[-total_length].item(), \
                f'some sequences contain only less than {total_length} elements, truncating is not allowed.'
            batch_sizes = batch_sizes[-total_length:]
        elif total_length > batch_sizes.size(0):
            batch_sizes = torch.cat([
                torch.full(
                    (total_length - batch_sizes.size(0),), fill_value=batch_sizes[0],
                    dtype=batch_sizes.dtype, device=batch_sizes.device,
                ), batch_sizes,
            ], dim=0)
    return batch_sizes
