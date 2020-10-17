from typing import Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import invert_permutation


@torch.no_grad()
def pack_to_mask(pack: PackedSequence, unsort: bool = True, filling_mask: bool = True, batch_first: bool = False, *,
                 dtype: torch.dtype = torch.bool, device: torch.device = None) -> Tensor:
    batch_size = pack.batch_sizes[0].item()
    if device is None:
        device = pack.data.device

    mask = torch.ones((batch_size, batch_size), dtype=torch.bool, device=device).triu(0)
    if unsort and pack.unsorted_indices is not None:
        mask = mask[pack.unsorted_indices]
    mask = mask[:, pack.batch_sizes - 1]

    if not filling_mask:
        mask = ~mask
    if not batch_first:
        mask = mask.transpose(0, 1)
    return mask.to(dtype=dtype)


@torch.no_grad()
def pack_to_lengths(pack: PackedSequence, unsort: bool = True, *,
                    dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    batch_size = pack.batch_sizes[0].item()
    if device is None:
        device = pack.data.device

    mask = torch.ones((batch_size, batch_size), dtype=torch.long, device=device).tril(0)
    lengths = mask[pack.batch_sizes - 1].sum(dim=0)

    if unsort and pack.unsorted_indices is not None:
        lengths = lengths[pack.unsorted_indices]
    return lengths.to(dtype=dtype)


@torch.no_grad()
def lengths_to_mask(lengths: Tensor, filling_mask: bool = True,
                    batch_first: bool = False, total_length: int = None, *,
                    dtype: torch.dtype = torch.bool, device: torch.device = None) -> Tensor:
    if device is None:
        device = lengths.device
    if total_length is None:
        total_length = lengths.max().item()

    token_indices = torch.arange(total_length, dtype=lengths.dtype, device=lengths.device)
    mask = token_indices[None, :] < lengths[:, None]

    if not filling_mask:
        mask = ~mask
    if not batch_first:
        mask = mask.transpose(0, 1)
    return mask.to(dtype=dtype, device=device)


@torch.no_grad()
def lengths_to_batch_sizes(lengths: Tensor, *, device: torch.device = None) -> Tuple[Tensor, Tensor, Tensor]:
    total_length = lengths.max().item()
    if device is None:
        device = lengths.device

    mask = torch.ones((total_length, total_length), dtype=torch.long, device=device).tril(0)

    batch_sizes = mask[lengths - 1].sum(dim=0)
    sorted_indices = lengths.argsort(dim=0, descending=True)
    unsorted_indices = invert_permutation(sorted_indices)

    return batch_sizes.to(device=torch.device('cpu')), sorted_indices, unsorted_indices
