from typing import List

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.catting import cat_sequence
from torchrua.core import CattedSequence
from torchrua.core import accumulate_sizes
from torchrua.core import minor_sizes_to_ptr
from torchrua.core import sizes_to_sorting

__all__ = [
    'pack_sequence',
    'pack_catted_indices', 'pack_catted_sequence',
    'pack_padded_indices', 'pack_padded_sequence',
]


def pack_sequence(sequences: List[Tensor], device: Device = None) -> PackedSequence:
    if device is None:
        device = sequences[0].device

    sequence = cat_sequence(sequences=sequences, device=device)
    return pack_catted_sequence(sequence=sequence, device=device)


@torch.no_grad()
def pack_catted_indices(token_sizes: Tensor, device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)

    token_sizes, sorted_indices, unsorted_indices = sizes_to_sorting(sizes=token_sizes, device=device)
    token_ptr, batch_ptr, batch_sizes = minor_sizes_to_ptr(sizes=token_sizes, major_ptr=sorted_indices)

    return acc_token_sizes[batch_ptr] + token_ptr, batch_sizes, sorted_indices, unsorted_indices


def pack_catted_sequence(sequence: CattedSequence, device: Device = None) -> PackedSequence:
    if device is None:
        device = sequence.data.device

    indices, batch_sizes, sorted_indices, unsorted_indices = pack_catted_indices(
        token_sizes=sequence.token_sizes, device=device,
    )

    return PackedSequence(
        data=sequence.data[indices],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


@torch.no_grad()
def pack_padded_indices(token_sizes: Tensor, batch_first: bool, device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)

    sorted_token_sizes, sorted_indices, unsorted_indices = sizes_to_sorting(sizes=token_sizes, device=device)
    token_ptr, batch_ptr, batch_sizes = minor_sizes_to_ptr(sizes=sorted_token_sizes, major_ptr=sorted_indices)

    if batch_first:
        return (batch_ptr, token_ptr), batch_sizes, sorted_indices, unsorted_indices
    else:
        return (token_ptr, batch_ptr), batch_sizes, sorted_indices, unsorted_indices


def pack_padded_sequence(sequence: Tensor, token_sizes: Tensor,
                         batch_first: bool, device: Device = None) -> PackedSequence:
    if device is None:
        device = sequence.device

    indices, batch_sizes, sorted_indices, unsorted_indices = pack_padded_indices(
        token_sizes=token_sizes, batch_first=batch_first, device=device,
    )

    return PackedSequence(
        data=sequence[indices],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )
