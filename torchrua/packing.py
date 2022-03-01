from typing import List

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.catting import cat_sequence
from torchrua.core import token_sizes_to_ptr, accumulate_sizes, sizes_to_sorting

__all__ = [
    'PackedSequence', 'pack_sequence',
    'pack_catted_indices', 'pack_catted_sequence',
    'pack_padded_indices', 'pack_padded_sequence',
]


def pack_sequence(sequences: List[Tensor], device: Device = None) -> PackedSequence:
    if device is None:
        device = sequences[0].device

    data, token_sizes = cat_sequence(sequences=sequences, device=device)
    return pack_catted_sequence(sequence=data, token_sizes=token_sizes, device=device)


@torch.no_grad()
def pack_catted_indices(token_sizes: Tensor, device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)

    token_sizes, sorted_indices, unsorted_indices = sizes_to_sorting(
        sizes=token_sizes, device=device,
    )
    token_ptr, batch_ptr, batch_sizes = token_sizes_to_ptr(
        token_sizes=token_sizes,
        batch_ptr=sorted_indices,
    )
    indices = acc_token_sizes[batch_ptr] + token_ptr

    return indices, batch_sizes, sorted_indices, unsorted_indices


def pack_catted_sequence(sequence: Tensor, token_sizes: Tensor, device: Device = None) -> PackedSequence:
    if device is None:
        device = sequence.device

    indices, batch_sizes, sorted_indices, unsorted_indices = pack_catted_indices(
        token_sizes=token_sizes, device=device,
    )

    return PackedSequence(
        data=sequence[indices],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


@torch.no_grad()
def pack_padded_indices(token_sizes: Tensor, batch_first: bool, device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)

    sorted_token_sizes, sorted_indices, unsorted_indices = sizes_to_sorting(
        sizes=token_sizes, device=device,
    )
    token_ptr, batch_ptr, batch_sizes = token_sizes_to_ptr(
        token_sizes=sorted_token_sizes,
        batch_ptr=sorted_indices,
    )

    if batch_first:
        indices = batch_ptr, token_ptr
    else:
        indices = token_ptr, batch_ptr
    return indices, batch_sizes, sorted_indices, unsorted_indices


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
