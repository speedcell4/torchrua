from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.catting import cat_sequence
from torchrua.core import minor_sizes_to_ptr, accumulate_sizes, sizes_to_sorting, major_sizes_to_ptr

__all__ = [
    'PackedSequence', 'pack_sequence',
    'pack_catted_indices', 'pack_catted_sequence',
    'pack_padded_indices', 'pack_padded_sequence',
    'trunc_packed_indices', 'trunc_packed_sequence',
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
    token_ptr, batch_ptr, batch_sizes = minor_sizes_to_ptr(
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
    token_ptr, batch_ptr, batch_sizes = minor_sizes_to_ptr(
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


@torch.no_grad()
def trunc_packed_indices(batch_sizes: Tensor, trunc: Tuple[int, int], device: Device = None):
    if device is None:
        device = batch_sizes.device

    batch_sizes = batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

    batch_sizes = batch_sizes[trunc[0] + trunc[1]:]
    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=batch_sizes)

    indices = acc_batch_sizes[token_ptr + trunc[0]] + batch_ptr

    return indices, batch_sizes


def trunc_packed_sequence(sequence: PackedSequence, trunc: Tuple[int, int]) -> PackedSequence:
    indices, batch_sizes = trunc_packed_indices(
        batch_sizes=sequence.batch_sizes, trunc=trunc,
        device=sequence.data.device,
    )

    return PackedSequence(
        data=sequence.data[indices],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
    )
