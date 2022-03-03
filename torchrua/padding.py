from typing import Tuple, List

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device, Number

from torchrua.catting import cat_sequence, CattedSequence
from torchrua.core import major_sizes_to_ptr

__all__ = [
    'PaddedSequence', 'pad_sequence',
    'pad_catted_indices', 'pad_catted_sequence',
    'pad_packed_indices', 'pad_packed_sequence',
]

PaddedSequence = Tuple[Tensor, Tensor]


def pad_sequence(sequences: List[Tensor], batch_first: bool,
                 padding_value: Number = 0, device: Device = None) -> PaddedSequence:
    if device is None:
        device = sequences[0].device

    sequence = cat_sequence(sequences=sequences, device=device)
    return pad_catted_sequence(
        sequence=sequence, batch_first=batch_first,
        padding_value=padding_value, device=device,
    )


@torch.no_grad()
def pad_packed_indices(batch_sizes: Tensor, batch_first: bool,
                       sorted_indices: Tensor = None, unsorted_indices: Tensor = None, device: Device = None):
    if device is None:
        if unsorted_indices is not None:
            device = unsorted_indices.device
        elif sorted_indices is not None:
            device = sorted_indices.device
        else:
            device = batch_sizes.device

    batch_sizes = batch_sizes.to(device=device)
    b = batch_sizes.max().item()
    t, *_ = batch_sizes.size()

    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=batch_sizes)
    _, token_sizes = torch.unique(batch_ptr, sorted=True, return_counts=True)

    if sorted_indices is not None:
        batch_ptr = sorted_indices[batch_ptr]
    if unsorted_indices is not None:
        token_sizes = token_sizes[unsorted_indices]

    if batch_first:
        return (b, t), (batch_ptr, token_ptr), token_sizes
    else:
        return (t, b), (token_ptr, batch_ptr), token_sizes


def pad_packed_sequence(sequence: PackedSequence, batch_first: bool,
                        padding_value: Number = 0, device: Device = None) -> PaddedSequence:
    if device is None:
        device = sequence.data.device

    sizes, indices, token_sizes = pad_packed_indices(
        batch_sizes=sequence.batch_sizes,
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
        batch_first=batch_first, device=device,
    )

    data = torch.full(
        (*sizes, *sequence.data.size()[1:]), fill_value=padding_value,
        dtype=sequence.data.dtype, device=device, requires_grad=False,
    )
    data[indices] = sequence.data

    return data, token_sizes


@torch.no_grad()
def pad_catted_indices(token_sizes: Tensor, batch_first: bool, device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)
    b, *_ = token_sizes.size()
    t = token_sizes.max().item()

    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)

    if batch_first:
        return (b, t), (batch_ptr, token_ptr)
    else:
        return (t, b), (token_ptr, batch_ptr)


def pad_catted_sequence(sequence: CattedSequence, batch_first: bool,
                        padding_value: Number = 0, device: Device = None) -> PaddedSequence:
    if device is None:
        device = sequence.data.device

    token_sizes = sequence.token_sizes.to(device=device)

    sizes, indices = pad_catted_indices(
        token_sizes=token_sizes,
        batch_first=batch_first,
        device=device,
    )

    data = torch.full(
        (*sizes, *sequence.data.size()[1:]), fill_value=padding_value,
        dtype=sequence.data.dtype, device=device, requires_grad=False,
    )
    data[indices] = sequence.data

    return data, token_sizes
