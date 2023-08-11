from functools import singledispatch
from typing import List
from typing import Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Number

from torchrua import get_device
from torchrua.catting import cat_sequence
from torchrua.core import CattedSequence
from torchrua.core import broadcast_devices
from torchrua.info import batch_sizes_to_major_ptr3
from torchrua.info import token_sizes_to_major_ptr2

Sequence = Union[CattedSequence, PackedSequence]

__all__ = [
    'pad_sequence', 'pad_indices',
    'pad_catted_indices', 'pad_catted_sequence',
    'pad_packed_indices', 'pad_packed_sequence',
]


@singledispatch
def pad_sequence(sequence: Union[List[Tensor], Sequence], batch_first: bool,
                 padding_value: Number = 0, device: torch.device = None):
    device = get_device(*sequence, device=device)

    sequence = cat_sequence(sequences=sequence, device=device)
    return pad_catted_sequence(
        sequence=sequence,
        batch_first=batch_first,
        padding_value=padding_value,
        device=device,
    )


def pad_indices(sequence: Sequence, batch_first: bool, device: torch.device = None):
    if isinstance(sequence, CattedSequence):
        return pad_catted_indices(
            token_sizes=sequence.token_sizes,
            batch_first=batch_first,
            device=device,
        )

    if isinstance(sequence, PackedSequence):
        return pad_packed_indices(
            batch_sizes=sequence.batch_sizes,
            sorted_indices=sequence.sorted_indices,
            unsorted_indices=sequence.unsorted_indices,
            batch_first=batch_first,
            device=device,
        )

    raise TypeError(f'type {type(sequence)} is not supported')


def pad_catted_indices(token_sizes: Tensor, batch_first: bool, device: torch.device = None):
    token_sizes, device = broadcast_devices(token_sizes, device=device)

    (b, t), (batch_ptr, token_ptr) = token_sizes_to_major_ptr2(token_sizes, device=device)

    if batch_first:
        return (b, t), (batch_ptr, token_ptr), token_sizes
    else:
        return (t, b), (token_ptr, batch_ptr), token_sizes


@pad_sequence.register
def pad_catted_sequence(sequence: CattedSequence, batch_first: bool,
                        padding_value: Number = 0, device: torch.device = None):
    sequence, token_sizes, device = broadcast_devices(*sequence, device=device)

    sizes, indices, token_sizes = pad_catted_indices(
        token_sizes=token_sizes,
        batch_first=batch_first,
        device=device,
    )

    data = torch.full(
        (*sizes, *sequence.size()[1:]),
        fill_value=padding_value, requires_grad=False,
        dtype=sequence.dtype, device=device,
    )
    data[indices] = sequence

    return data, token_sizes


def pad_packed_indices(batch_sizes: Tensor, sorted_indices: Tensor, unsorted_indices: Tensor,
                       batch_first: bool, device: torch.device = None):
    (b, t), (batch_ptr, token_ptr), token_sizes = batch_sizes_to_major_ptr3(
        batch_sizes=batch_sizes,
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
        device=device,
    )

    if batch_first:
        return (b, t), (batch_ptr, token_ptr), token_sizes
    else:
        return (t, b), (token_ptr, batch_ptr), token_sizes


@pad_sequence.register
def pad_packed_sequence(sequence: PackedSequence, batch_first: bool,
                        padding_value: Number = 0, device: torch.device = None):
    sequence, batch_sizes, sorted_indices, unsorted_indices, device = broadcast_devices(
        *sequence, device=device,
    )

    sizes, indices, token_sizes = pad_packed_indices(
        batch_sizes=batch_sizes,
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
        batch_first=batch_first,
        device=device,
    )

    data = torch.full(
        (*sizes, *sequence.size()[1:]),
        fill_value=padding_value, requires_grad=False,
        dtype=sequence.dtype, device=device,
    )
    data[indices] = sequence

    return data, token_sizes
