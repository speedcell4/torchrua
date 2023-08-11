from functools import singledispatch
from typing import List
from typing import Tuple
from typing import Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device
from torch.types import Number

from torchrua.catting import cat_sequence
from torchrua.core import CattedSequence
from torchrua.core import major_masked_select
from torchrua.core import major_sizes_to_ptr
from torchrua.core import major_sizes_to_shape

__all__ = [
    'pad_sequence', 'pad_indices',
    'pad_catted_indices', 'pad_catted_sequence',
    'pad_packed_indices', 'pad_packed_sequence',
]


@singledispatch
def pad_sequence(sequence: Union[List[Tensor], CattedSequence, PackedSequence], batch_first: bool,
                 padding_value: Number = 0, device: Device = None) -> Tuple[Tensor, Tensor]:
    if device is None:
        device = sequence[0].device

    sequence = cat_sequence(sequences=sequence, device=device)
    return pad_catted_sequence(
        sequence=sequence, batch_first=batch_first,
        padding_value=padding_value, device=device,
    )


def pad_indices(sequence: Union[CattedSequence, PackedSequence], batch_first: bool, device: Device = None):
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


@torch.no_grad()
def pad_catted_indices(token_sizes: Tensor, batch_first: bool, device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)
    t, b = major_sizes_to_shape(sizes=token_sizes)

    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)

    if batch_first:
        return (b, t), (batch_ptr, token_ptr), token_sizes
    else:
        return (t, b), (token_ptr, batch_ptr), token_sizes


@pad_sequence.register
def pad_catted_sequence(sequence: CattedSequence, batch_first: bool,
                        padding_value: Number = 0, device: Device = None) -> Tuple[Tensor, Tensor]:
    if device is None:
        device = sequence.data.device

    sizes, indices, token_sizes = pad_catted_indices(
        token_sizes=sequence.token_sizes,
        batch_first=batch_first,
        device=device,
    )

    data = torch.full(
        (*sizes, *sequence.data.size()[1:]), fill_value=padding_value,
        dtype=sequence.data.dtype, device=device, requires_grad=False,
    )
    data[indices] = sequence.data

    return data, token_sizes


@torch.no_grad()
def pad_packed_indices(batch_sizes: Tensor, sorted_indices: Tensor, unsorted_indices: Tensor,
                       batch_first: bool, device: Device = None):
    if device is None:
        if unsorted_indices is not None:
            device = unsorted_indices.device
        elif sorted_indices is not None:
            device = sorted_indices.device
        elif batch_sizes is not None:
            device = batch_sizes.device
        else:
            raise RuntimeError('batch_sizes, sorted_indices, and unsorted_indices are all None')

    batch_sizes = batch_sizes.to(device=device)
    b, t = major_sizes_to_shape(sizes=batch_sizes)

    batch_ptr, token_ptr, token_sizes = major_masked_select(sizes=batch_sizes, device=device)

    if sorted_indices is not None:
        batch_ptr = sorted_indices[batch_ptr]
    if unsorted_indices is not None:
        token_sizes = token_sizes[unsorted_indices]

    if batch_first:
        return (b, t), (batch_ptr, token_ptr), token_sizes
    else:
        return (t, b), (token_ptr, batch_ptr), token_sizes


@pad_sequence.register
def pad_packed_sequence(sequence: PackedSequence, batch_first: bool,
                        padding_value: Number = 0, device: Device = None) -> Tuple[Tensor, Tensor]:
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
