from typing import Tuple
from typing import Union

import torch
from torch.types import Number

from torchrua.catting import cat_sequence
from torchrua.core import broadcast_devices
from torchrua.info import batch_sizes_to_major_ptr3
from torchrua.info import token_sizes_to_major_ptr2
from torchrua.ty import C
from torchrua.ty import D
from torchrua.ty import P
from torchrua.ty import PaddedSequence
from torchrua.ty import T
from torchrua.ty import Ts
from torchrua.ty import is_type

__all__ = [
    'pad_sequence', 'pad_indices',
    'pad_catted_indices', 'pad_packed_indices',
]


def pad_sequence(sequence: Union[Ts, C, P], fill_value: Number = 0, device: torch.device = None) -> D:
    if is_type(sequence, Ts):
        sequence = cat_sequence(sequence, device=device)

    (b, t), (batch_ptr, token_ptr), token_sizes = sequence.ptr()

    tensor = sequence.data.new_full((b, t, *sequence.data.size()[1:]), fill_value=fill_value)
    tensor[batch_ptr, token_ptr] = sequence.data

    return PaddedSequence(data=tensor, token_sizes=token_sizes)


C.pad = pad_sequence
D.pad = D.to
P.pad = pad_sequence


def pad_indices(sequence: Union[C, D, P], device: torch.device = None) -> Tuple[Tuple[int, int], Tuple[T, T], T]:
    if is_type(sequence, Union[C, D]):
        return pad_catted_indices(
            token_sizes=sequence.token_sizes,
            device=device,
        )

    if is_type(sequence, P):
        return pad_packed_indices(
            batch_sizes=sequence.batch_sizes,
            sorted_indices=sequence.sorted_indices,
            unsorted_indices=sequence.unsorted_indices,
            device=device,
        )

    raise TypeError(f'type {type(sequence)} is not supported')


C.ptr = pad_indices
D.ptr = pad_indices
P.ptr = pad_indices


def pad_catted_indices(token_sizes: T, device: torch.device = None):
    token_sizes, device = broadcast_devices(token_sizes, device=device)

    (b, t), (batch_ptr, token_ptr) = token_sizes_to_major_ptr2(token_sizes, device=device)
    return (b, t), (batch_ptr, token_ptr), token_sizes


def pad_packed_indices(batch_sizes: T, sorted_indices: T, unsorted_indices: T, device: torch.device = None):
    (b, t), (batch_ptr, token_ptr), (_, token_sizes) = batch_sizes_to_major_ptr3(
        batch_sizes=batch_sizes,
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
        device=device,
    )

    return (b, t), (batch_ptr, token_ptr), token_sizes
