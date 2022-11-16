from functools import singledispatch
from typing import Tuple, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.catting import CattedSequence
from torchrua.core import major_sizes_to_size, major_sizes_to_ptr

__all__ = [
    'sequence_shape',
    'batch_first_filling_mask_shape',
    'catted_sequence_shape', 'packed_sequence_shape',

    'sequence_ptr',
    'batch_first_filling_mask_ptr',
    'catted_sequence_ptr', 'packed_sequence_ptr',
]

Sequence = Union[Tensor, CattedSequence, PackedSequence]

Shape = Tuple[int, int, int]


@singledispatch
def sequence_shape(sequence: Sequence, batch_first: bool = True) -> Shape:
    raise TypeError(f'type {type(sequence)} is not supported')


@sequence_shape.register
def batch_first_filling_mask_shape(sequence: Tensor, batch_first: bool = True) -> Shape:
    assert sequence.dtype is torch.bool
    assert sequence.dim() == 2

    b, t = sequence.size()
    n = sequence.long().sum().item()
    return (n, b, t) if batch_first else (n, t, b)


@sequence_shape.register
def catted_sequence_shape(sequence: CattedSequence, batch_first: bool = True) -> Shape:
    t, b = major_sizes_to_size(sizes=sequence.token_sizes)
    n, *_ = sequence.data.size()
    return (n, b, t) if batch_first else (n, t, b)


@sequence_shape.register
def packed_sequence_shape(sequence: PackedSequence, batch_first: bool = True) -> Shape:
    b, t = major_sizes_to_size(sizes=sequence.batch_sizes)
    n, *_ = sequence.data.size()
    return (n, b, t) if batch_first else (n, t, b)


Ptr = Tuple[Tensor, Tensor]


@singledispatch
def sequence_ptr(sequence: Sequence, batch_first: bool = True, device: Device = None) -> Ptr:
    raise TypeError(f'type {type(sequence)} is not supported')


@sequence_ptr.register
def batch_first_filling_mask_ptr(sequence: Tensor, batch_first: bool = True, device: Device = None) -> Ptr:
    assert sequence.dtype is torch.bool
    assert sequence.dim() == 2

    if device is None:
        device = sequence.device

    b, t = sequence.size()
    index1 = torch.arange(b, dtype=torch.long, device=device)
    index2 = torch.arange(t, dtype=torch.long, device=device)
    batch_ptr = torch.masked_select(index1[:, None], mask=sequence)
    token_ptr = torch.masked_select(index2[None, :], mask=sequence)
    return (batch_ptr, token_ptr) if batch_first else (token_ptr, batch_ptr)


@sequence_ptr.register
def catted_sequence_ptr(sequence: CattedSequence, batch_first: bool = True, device: Device = None) -> Ptr:
    if device is None:
        device = sequence.data.device

    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=sequence.token_sizes.to(device=device))
    return (batch_ptr, token_ptr) if batch_first else (token_ptr, batch_ptr)


@sequence_ptr.register
def packed_sequence_ptr(sequence: PackedSequence, batch_first: bool = True, device: Device = None) -> Ptr:
    if device is None:
        device = sequence.data.device

    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=sequence.batch_sizes.to(device=device))
    return (batch_ptr, token_ptr) if batch_first else (token_ptr, batch_ptr)
